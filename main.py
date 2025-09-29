from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from os.path import join, exists

import torch

from tvr.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from tvr.dataloaders.data_dataloaders import DATALOADER_DICT
from tvr.models.modeling import VideoColBERT, AllGather
from tvr.models.optimization import AdamW, BertAdam
from tvr.utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

from tvr.utils.comm import is_main_process, synchronize
from tvr.utils.logger import setup_logger
from tvr.utils.metric_logger import MetricLogger

allgather = AllGather.apply

global logger


def get_args(description='Disentangled Representation Learning for Text-Video Retrieval'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", type=int, default=0, help="Whether to run training.")
    parser.add_argument("--do_eval", type=int, default=0, help="Whether to run evaluation.")

    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument('--anno_path', type=str, default='data/MSR-VTT/anns', help='annotation path')
    parser.add_argument('--video_path', type=str, default='data/MSR-VTT/videos', help='video path')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--coef_lr', type=float, default=1e-3, help='coefficient for bert branch.')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=128, help='batch size eval')

    parser.add_argument('--max_words', type=int, default=32, help='max text token number')  # 文本最大 token 数
    parser.add_argument('--max_frames', type=int, default=12, help='max key frames')  # 视频关键帧最大数量
    parser.add_argument('--video_framerate', type=int, default=1, help='framerate to sample video frame')  # 视频帧采样率

    parser.add_argument("--device", default='cpu', type=str, help="cpu/cuda")
    parser.add_argument("--world_size", default=1, type=int, help="distributed training")
    # 修改这里：默认从环境变量 LOCAL_RANK 读取
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)),
                        help="distributed training local rank")
    parser.add_argument("--distributed", default=0, type=int, help="multi machine DDP")

    parser.add_argument('--n_display', type=int, default=100, help='Information display frequency')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--base_encoder", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument('--agg_module', type=str, default="seqTransf", choices=["None", "seqLSTM", "seqTransf"],
                        help="choice a feature aggregation module for video.")
    parser.add_argument('--interaction', type=str, default='mms', help="interaction type for retrieval.")

    parser.add_argument('--use_query_expansion', action='store_true', help="是否在VideoColBERT中使用查询扩展")
    parser.add_argument('--query_pad_length', type=int, default=32, help="VideoColBERT查询填充长度")
    parser.add_argument('--visual_expansion_tokens', type=int, default=2, help="视觉扩展token数量")
    parser.add_argument('--temporal_layers', type=int, default=4, help="时间编码器层数")
    parser.add_argument('--temporal_heads', type=int, default=8, help="时间编码器注意力头数")
    parser.add_argument('--temporal_ff_dim', type=int, default=2048, help="时间编码器前馈维度")
    parser.add_argument('--temporal_dropout', type=float, default=0.1, help="时间编码器dropout率")
    
    # ===== VideoColBERT 损失函数参数 =====
    parser.add_argument('--lambda_f', type=float, default=1.0, help="帧级损失权重")
    parser.add_argument('--lambda_v', type=float, default=1.0, help="视频级损失权重")

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")

    args = parser.parse_args()

    return args


def set_seed_logger(args):
    global logger
    import torch.distributed as dist

    # 固定随机种子
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 初始化分布式环境
    if torch.cuda.is_available():
        # 先绑定本地 GPU，防止 NCCL 警告
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        # world_size 默认从环境变量获取
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))
        # 初始化分布式进程组，环境变量管理rank和world_size
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.local_rank,
        )
        # 同步所有进程
        dist.barrier()
        logger.info(f"[Rank {args.local_rank}] Using GPU {torch.cuda.current_device()}")
    else:
        args.device = torch.device("cpu")
        args.world_size = 1

    # 检查 batch_size 是否能被 world_size 整除
    if args.batch_size % args.world_size != 0 or args.batch_size_val % args.world_size != 0:
        raise ValueError(
            f"Invalid batch_size/batch_size_val and world_size parameter: "
            f"{args.batch_size}%{args.world_size} and {args.batch_size_val}%{args.world_size} should be 0."
        )

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info(f"  <<< {key}: {args.__dict__[key]}")

    return args


def build_model(args):
    model = VideoColBERT(args)
    if args.init_model:
        if not exists(args.init_model):
            raise FileNotFoundError
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)

    model.to(args.device)
    return model


def build_dataloader(args):
    ## ####################################
    # dataloader loading
    ## ####################################
    tokenizer = ClipTokenizer()
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if isinstance(test_length, int):
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
    elif len(test_length) == 2:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %dt %dv", test_length[0], test_length[1])
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d %d", len(test_dataloader[0]), len(test_dataloader[1]))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %dt %dv", val_length[0], val_length[1])

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", len(train_dataloader) * args.epochs)
    else:
        train_dataloader, train_sampler = None, None

    return test_dataloader, val_dataloader, train_dataloader, train_sampler


def prep_optimizer(args, model, num_train_optimization_steps, local_rank):
    if hasattr(model, 'module'):
        model = model.module
    lr = args.lr  # 0.0001
    coef_lr = args.coef_lr  # 0.001
    weight_decay = args.weight_decay  # 0.2
    warmup_proportion = args.warmup_proportion
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=warmup_proportion,
                         schedule='warmup_linear', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=True)

    return optimizer, scheduler, model


def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file


def reduce_loss(loss, args):
    world_size = args.world_size
    if world_size < 2:
        return loss
    with torch.no_grad():
        torch.distributed.reduce(loss, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    return loss


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, max_steps):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    total_loss = 0

    meters = MetricLogger(delimiter="  ")
    end = time.time()

    for step, batch in enumerate(train_dataloader, start=1):
        global_step += 1
        data_time = time.time() - end

        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        text_ids, text_mask, video, video_mask, inds = batch
        loss, loss_f, loss_v = model(text_ids, text_mask, video, video_mask)

        if n_gpu > 1:
            # print(loss.shape)
            loss = loss.mean()  # mean() to average on multi-gpu.
            loss_f = loss_f.mean()
            loss_v = loss_v.mean()

        with torch.autograd.detect_anomaly():
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scheduler is not None:
            scheduler.step()  # Update learning rate schedule

        optimizer.step()
        optimizer.zero_grad()

        # https://github.com/openai/CLIP/issues/46

        batch_time = time.time() - end
        end = time.time()

        reduced_l = reduce_loss(loss, args)
        reduced_l_f = reduce_loss(loss_f, args)
        reduced_l_v = reduce_loss(loss_v, args)
        meters.update(time=batch_time, data=data_time, loss=float(reduced_l), loss_f=float(reduced_l_f), loss_v=float(reduced_l_v))

        eta_seconds = meters.time.global_avg * (max_steps - global_step)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        real_model = getattr(model, 'module', model)
        t = real_model.loss.temperature.item()
        b = real_model.loss.bias.item()

        if global_step % log_step == 0 and is_main_process():
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "epoch: {epoch}/{max_epoch}",
                        "iteration: {iteration}/{max_iteration}",
                        "{meters}",
                        "lr: {lr}",
                        "t: {t:.4f}",
                        "b: {b:.4f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch,
                    max_epoch=args.epochs,
                    iteration=global_step,
                    max_iteration=max_steps,
                    meters=str(meters),
                    lr="/".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                    t=t,
                    b=b,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(model, t_mask_list, v_mask_list, t_feat_list, f_feat_list, v_feat_list, mini_batch=32):
    sim_matrix = []
    logger.info('[start] map to main gpu')
    
    # 将大批量数据分割成小批次，避免内存溢出
    batch_t_mask = torch.split(t_mask_list, mini_batch)  # 分割文本掩码
    batch_v_mask = torch.split(v_mask_list, mini_batch)  # 分割视频掩码
    batch_t_feat = torch.split(t_feat_list, mini_batch)  # 分割文本特征
    batch_f_feat = torch.split(f_feat_list, mini_batch)  # 分割帧特征
    batch_v_feat = torch.split(v_feat_list, mini_batch)  # 分割视频特征

    logger.info('[finish] map to main gpu')
    with torch.no_grad():  # 禁用梯度计算以节省内存
        # 外层循环：遍历每个文本批次
        for idx1, t_feat in enumerate(batch_t_feat):
            each_row = []  # 存储当前文本批次与所有视频的相似度
            # 内层遍历所有 video chunks，同时取得对应的 frame_chunk 与 video_chunk
            for idx2, (f_feat, v_feat) in enumerate(zip(batch_f_feat, batch_v_feat)):
                # t_feat: [b_t, Q, D], f_feat: [b_v, F, D], v_feat: [b_v, F+exp, D]
                # shaped=True 因为输入已经是 token/frame shaped
                b1b2_logits, *_tmp = model.get_similarity_logits(t_feat, f_feat, v_feat, shaped=True)
                # b1b2_logits should be [b_t, b_v]
                b1b2_logits = b1b2_logits.cpu().detach().numpy()
                each_row.append(b1b2_logits)
            # 把当前 text-chunk 对应的所有 video-chunks 横向拼接成一行
            each_row = np.concatenate(tuple(each_row), axis=-1)  # -> [b_t, total_num_videos]
            sim_matrix.append(each_row)
    return sim_matrix


def eval_epoch(args, model, test_dataloader, device):
    """
    评估模型在测试集上的性能
    
    该函数执行完整的评估流程：
    1. 提取所有文本和视频的特征表示
    2. 计算文本-视频相似度矩阵
    3. 基于相似度矩阵计算检索指标（R@1, R@5, R@10等）
    
    参数:
        args: 命令行参数配置
        model: 要评估的模型
        test_dataloader: 测试数据加载器
        device: 计算设备(CPU/GPU)
        
    返回:
        R1: Text-to-Video检索的R@1指标值
    """
    # 确保模型在正确的设备上
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## 以下变量用于多句子检索任务
    # multi_sentence_: 是否为多句子检索的重要标识
    # cut_off_points: 计算指标时用于标记标签的切分点
    # sentence_num: 用于切分句子表示
    # video_num: 用于切分视频表示
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    
    # 检查是否为多句子检索设置（一个视频对应多个描述）
    if isinstance(test_dataloader, list) and hasattr(test_dataloader[0].dataset, 'multi_sentence_per_video') \
            and test_dataloader[0].dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader[0].dataset.cut_off_points
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]  # 转换为0索引
        sentence_num_ = test_dataloader[0].dataset.get_text_len()
        video_num_ = test_dataloader[0].dataset.get_video_len()

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()  # 设置模型为评估模式
    
    # ----------------------------
    # 1. 缓存特征表示
    # ----------------------------
    batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_f, batch_feat_v, ids_t, ids_v = [], [], [], [], [], [], []

    with torch.no_grad():  # 禁用梯度计算
        tic = time.time()
        
        if multi_sentence_:  # 多句子检索：一个视频片段有两个或更多描述
            # ========== 提取文本特征 ==========
            logger.info('[start] extract text feature')
            for batch in tqdm(test_dataloader[0]):  # 文本数据加载器
                batch = tuple(t.to(device) for t in batch)
                text_ids, text_mask, inds = batch
                # 获取文本序列特征表示
                sequence_output = model.get_sequence_output(text_ids, text_mask)
                ids_t.append(inds)
                batch_mask_t.append(text_mask)
                batch_feat_t.append(sequence_output)
            
            # 聚集分布式训练中各GPU的结果
            ids_t = allgather(torch.cat(ids_t, dim=0), args)
            batch_feat_t = allgather(torch.cat(batch_feat_t, dim=0), args)
            batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
            
            # 根据索引重新排序，确保数据顺序正确
            batch_feat_t[ids_t] = batch_feat_t.clone()
            batch_mask_t[ids_t] = batch_mask_t.clone()
            batch_feat_t = batch_feat_t[:ids_t.max() + 1, ...]
            batch_mask_t = batch_mask_t[:ids_t.max() + 1, ...]
            logger.info('[finish] extract text feature')

            # ========== 提取视频特征 ==========
            logger.info('[start] extract video feature')
            for batch in tqdm(test_dataloader[1]):  # 视频数据加载器
                batch = tuple(t.to(device) for t in batch)
                video, video_mask, inds = batch
                # 获取视频特征表示
                video_feat = model.get_video_feat(video, video_mask)
                ids_v.append(inds)
                batch_mask_v.append(video_mask)
                batch_feat_v.append(video_feat)
            
            # 聚集分布式训练中各GPU的结果
            ids_v = allgather(torch.cat(ids_v, dim=0), args)
            batch_feat_v = allgather(torch.cat(batch_feat_v, dim=0), args)
            batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)
            
            # 根据索引重新排序
            batch_feat_v[ids_v] = batch_feat_v.clone()
            batch_mask_v[ids_v] = batch_mask_v.clone()
            batch_feat_v = batch_feat_v[:ids_v.max() + 1, ...]
            batch_mask_v = batch_mask_v[:ids_v.max() + 1, ...]
            logger.info('[finish] extract video feature')
        else:
            # ========== 同时提取文本和视频特征 ==========
            logger.info('[start] extract text+video feature')
            for batch in tqdm(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                text_ids, text_mask, video, video_mask, inds = batch
                # 同时获取文本和视频特征
                text_feat, frame_feat, video_feat = model.get_text_video_feat(text_ids, text_mask, video, video_mask)
                ids_t.append(inds)
                batch_mask_t.append(text_mask)
                batch_mask_v.append(video_mask)
                batch_feat_t.append(text_feat)
                batch_feat_f.append(frame_feat)
                batch_feat_v.append(video_feat)
            
            # 聚集分布式训练中各GPU的结果
            ids_t = allgather(torch.cat(ids_t, dim=0), args).squeeze()
            batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
            batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)
            batch_feat_t = allgather(torch.cat(batch_feat_t, dim=0), args)
            batch_feat_f = allgather(torch.cat(batch_feat_f, dim=0), args)
            batch_feat_v = allgather(torch.cat(batch_feat_v, dim=0), args)
            
            # 根据索引重新排序，确保数据顺序正确
            batch_mask_t[ids_t] = batch_mask_t.clone()
            batch_mask_v[ids_t] = batch_mask_v.clone()
            batch_feat_t[ids_t] = batch_feat_t.clone()
            batch_feat_f[ids_t] = batch_feat_f.clone()
            batch_feat_v[ids_t] = batch_feat_v.clone()
            batch_mask_t = batch_mask_t[:ids_t.max() + 1, ...]
            batch_mask_v = batch_mask_v[:ids_t.max() + 1, ...]
            batch_feat_t = batch_feat_t[:ids_t.max() + 1, ...]
            batch_feat_f = batch_feat_f[:ids_t.max() + 1, ...]
            batch_feat_v = batch_feat_v[:ids_t.max() + 1, ...]
            logger.info('[finish] extract text+video feature')

    toc1 = time.time()

    logger.info('{} {} {} {}'.format(len(batch_mask_t), len(batch_mask_v), len(batch_feat_t), len(batch_feat_v)))
    
    # ----------------------------------
    # 2. 计算相似度矩阵
    # ----------------------------------
    logger.info('[start] calculate the similarity')
    with torch.no_grad():
        # 调用单GPU函数计算所有文本-视频对的相似度
        sim_matrix = _run_on_single_gpu(model, batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_f, batch_feat_v)
        # 将列表形式的相似度矩阵拼接成完整的numpy数组
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        sim_matrix_sorted = -np.sort(-sim_matrix, axis=1)
    logger.info('[end] calculate the similarity')

    toc2 = time.time()
    logger.info('[start] compute_metrics')
    
    # ----------------------------------
    # 3. 计算检索指标
    # ----------------------------------
    if multi_sentence_:
        # 多句子检索：需要重新整理相似度矩阵的形状
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        # 计算最大句子数量，用于填充
        max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        
        # 为每个视频片段重新组织其对应的多个句子
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            # 对不足最大长度的部分用-inf填充
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)),
                                                 axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))
        
        # 计算多句子检索的指标
        tv_metrics = tensor_text_to_video_metrics(sim_matrix)  # Text-to-Video指标
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))  # Video-to-Text指标
    else:
        # 标准检索：直接计算指标
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)      # Text-to-Video指标
        vt_metrics = compute_metrics(sim_matrix.T)    # Video-to-Text指标（转置矩阵）
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))
    logger.info('[end] compute_metrics')

    toc3 = time.time()
    # 记录各阶段耗时
    logger.info("time profile: feat {:.1f}s match {:.5f}s metrics {:.5f}s".format(toc1 - tic, toc2 - toc1, toc3 - toc2))

    # 输出最终的检索性能指标
    logger.info("Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".format(
        vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    return tv_metrics['R1']  # 返回Text-to-Video的R@1作为主要评估指标


def main():
    global logger
    args = get_args()
    if not exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('tvr', args.output_dir, args.local_rank)

    args = set_seed_logger(args)

    model = build_model(args)

    test_dataloader, val_dataloader, train_dataloader, train_sampler = build_dataloader(args)

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        tic = time.time()
        max_steps = len(train_dataloader) * args.epochs
        optimizer, scheduler, model = prep_optimizer(args, model, max_steps, args.local_rank)

        best_score = 0.00001
        best_output_model_file = "None"
        global_step = 0
        for epoch in range(args.epochs):
            if train_sampler is not None: train_sampler.set_epoch(epoch)
            synchronize()
            torch.cuda.empty_cache()
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader,
                                               args.device, args.world_size, optimizer,
                                               scheduler, global_step, max_steps)
            torch.cuda.empty_cache()
            R1 = eval_epoch(args, model, val_dataloader, args.device)
            torch.cuda.empty_cache()
            synchronize()

            if args.local_rank == 0:
                output_model_file = save_model(epoch, args, model, type_name="")

                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                    torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                               'best.pth')
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))
            synchronize()
        toc = time.time() - tic
        training_time = time.strftime("%Hh %Mmin %Ss", time.gmtime(toc))
        logger.info("*" * 20 + '\n' + f'training finished with {training_time}' + "*" * 20 + '\n')

        # test on the best checkpoint
        model = model.module
        if args.local_rank == 0:
            model.load_state_dict(torch.load('best.pth', map_location='cpu'), strict=False)
        if torch.cuda.is_available():
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              find_unused_parameters=True)

        torch.cuda.empty_cache()
        eval_epoch(args, model, test_dataloader, args.device)
        synchronize()

    elif args.do_eval:
        eval_epoch(args, model, test_dataloader, args.device)


if __name__ == "__main__":
    main()
