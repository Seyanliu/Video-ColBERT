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
    
    parser.add_argument('--clip_lr', type=float, default=1e-7, help='initial learning rate')
    parser.add_argument('--noclip_lr', type=float, default=1e-4, help='coefficient for bert branch.')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=128, help='batch size eval')

    parser.add_argument('--max_words', type=int, default=32, help='max text token number')
    parser.add_argument('--max_frames', type=int, default=12, help='max key frames')
    parser.add_argument('--max_words', type=int, default=32, help='max text token number')
    parser.add_argument('--max_frames', type=int, default=12, help='max key frames')
    parser.add_argument('--video_framerate', type=int, default=1, help='framerate to sample video frame')

    parser.add_argument("--device", default='cpu', type=str, help="cpu/cuda")
    parser.add_argument("--world_size", default=1, type=int, help="distributed training")
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
    
    parser.add_argument('--lambda_f', type=float, default=1.0, help="帧级损失权重")
    parser.add_argument('--lambda_v', type=float, default=1.0, help="视频级损失权重")

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")

    args = parser.parse_args()

    return args


def set_seed_logger(args):
    global logger
    import torch.distributed as dist

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.local_rank,
        )
        dist.barrier()
        logger.info(f"[Rank {args.local_rank}] Using GPU {torch.cuda.current_device()}")
    else:
        args.device = torch.device("cpu")
        args.world_size = 1

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
    clip_lr = args.clip_lr
    noclip_lr = args.noclip_lr
    weight_decay = args.weight_decay
    warmup_proportion = args.warmup_proportion
    param_optimizer = list(model.named_parameters())

    decay_clip_param_tp = [(n, p) for n, p in param_optimizer if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in param_optimizer if "clip." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': clip_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay, 'lr': noclip_lr},
    ]

    scheduler = None
    optimizer = AdamW(optimizer_grouped_parameters, warmup=warmup_proportion,
                         schedule='warmup_linear', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=True)

    return optimizer, scheduler, model


def save_model(epoch, args, model, type_name=""):
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
            loss /= world_size
    return loss


def train_epoch(epoch, args, model, train_dataloader, val_dataloader, device, n_gpu, optimizer, scheduler, global_step, max_steps, best_score):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    total_loss = 0.0

    meters = MetricLogger(delimiter="  ")
    end = time.time()

    # 计算当前epoch内10个均匀分布的测评节点
    total_steps_in_epoch = len(train_dataloader)
    eval_steps = [int(total_steps_in_epoch * (i / 10)) for i in range(1, 11)]
    eval_step_idx = 0

    for step, batch in enumerate(train_dataloader, start=1):
        current_epoch_step = step
        global_step += 1
        data_time = time.time() - end

        # 数据迁移与模型前向计算
        if n_gpu == 1:
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        text_ids, text_mask, video, video_mask, inds = batch
        loss, loss_f, loss_v = model(text_ids, text_mask, video, video_mask)

        # 多GPU损失平均
        if n_gpu > 1:
            loss = loss.mean()
            loss_f = loss_f.mean()
            loss_v = loss_v.mean()

        # 反向传播与参数更新
        with torch.autograd.detect_anomaly():
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scheduler is not None:
            scheduler.step()
        optimizer.step()
        optimizer.zero_grad()

        # 日志记录
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

        # 到达测评节点，执行中间测评并更新最优模型
        if eval_step_idx < 10 and current_epoch_step == eval_steps[eval_step_idx]:
            logger.info(f"\n===== Epoch {epoch} - 第 {eval_step_idx+1}/10 次中间测评（步骤 {current_epoch_step}/{total_steps_in_epoch}）=====")
            model.eval()
            current_R1 = eval_epoch(args, model, val_dataloader, device)
            
            # 主进程更新best.pth
            if args.local_rank == 0 and current_R1 > best_score:
                best_score = current_R1
                best_model_path = os.path.join(args.output_dir, 'best.pth')
                best_model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(best_model_state, best_model_path)
                logger.info(f"===== 中间测评最优更新！当前best R1: {best_score:.4f}，保存路径：{best_model_path} =====")
            
            model.train()
            eval_step_idx += 1
            torch.cuda.empty_cache()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step, best_score


def _run_on_single_gpu(model, t_mask_list, v_mask_list, t_feat_list, f_feat_list, v_feat_list, mini_batch=32):
    sim_matrix = []
    logger.info('[start] map to main gpu')
    
    batch_t_mask = torch.split(t_mask_list, mini_batch)
    batch_v_mask = torch.split(v_mask_list, mini_batch)
    batch_t_feat = torch.split(t_feat_list, mini_batch)
    batch_f_feat = torch.split(f_feat_list, mini_batch)
    batch_v_feat = torch.split(v_feat_list, mini_batch)

    logger.info('[finish] map to main gpu')
    with torch.no_grad():
        for idx1, t_feat in enumerate(batch_t_feat):
            each_row = []
            for idx2, (f_feat, v_feat) in enumerate(zip(batch_f_feat, batch_v_feat)):
                b1b2_logits, *_tmp = model.get_similarity_logits(t_feat, f_feat, v_feat, shaped=True)
                b1b2_logits = b1b2_logits.cpu().detach().numpy()
                each_row.append(b1b2_logits)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row)
    return sim_matrix


def eval_epoch(args, model, test_dataloader, device):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # 多句子检索相关变量初始化
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    
    if isinstance(test_dataloader, list) and hasattr(test_dataloader[0].dataset, 'multi_sentence_per_video') \
            and test_dataloader[0].dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader[0].dataset.cut_off_points
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]
        sentence_num_ = test_dataloader[0].dataset.get_text_len()
        video_num_ = test_dataloader[0].dataset.get_video_len()

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_f, batch_feat_v, ids_t, ids_v = [], [], [], [], [], [], []

    with torch.no_grad():
        tic = time.time()
        
        if multi_sentence_:
            # 提取文本特征
            logger.info('[start] extract text feature')
            for batch in tqdm(test_dataloader[0]):
                batch = tuple(t.to(device) for t in batch)
                text_ids, text_mask, inds = batch
                sequence_output = model.get_sequence_output(text_ids, text_mask)
                ids_t.append(inds)
                batch_mask_t.append(text_mask)
                batch_feat_t.append(sequence_output)
            
            # 分布式特征聚集与排序
            ids_t = allgather(torch.cat(ids_t, dim=0), args)
            batch_feat_t = allgather(torch.cat(batch_feat_t, dim=0), args)
            batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
            batch_feat_t[ids_t] = batch_feat_t.clone()
            batch_mask_t[ids_t] = batch_mask_t.clone()
            batch_feat_t = batch_feat_t[:ids_t.max() + 1, ...]
            batch_mask_t = batch_mask_t[:ids_t.max() + 1, ...]
            logger.info('[finish] extract text feature')

            # 提取视频特征
            logger.info('[start] extract video feature')
            for batch in tqdm(test_dataloader[1]):
                batch = tuple(t.to(device) for t in batch)
                video, video_mask, inds = batch
                video_feat = model.get_video_feat(video, video_mask)
                ids_v.append(inds)
                batch_mask_v.append(video_mask)
                batch_feat_v.append(video_feat)
            
            # 分布式特征聚集与排序
            ids_v = allgather(torch.cat(ids_v, dim=0), args)
            batch_feat_v = allgather(torch.cat(batch_feat_v, dim=0), args)
            batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)
            batch_feat_v[ids_v] = batch_feat_v.clone()
            batch_mask_v[ids_v] = batch_mask_v.clone()
            batch_feat_v = batch_feat_v[:ids_v.max() + 1, ...]
            batch_mask_v = batch_mask_v[:ids_v.max() + 1, ...]
            logger.info('[finish] extract video feature')
        else:
            # 同时提取文本与视频特征
            logger.info('[start] extract text+video feature')
            for batch in tqdm(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                text_ids, text_mask, video, video_mask, inds = batch
                text_feat, frame_feat, video_feat = model.get_text_video_feat(text_ids, text_mask, video, video_mask)
                ids_t.append(inds)
                batch_mask_t.append(text_mask)
                batch_mask_v.append(video_mask)
                batch_feat_t.append(text_feat)
                batch_feat_f.append(frame_feat)
                batch_feat_v.append(video_feat)
            
            # 分布式特征聚集与排序
            ids_t = allgather(torch.cat(ids_t, dim=0), args).squeeze()
            batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
            batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)
            batch_feat_t = allgather(torch.cat(batch_feat_t, dim=0), args)
            batch_feat_f = allgather(torch.cat(batch_feat_f, dim=0), args)
            batch_feat_v = allgather(torch.cat(batch_feat_v, dim=0), args)
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
    
    # 计算相似度矩阵
    logger.info('[start] calculate the similarity')
    with torch.no_grad():
        sim_matrix = _run_on_single_gpu(model, batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_f, batch_feat_v)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        sim_matrix_sorted = -np.sort(-sim_matrix, axis=1)
    logger.info('[end] calculate the similarity')

    toc2 = time.time()
    logger.info('[start] compute_metrics')
    
    # 计算检索指标
    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)),
                                                 axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))
        
        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))
    logger.info('[end] compute_metrics')

    toc3 = time.time()
    logger.info("time profile: feat {:.1f}s match {:.5f}s metrics {:.5f}s".format(toc1 - tic, toc2 - toc1, toc3 - toc2))

    # 输出指标日志
    logger.info("Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".format(
        vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    return tv_metrics['R1']


def main():
    global logger
    args = get_args()
    if not exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('tvr', args.output_dir, args.local_rank)

    args = set_seed_logger(args)
    model = build_model(args)
    test_dataloader, val_dataloader, train_dataloader, train_sampler = build_dataloader(args)

    if args.do_train:
        tic = time.time()
        max_steps = len(train_dataloader) * args.epochs
        optimizer, scheduler, model = prep_optimizer(args, model, max_steps, args.local_rank)

        best_score = 0.00001
        global_step = 0

        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            synchronize()
            torch.cuda.empty_cache()

            # 训练epoch（含10次中间测评）
            tr_loss, global_step, best_score = train_epoch(
                epoch=epoch,
                args=args,
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                device=args.device,
                n_gpu=args.world_size,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                max_steps=max_steps,
                best_score=best_score
            )

            torch.cuda.empty_cache()
            synchronize()

            # 每个epoch结束后强制保存当前参数
            if args.local_rank == 0:
                epoch_model_path = os.path.join(args.output_dir, f'epoch_{epoch}.pth')
                epoch_model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(epoch_model_state, epoch_model_path)
                logger.info(f"\n===== Epoch {epoch} 训练结束！已保存epoch参数至：{epoch_model_path} =====")
                logger.info(f"当前全局最优R1：{best_score:.4f}，最优模型路径：{os.path.join(args.output_dir, 'best.pth')}")

        # 训练结束后用最优模型测试
        toc = time.time() - tic
        training_time = time.strftime("%Hh %Mmin %Ss", time.gmtime(toc))
        logger.info("*" * 20 + '\n' + f'training finished with {training_time}' + "*" * 20 + '\n')

        model = model.module
        if args.local_rank == 0:
            best_model_path = os.path.join(args.output_dir, 'best.pth')
            model.load_state_dict(torch.load(best_model_path, map_location='cpu'), strict=False)
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