import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode  # 移除CenterCrop
# pip install opencv-python
import cv2


class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        self.centercrop = centercrop  # 兼容参数，实际已按论文要求禁用
        self.size = size  # 固定为224，对应论文的224×224
        self.framerate = framerate  # 采样帧率，-1表示使用视频原始帧率
        self.transform = self._transform(self.size)  # 初始化图像变换（无中心裁剪）

    def _transform(self, n_px):
        """
        关键修改1：移除CenterCrop，直接resize到224×224（不保持纵横比）
        避免中心裁剪导致的边缘信息损失，完全符合论文描述
        """
        return Compose([
            # 直接将帧调整为(n_px, n_px)即224×224，InterpolationMode.BICUBIC保证 resize 质量
            Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC),
            lambda image: image.convert("RGB"),  # 确保图像为RGB通道
            ToTensor(),  # 转为Tensor格式（C×H×W）
            # 保持原标准化参数，与上游数据集类兼容
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None, _no_process=False):
        """
        关键修改2：实现TSN风格均匀采样
        TSN核心逻辑：将视频时间轴划分为等间隔段，每段取1帧，确保覆盖整个视频内容
        """
        # 时间范围参数校验（确保起始时间合法）
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time, "起始时间需为非负整数，且结束时间>起始时间"
        assert sample_fp > -1, "采样帧率（sample_fp）需≥0"

        # 初始化视频读取器
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{video_file}")
        
        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
        fps_video = int(cap.get(cv2.CAP_PROP_FPS))  # 视频原始帧率
        if fps_video == 0:
            raise ValueError(f"视频{video_file}帧率读取失败，无法进行采样")
        
        # 计算视频总时长（秒），处理帧数不能被帧率整除的情况
        total_duration = (frame_count_total + fps_video - 1) // fps_video
        # 确定采样的时间范围（默认全视频，否则用指定的start_time~end_time）
        start_sec = start_time if start_time is not None else 0
        end_sec = end_time if (end_time is not None and end_time <= total_duration) else total_duration
        # 计算采样时间范围内的总帧数
        frame_count_sample_range = int((end_sec - start_sec) * fps_video)
        if frame_count_sample_range <= 0:
            cap.release()
            raise ValueError(f"采样时间范围[{start_sec}, {end_sec}]内无有效帧，视频总时长仅{total_duration}秒")

        # -------------------------- TSN风格均匀采样核心逻辑 --------------------------
        # 1. 确定目标采样帧数：若sample_fp>0，按“采样帧率×时间范围”计算；否则取原始帧率对应的帧数
        if sample_fp > 0:
            target_sample_frames = sample_fp * (end_sec - start_sec)
        else:
            target_sample_frames = frame_count_sample_range  # 等同于使用原始帧率
        
        # 2. 生成均匀采样的帧索引（覆盖整个采样时间范围，避免局部采样偏差）
        # 从“起始时间对应的帧索引”到“结束时间对应的帧索引-1”均匀取target_sample_frames个帧
        start_frame_idx = int(start_sec * fps_video)
        end_frame_idx = int(end_sec * fps_video) - 1  # 避免超出时间范围的最后一帧
        if target_sample_frames >= (end_frame_idx - start_frame_idx + 1):
            # 若目标采样帧数≥可用帧数，取全部可用帧（避免重复采样）
            sample_frame_indices = np.arange(start_frame_idx, end_frame_idx + 1, dtype=int)
        else:
            # 若目标采样帧数<可用帧数，用linspace均匀划分并取整（TSN核心操作）
            sample_frame_indices = np.linspace(
                start_frame_idx, end_frame_idx, target_sample_frames, dtype=int
            )
        # -----------------------------------------------------------------------------

        # 读取采样帧并应用变换
        images = []
        for frame_idx in sample_frame_indices:
            # 定位到目标帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue  # 跳过读取失败的帧（避免程序崩溃）
            # CV2读取的帧为BGR格式，转为RGB（与PIL/模型输入格式一致）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 按需求选择是否应用预处理（_no_process用于原始帧保存场景）
            if _no_process:
                images.append(Image.fromarray(frame_rgb).convert("RGB"))
            else:
                images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
        
        # 释放视频读取器（避免资源泄漏）
        cap.release()

        # 处理输出格式：若有有效帧，转为Tensor；否则返回空Tensor（兼容上游错误处理）
        if len(images) > 0:
            if _no_process:
                video_data = images  # 原始PIL图像列表（_no_process=True时）
            else:
                video_data = th.tensor(np.stack(images))  # 转为Tensor（形状：[T, C, H, W]）
        else:
            video_data = th.zeros(1)  # 无有效帧时返回占位Tensor
        
        return {'video': video_data}

    def get_video_data(self, video_path, start_time=None, end_time=None, _no_process=False):
        """
        对外接口：获取处理后的视频数据
        调用video_to_tensor，传入初始化好的transform和采样帧率
        """
        image_input = self.video_to_tensor(
            video_path, 
            self.transform, 
            sample_fp=self.framerate,  # 传入采样帧率（从__init__参数获取）
            start_time=start_time, 
            end_time=end_time, 
            _no_process=_no_process
        )
        return image_input

    def process_raw_data(self, raw_video_data):
        """
        原始视频数据格式调整：将[T, C, H, W]转为[T, 1, C, H, W]
        适配部分模型的输入格式（增加一个“段”维度，兼容TSN的多段采样逻辑）
        """
        tensor_size = raw_video_data.size()
        # 处理单帧情况（避免维度错误）
        if len(tensor_size) == 3:
            tensor = raw_video_data.unsqueeze(0).unsqueeze(1)  # [1, 1, C, H, W]
        else:
            tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])  # [T, 1, C, H, W]
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        """
        帧顺序处理：支持正序、逆序、随机序（兼容数据增强需求）
        0: 正序（默认，符合TSN标准采样）；1: 逆序；2: 随机序
        """
        if frame_order == 0:
            pass  # 保持原顺序
        elif frame_order == 1:
            # 逆序：从最后一帧到第一帧
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            # 随机序：打乱帧顺序（用于数据增强）
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]
        return raw_video_data


# 对外暴露类别名，与上游数据集代码（RetrievalDataset）的导入逻辑兼容
RawVideoExtractor = RawVideoExtractorCV2