import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
import logging

logger = logging.getLogger(__name__)

def warmup_cosine(x, warmup):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + math.cos(math.pi * x))

def warmup_constant(x, warmup):
    """ Linearly increases learning rate over `warmup`*`t_total` (as provided to AdamW) training steps.
        Learning rate is 1. afterwards. """
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to AdamW) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x/warmup
    return max((x-1.)/(warmup-1.), 0)

SCHEDULES = {
    'warmup_cosine':   warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear':   warmup_linear,
}


class AdamW(Optimizer):
    """
    标准 AdamW 优化器实现（符合 PyTorch 原生 AdamW 逻辑）
    额外集成原 BertAdam 中的学习率调度和梯度裁剪功能
    
    核心特点：
    1. 权重衰减与梯度更新分离（标准 AdamW 行为）
    2. 支持三种学习率调度策略
    3. 可选梯度范数裁剪
    4. 包含 Adam 标准的偏差修正
    
    Params:
        lr: 基础学习率（必填）
        betas: Adam 动量参数 (beta1, beta2)，默认 (0.9, 0.98)
        eps: 数值稳定性epsilon，默认 1e-6
        weight_decay: 权重衰减系数，默认 0.01
        max_grad_norm: 梯度裁剪最大范数（-1 表示不裁剪），默认 1.0
        warmup: 预热占总步数的比例（-1 表示不预热），默认 -1
        t_total: 总训练步数（用于计算预热进度，-1 表示恒定学习率），默认 -1
        schedule: 学习率调度策略，可选 'warmup_cosine'/'warmup_constant'/'warmup_linear'，默认 'warmup_linear'
    """
    def __init__(
        self,
        params,
        lr=required,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup=-1,
        t_total=-1,
        schedule='warmup_linear'
    ):
        # 参数合法性校验
        if lr is not required and lr < 0.0:
            raise ValueError(f"无效学习率: {lr} - 必须 >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效 beta1 参数: {betas[0]} - 必须在 [0.0, 1.0) 区间")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效 beta2 参数: {betas[1]} - 必须在 [0.0, 1.0) 区间")
        if not eps >= 0.0:
            raise ValueError(f"无效 epsilon 值: {eps} - 必须 >= 0.0")
        if not 0.0 <= weight_decay:
            raise ValueError(f"无效权重衰减系数: {weight_decay} - 必须 >= 0.0")
        if schedule not in SCHEDULES:
            raise ValueError(f"无效调度策略: {schedule} - 可选值: {list(SCHEDULES.keys())}")
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError(f"无效预热比例: {warmup} - 必须在 [0.0, 1.0) 区间或设为 -1（不预热）")

        # 初始化优化器参数组
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            warmup=warmup,
            t_total=t_total,
            schedule=schedule
        )
        super(AdamW, self).__init__(params, defaults)

    def get_lr(self):
        """获取当前批次的学习率（用于日志或调试）"""
        current_lrs = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    current_lrs.append(0.0)
                    continue
                state = self.state[p]
                if len(state) == 0:
                    current_lrs.append(group['lr'])
                    continue
                # 计算调度后的学习率
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    progress = state['step'] / group['t_total']
                    lr_scheduled = group['lr'] * schedule_fct(progress, group['warmup'])
                else:
                    lr_scheduled = group['lr']
                current_lrs.append(lr_scheduled)
        return current_lrs

    def step(self, closure=None):
        """执行单步优化（核心逻辑）"""
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group['params']:
                # 跳过无梯度的参数
                if p.grad is None:
                    continue
                grad = p.grad.data
                # AdamW 不支持稀疏梯度（与 PyTorch 原生一致）
                if grad.is_sparse:
                    raise RuntimeError("AdamW 不支持稀疏梯度，请使用 SparseAdam 替代")

                # 初始化参数状态（step 计数、一阶动量、二阶动量）
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)  # 一阶动量（梯度移动平均）
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)  # 二阶动量（梯度平方移动平均）

                # 提取状态变量和组参数
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1  # 步数自增

                # 1. 更新一阶和二阶动量（带动量衰减）
                # 一阶动量：exp_avg = exp_avg * beta1 + grad * (1 - beta1)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 二阶动量：exp_avg_sq = exp_avg_sq * beta2 + grad² * (1 - beta2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 2. Adam 偏差修正（补偿初始动量为0的偏差）
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                sqrt_bias_correction2 = math.sqrt(bias_correction2)

                # 3. 计算学习率（应用调度策略）
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    progress = state['step'] / group['t_total']
                    lr_scheduled = group['lr'] * schedule_fct(progress, group['warmup'])
                else:
                    lr_scheduled = group['lr']

                # 4. 计算参数更新量（标准 AdamW 公式）
                step_size = lr_scheduled / bias_correction1
                denom = (exp_avg_sq.sqrt() / sqrt_bias_correction2) + group['eps']
                param_update = exp_avg / denom

                # 5. 更新参数（参数 = 参数 - 学习率 * 更新量）
                p.data.add_(param_update, alpha=-step_size)

                # 6. 独立应用权重衰减（AdamW 核心：与梯度更新分离）
                if group['weight_decay'] > 0.0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * lr_scheduled)

        return loss