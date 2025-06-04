# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math
from models.vc.flow_matching_transformer.llama_nar import DiffLlama
import torch.nn.functional as F


class FlowMatchingTransformer(nn.Module):
    def __init__(
        self,
        mel_dim=100,
        hidden_size=1024,
        num_layers=12,
        num_heads=16,
        cfg_scale=0.2,
        use_cond_code=True,
        cond_codebook_size=1024,
        cond_dim=1024,
        cond_scale_factor=1,
        sigma=1e-5,
        time_scheduler="linear",
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg

        mel_dim = (
            cfg.mel_dim if cfg is not None and hasattr(cfg, "mel_dim") else mel_dim
        )
        hidden_size = (
            cfg.hidden_size
            if cfg is not None and hasattr(cfg, "hidden_size")
            else hidden_size
        )
        num_layers = (
            cfg.num_layers
            if cfg is not None and hasattr(cfg, "num_layers")
            else num_layers
        )
        num_heads = (
            cfg.num_heads
            if cfg is not None and hasattr(cfg, "num_heads")
            else num_heads
        )
        cfg_scale = (
            cfg.cfg_scale
            if cfg is not None and hasattr(cfg, "cfg_scale")
            else cfg_scale
        )
        use_cond_code = (
            cfg.use_cond_code
            if cfg is not None and hasattr(cfg, "use_cond_code")
            else use_cond_code
        )
        cond_codebook_size = (
            cfg.cond_codebook_size
            if cfg is not None and hasattr(cfg, "cond_codebook_size")
            else cond_codebook_size
        )
        cond_dim = (
            cfg.cond_dim if cfg is not None and hasattr(cfg, "cond_dim") else cond_dim
        )
        time_scheduler = (
            cfg.time_scheduler
            if cfg is not None and hasattr(cfg, "time_scheduler")
            else time_scheduler
        )
        sigma = cfg.sigma if cfg is not None and hasattr(cfg, "sigma") else sigma
        cond_scale_factor = (
            cfg.cond_scale_factor
            if cfg is not None and hasattr(cfg, "cond_scale_factor")
            else cond_scale_factor
        )

        self.mel_dim = mel_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cfg_scale = cfg_scale
        self.use_cond_code = use_cond_code
        self.cond_codebook_size = cond_codebook_size
        self.cond_dim = cond_dim
        self.time_scheduler = time_scheduler
        self.sigma = sigma
        self.cond_scale_factor = cond_scale_factor

        if self.use_cond_code:
            self.cond_emb = nn.Embedding(cond_codebook_size, self.hidden_size)
        else:
            self.cond_emb = nn.Linear(self.cond_dim, self.hidden_size)

        self.reset_parameters()

        self.diff_estimator = DiffLlama(
            mel_dim=mel_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        self.sigma = sigma

    @torch.no_grad()
    def forward_diffusion(self, x, t):
        """
        x: (B, T, mel_dim)
        t: (B,)
        """

        new_t = t
        t = t.unsqueeze(-1).unsqueeze(-1)
        z = torch.randn(
            x.shape, dtype=x.dtype, device=x.device, requires_grad=False
        )  # (B, T, mel_dim)

        cfg_scale = self.cfg_scale

        # get prompt len
        if torch.rand(1) > cfg_scale:
            prompt_len = torch.randint(
                min(x.shape[1] // 4, 5), int(x.shape[1] * 0.4), (x.shape[0],)
            ).to(
                x.device
            )  # (B,)
        else:
            prompt_len = torch.zeros(x.shape[0]).to(x)  # (B,)

        # get is prompt
        is_prompt = torch.zeros_like(x[:, :, 0])  # (B, T)
        col_indices = (
            torch.arange(is_prompt.shape[1])
            .repeat(is_prompt.shape[0], 1)
            .to(prompt_len)
        )  # (B, T)
        is_prompt[col_indices < prompt_len.unsqueeze(1)] = 1  # (B, T) 1 if prompt

        mask = torch.ones_like(x[:, :, 0])  # mask if 1, not mask if 0
        mask[is_prompt.bool()] = 0
        mask = mask[:, :, None]

        # flow matching: xt = (1 - (1 - sigma) * t) * x0 + t * x; where x0 ~ N(0, 1), x is a sample
        # flow gt: x - (1 - sigma) * x0 = x - (1 - sigma) * noise
        xt = ((1 - (1 - self.sigma) * t) * z + t * x) * mask + x * (1 - mask)

        return xt, z, new_t, prompt_len, mask

    def loss_t(
        self,
        x,
        x_mask,
        t,
        cond=None,
    ):
        xt, z, new_t, prompt_len, mask = self.forward_diffusion(x, t)

        noise = z

        # drop all condition for cfg, so if prompt_len is 0, we also drop cond
        if cond is not None:
            cond = cond * torch.where(
                prompt_len > 0,
                torch.ones_like(prompt_len),
                torch.zeros_like(prompt_len),
            ).to(cond.device).unsqueeze(-1).unsqueeze(-1)

        flow_pred = self.diff_estimator(xt, new_t, cond, x_mask)  # (B, T, mel_dim)

        # final mask used for loss calculation
        final_mask = mask * x_mask[..., None]  # (B, T, 1)

        return noise, x, flow_pred, final_mask, prompt_len

    def compute_loss(self, x, x_mask, cond=None):
        # x0: (B, T, num_quantizer)
        # x_mask: (B, T) mask is 0 for padding
        t = torch.rand(x.shape[0], device=x.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)
        # from CosyVoice: considering the generation process at the beginning is harder than follows, we involve a cosine scheduler for the timestep t
        if self.time_scheduler == "cos":
            t = 1 - torch.cos(t * math.pi * 0.5)
        else:
            pass
        return self.loss_t(x, x_mask, t, cond)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)

    #欧拉积分解决了：如何从噪声到有意义的数据平滑转变，实现端到端生成过程
    #CFG解决了：如何精确控制生成内容特定属性，使结果更接近音色目标
    @torch.no_grad()
    def reverse_diffusion(
        self,
        cond,                               #条件嵌入（VQ编码）
        prompt,                             #mel
        x_mask=None,                        #目标序列掩码   
        prompt_mask=None,                   #参考序列掩码
        n_timesteps=10,                     #扩散步数
        cfg=1.0,                            #分类器自由引导强度
        rescale_cfg=0.75,                   #引导后缩放因子
    ):
        h = 1.0 / n_timesteps               #积分步长
        prompt_len = prompt.shape[1]
        target_len = cond.shape[1] - prompt_len

        if x_mask == None:
            x_mask = torch.ones(cond.shape[0], target_len).to(cond.device)  # (B, T)
        if prompt_mask == None:
            prompt_mask = torch.ones(cond.shape[0], prompt_len).to(
                cond.device
            )  # (B, prompt_len)
        xt_mask = torch.cat([prompt_mask, x_mask], dim=1)
        #从高斯噪声初始化
        z = torch.randn(
            (cond.shape[0], target_len, self.mel_dim),
            dtype=cond.dtype,
            device=cond.device,
            requires_grad=False,
        )
        #初始状态是纯噪声
        xt = z
        # t from 0 to 1: x0 = z ~ N(0, 1)
        for i in range(n_timesteps):
            #拼接参考提示和当前生成内容
            xt_input = torch.cat([prompt, xt], dim=1)
            #连续轨迹上的位置，t代表了从噪声分布到数据分布的连续转换过程中特定时刻
            # 中点欧拉法 -> (i+0.5)*h
            t = (0 + (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            #预测向量场
            flow_pred = self.diff_estimator(xt_input, t, cond, xt_mask)
            flow_pred = flow_pred[:, prompt_len:, :]
            
            # 分类器自由引导
            # 通过有条件与无条件预测之间的差异来引导生成，而无需额外的分类器
            if cfg > 0:
                #无条件下的预测：通过将条件向量置为0，模型生成“无条件”预测，代表生成过程中与条件无关的部分
                uncond_flow_pred = self.diff_estimator(
                    xt, t, torch.zeros_like(cond)[:, : xt.shape[1], :], x_mask
                )
                #记录有条件预测的标准差
                pos_flow_pred_std = flow_pred.std()
                #应用CFG公式：通过放大又条件和无条件预测的差异，增强条件对生成过程的控制
                flow_pred_cfg = flow_pred + cfg * (flow_pred - uncond_flow_pred)
                #方差校正，防止过大偏移，为了保持预测的分布一致性
                rescale_flow_pred = (
                    flow_pred_cfg * pos_flow_pred_std / flow_pred_cfg.std()
                )
                #混合原始和校正后的预测：平衡原始预测和校正预测之间的差异
                flow_pred = (
                    rescale_cfg * rescale_flow_pred + (1 - rescale_cfg) * flow_pred_cfg
                )
            
            #欧拉积分步骤
            #在向量场的指导下，特征沿着预测方向移动了预测的一个时间步
            dxt = flow_pred * h
            xt = xt + dxt

        return xt

    def forward(self, x, x_mask, cond_code=None, cond_feature=None):
        """
        Args:
            x: (B, T, mel_dim)
            x_mask: (B, T)
            cond_code: (B, T) if not None
            cond_feature: (B, T, D) if not None
        """
        if cond_code != None and self.use_cond_code:
            cond = self.cond_emb(cond_code)
            # TODO: use the target_len to interpolate (after cond_emb for code)

        elif cond_feature != None and not self.use_cond_code:

            # TODO: use the target_len to interpolate
            if self.cond_scale_factor != 1:
                cond_feature = F.interpolate(
                    cond_feature.transpose(1, 2), scale_factor=self.cond_scale_factor
                ).transpose(1, 2)

            cond = self.cond_emb(cond_feature)

        noise, x, flow_pred, final_mask, prompt_len = self.compute_loss(x, x_mask, cond)
        return noise, x, flow_pred, final_mask, prompt_len
