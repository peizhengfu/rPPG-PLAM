import torch
import typing as t
import torch.nn as nn
from einops import rearrange
from mmengine.model import BaseModule
from thop import profile
 
# 确保正确导入SCSA
class SCSA(BaseModule):
    # SCSA类的初始化和方法定义见上方代码
 
    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            norm_cfg: t.Dict = dict(type='BN'),
            act_cfg: t.Dict = dict(type='ReLU'),
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(SCSA, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode
 
        assert self.dim // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4
 
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)
 
        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()
 
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                # dimensionality reduction
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The dim of x is (B, C, H, W)
        """
        # Spatial attention priority calculation
        b, c, h_, w_ = x.size()
        # (B, C, H)
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # (B, C, W)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)
 
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)
 
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)
 
        x = x * x_h_attn * x_w_attn
 
        # Channel attention based on self attention
        # reduce calculations
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()
 
        # normalization first, then reshape -> (B, H, W, C) -> (B, C, H * W) and generate q, k and v
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
 
        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x
 
# # 使用示例
# if __name__ == "__main__":
#     scsa = SCSA(dim=256, head_num=4)
#     x = torch.randn(1, 256, 64, 64)  # 生成随机输入
 
#     if torch.cuda.is_available():
#         x = x.cuda()
#         scsa = scsa.cuda()
 
#     y = scsa(x)  # 计算输出
#     print("输出维度为", y.shape)
 
    # flops, params = profile(scsa, inputs=(x,))  # 计算 FLOPs
    # print(f"Total FLOPs: {flops / (10 ** 9):.2f} GFLOPs")
 
    # total_num = sum(p.numel() for p in scsa.parameters())
    # trainable_num = sum(p.numel() for p in scsa.parameters() if p.requires_grad)
    # print("总参数量为", total_num / 1000000, "M")
# class SCSA(BaseModule):
#     def __init__(
#             self,
#             dim: int,
#             head_num: int,
#             window_size: int = 7,
#             group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
#             qkv_bias: bool = False,
#             fuse_bn: bool = False,
#             norm_cfg: t.Dict = dict(type='BN'),
#             act_cfg: t.Dict = dict(type='ReLU'),
#             down_sample_mode: str = 'avg_pool',
#             attn_drop_ratio: float = 0.1,  # 增加 dropout
#             gate_layer: str = 'swish',  # 替换为 Swish
#     ):
#         super(SCSA, self).__init__()
#         self.dim = dim
#         self.head_num = head_num
#         self.head_dim = dim // head_num
#         self.scaler = self.head_dim ** -0.5
#         self.group_kernel_sizes = group_kernel_sizes
#         self.window_size = window_size
#         self.qkv_bias = qkv_bias
#         self.fuse_bn = fuse_bn
#         self.down_sample_mode = down_sample_mode
 
#         assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'
#         self.group_chans = group_chans = self.dim // 4
 
#         # 使用深度可分离卷积减少计算量
#         self.local_dwc = nn.Conv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
#                                    padding=group_kernel_sizes[0] // 2, groups=group_chans)
#         self.global_dwc_s = nn.Conv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
#                                       padding=group_kernel_sizes[1] // 2, groups=group_chans)
#         self.global_dwc_m = nn.Conv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
#                                       padding=group_kernel_sizes[2] // 2, groups=group_chans)
#         self.global_dwc_l = nn.Conv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
#                                       padding=group_kernel_sizes[3] // 2, groups=group_chans)
#         # 替换注意力激活函数
#         self.sa_gate = nn.SiLU() if gate_layer == 'swish' else nn.Sigmoid()
#         self.norm_h = nn.LayerNorm([dim])  # 替换 GroupNorm 为 LayerNorm
#         self.norm_w = nn.LayerNorm([dim])
 
#         self.conv_d = nn.Identity()
#         self.norm = nn.GroupNorm(1, dim)  # 这里保留 GroupNorm
#         # 使用更小的 qkv 卷积层
#         self.q = nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=1, bias=qkv_bias)
#         self.k = nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=1, bias=qkv_bias)
#         self.v = nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=1, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop_ratio)
#         self.ca_gate = nn.Sigmoid()  # 使用默认的 Sigmoid
 
#         if window_size == -1:
#             self.down_func = nn.AdaptiveAvgPool2d((1, 1))
#         else:
#             if down_sample_mode == 'recombination':
#                 self.down_func = self.space_to_chans
#                 self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
#             elif down_sample_mode == 'avg_pool':
#                 self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
#             elif down_sample_mode == 'max_pool':
#                 self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)
 
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         b, c, h_, w_ = x.size()
#         x_h = x.mean(dim=3)
#         l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
#         x_w = x.mean(dim=2)
#         l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)
 
#         x_h_attn = self.sa_gate(self.norm_h(torch.cat((
#             self.local_dwc(l_x_h),
#             self.global_dwc_s(g_x_h_s),
#             self.global_dwc_m(g_x_h_m),
#             self.global_dwc_l(g_x_h_l),
#         ), dim=1)))
#         x_h_attn = x_h_attn.view(b, c, h_, 1)
 
#         x_w_attn = self.sa_gate(self.norm_w(torch.cat((
#             self.local_dwc(l_x_w),
#             self.global_dwc_s(g_x_w_s),
#             self.global_dwc_m(g_x_w_m),
#             self.global_dwc_l(g_x_w_l)
#         ), dim=1)))
#         x_w_attn = x_w_attn.view(b, c, 1, w_)
 
#         x = x * x_h_attn * x_w_attn
 
#         y = self.down_func(x)
#         y = self.conv_d(y)
#         _, _, h_, w_ = y.size()
 
#         y = self.norm(y)
#         q = self.q(y)
#         k = self.k(y)
#         v = self.v(y)
#         q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
#                       head_dim=int(self.head_dim))
#         k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
#                       head_dim=int(self.head_dim))
#         v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
#                       head_dim=int(self.head_dim))
 
#         attn = q @ k.transpose(-2, -1) * self.scaler
#         attn = self.attn_drop(attn.softmax(dim=-1))
#         attn = attn @ v
#         attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
#         attn = attn.mean((2, 3), keepdim=True)
#         attn = self.ca_gate(attn)
#         return attn * x
 
