import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import math
import os, json
import numpy as np
from einops import rearrange, repeat
from networks.mamba import CZSS

class encoder(nn.Module):
    def __init__(self, in_ch=1, zero_output=False, dim=48):
        super(encoder, self).__init__()
        self.zero_output = zero_output
        in_channels = in_ch

        ####################################
        # Encode Blocks
        ####################################

        # Layers: enc_conv0, enc_conv1, pool1
        self.encode_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        # Layers: enc_conv(i), pool(i); i=2..5
        def _encode_block_2_3_4_5() -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(dim, dim, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.MaxPool2d(2)
            )

        # Separate instances of same encode module definition created
        self.encode_block_2 = _encode_block_2_3_4_5()
        self.encode_block_3 = _encode_block_2_3_4_5()
        self.encode_block_4 = _encode_block_2_3_4_5()
        self.encode_block_5 = _encode_block_2_3_4_5()

        # Layers: enc_conv6
        self.encode_block_6 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initializes weights using Kaiming  He et al. (2015).

        Only convolution layers have learnable weights. All convolutions use a leaky
        relu activation function (negative_slope = 0.1) except the last which is just
        a linear output.
        """
        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()

        # Initialise last output layer
        # if self.zero_output:
        #     self.output_conv.weight.zero_()
        # else:
        #     nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")

    def forward(self, x):
        # Encoder
        pool1 = self.encode_block_1(x)
        pool2 = self.encode_block_2(pool1)
        pool3 = self.encode_block_3(pool2)
        pool4 = self.encode_block_4(pool3)
        pool5 = self.encode_block_5(pool4)
        encoded = self.encode_block_6(pool5)

        return [pool1, pool2, pool3, pool4, pool5], encoded

class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, D, W, H)
        Returns: same shape as hidden_states
        """
        batch, dim, h, w = hidden_states.shape
        hidden_states = hidden_states.flatten(2).transpose(1, 2) #(B,D,W,H) -> (B, H*W, D) = (B, L, D)
        
        
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        
        out = out.transpose(1, 2).view(batch, dim, h, w) # (B, L, D) = (B, H*W, D) -> (B, D, W, H)
        return out


class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.mamba = Mamba(
            d_model=dim * 4,  
            dt_rank="auto",
            d_state=16,  
            d_conv=3,  
            expand=2  
        )
        
        index_path = f"{os.path.abspath('')}/"
        with open(os.path.join(index_path, "index/ZScan.json"), "r", encoding="utf-8") as f:
            self.ZScan_dict = json.load(f)
        
        self.out_norm = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.LayerNorm(dim),
            Permute(0, 3, 1, 2),
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W

        ZScan = torch.tensor(np.array(self.ZScan_dict[f"{H}_ZScan"][0]), dtype=torch.long).cuda()
        ZScan_R = torch.tensor(np.array(self.ZScan_dict[f"{H}_ZScan"][1]), dtype=torch.long).cuda()
        ZScan_ = ZScan.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        ZScan_R_ = ZScan_R.unsqueeze(0).unsqueeze(0).expand(B, C, -1)

        x1 = x.flatten(start_dim=2)  
        x1 = x1.gather(2, ZScan_)

        x2 = torch.transpose(x, dim0=2, dim1=3).contiguous()
        x2 = x2.flatten(start_dim=2)
        x2 = x2.gather(2, ZScan_)

        x_hwwh = torch.stack([x1.view(B, -1, L), x2.view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        xs = xs.float().view(B, -1, L).transpose(1, 2)
        # print(xs.shape)
 
        # 四个数据处理完毕
        
        out_y = self.mamba(xs).transpose(1, 2).view(B, 4, -1, L)

        y1 = out_y[:, 0].gather(2, ZScan_R_)
                
        y2 = out_y[:, 1].gather(2, ZScan_R_).reshape(B, C, H, W)
        y2 = torch.transpose(y2, dim0=2, dim1=3).contiguous().view(B, -1, L)

        y3 = torch.flip(out_y[:, 2], dims=[-1]).gather(2, ZScan_R_)

        y4 = torch.flip(out_y[:, 3], dims=[-1]).gather(2, ZScan_R_).reshape(B, C, H, W)
        y4 = torch.transpose(y4, dim0=2, dim1=3).contiguous().view(B, -1, L)

        y = y1 + y2 + y3 + y4
        y = torch.transpose(y1, dim0=1, dim1=2).contiguous().view(B, -1, H, W).cuda()
        out = self.out_norm(y)
        
        del ZScan, ZScan_R, ZScan_, ZScan_R_
        return out

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(self.dims)

class MambaConvBlock_parallel_freq(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.avgpool = nn.AvgPool2d(7, 1, 3)
        self.dilationConv = nn.Conv2d(dim, dim, kernel_size=3, dilation=4, padding=4)
        self.sigmoid = nn.Sigmoid()
        self.gAvgpool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.layerNorm = nn.LayerNorm(dim)
        
        # self.mamba = MambaVisionMixer(dim)
        # self.mamba = MambaBlock(dim)
        self.mamba = CZSS(hidden_dim=dim)
        
        self.conv_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        
        self.fusion = nn.Conv2d(2 * dim, dim, kernel_size=1)  # 微调融合特征
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.in_proj = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.LayerNorm(dim),
            Permute(0, 3, 1, 2),
        )
        
    def forward(self, x):
        
        x = self.in_proj(x)
        
        low_freq = self.avgpool(self.conv1(x))
        high_freq = self.conv1(x - self.conv1(low_freq))
        
        mamba_out = self.mamba(low_freq)
        conv_out = self.conv_branch(high_freq)
        
        edge_guide = self.dilationConv(mamba_out)
        edge_guide = self.sigmoid(edge_guide)
        high_enhanced = conv_out * (1 + edge_guide)
        
        local_corr = self.gAvgpool(conv_out)
        local_corr = self.act(local_corr)
        low_enhanced = mamba_out + mamba_out * local_corr
        
        fused = torch.cat([low_enhanced, high_enhanced], dim = 1)
        out = self.act(self.fusion(fused) + x)
        return out


class mamba_encoder_parallel(nn.Module):
    def __init__(self, in_ch=1, dim=48):
        super().__init__()
        self.in_ch = in_ch
        self.dim = dim

        # 初始卷积块
        self.encode_block_1 = nn.Sequential(
            nn.Conv2d(in_ch, dim, kernel_size=3, padding=1), 
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            MambaConvBlock_parallel_freq(dim), 
            nn.MaxPool2d(2)  
        )

        # 编码块2-5
        def _encode_block() -> nn.Module:
            return nn.Sequential(
                MambaConvBlock_parallel_freq(dim), 
                nn.MaxPool2d(2)  
            )

        self.encode_block_2 = _encode_block()  # 64x64 → 32x32（skip2）
        self.encode_block_3 = _encode_block()  # 32x32 → 16x16（skip3）
        self.encode_block_4 = _encode_block()  # 16x16 → 8x8（skip4）
        self.encode_block_5 = _encode_block()  # 8x8 → 4x4（skip5）

        # 最终编码块（保持尺寸）
        self.encode_block_6 = nn.Sequential(
            MambaConvBlock_parallel_freq(dim) 
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # (1, 1, 128, 128)
        skip1 = self.encode_block_1(x)  # (1, 48, 64, 64)
        skip2 = self.encode_block_2(skip1)  # (1, 48, 32, 32)
        skip3 = self.encode_block_3(skip2)  # (1, 48, 16, 16)
        skip4 = self.encode_block_4(skip3)  # (1, 48, 8, 8)
        skip5 = self.encode_block_5(skip4)  # (1, 48, 4, 4)
        encoded = self.encode_block_6(skip5)  # (1, 48, 4, 4)

        return [skip1, skip2, skip3, skip4, skip5], encoded

# 普通decoder
class decoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, zero_output=False, dim=48):
        super(decoder, self).__init__()
        self.zero_output = zero_output
        in_channels = in_ch
        out_channels = out_ch

        # Layers: upsample5
        self.decode_block_6 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self.decode_block_5 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim * 2, dim * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        def _decode_block_4_3_2() -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(dim * 3, dim * 2, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(dim * 2, dim * 2, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

        # Separate instances of same decode module definition created
        self.decode_block_4 = _decode_block_4_3_2()
        self.decode_block_3 = _decode_block_4_3_2()
        self.decode_block_2 = _decode_block_4_3_2()

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self.decode_block_1 = nn.Sequential(
            nn.Conv2d(dim * 2 + in_channels, dim * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim * 2, dim * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        ####################################
        # Output Block
        ####################################

        # nin_a,b,c, linear_act
        self.output_conv = nn.Conv2d(dim * 2, out_channels, 1)

        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initializes weights using Kaiming  He et al. (2015).

        Only convolution layers have learnable weights. All convolutions use a leaky
        relu activation function (negative_slope = 0.1) except the last which is just
        a linear output.
        """
        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()

        # Initialise last output layer
        if self.zero_output:
            self.output_conv.weight.zero_()
        else:
            nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")

    def forward(self, x, skips, encoded):
        pool1, pool2, pool3, pool4, pool5 = skips
        upsample5 = self.decode_block_6(encoded)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode_block_5(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode_block_4(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode_block_3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode_block_2(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        x = self.decode_block_1(concat1) 

        x = self.output_conv(x)        
        return x

if __name__ == "__main__":
    # 确保中文显示正常（可选）
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    
    # 1. 配置参数
    batch_size = 10
    in_channels = 1
    out_channels = 1
    img_size = (128, 128)
    dim = 1

    # 2. 选择设备（优先GPU，无GPU则用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 3. 创建模型并移到设备上
    encoder_model = mamba_encoder_parallel(in_ch=in_channels).to(device)
    decoder_model = decoder(in_ch=in_channels, out_ch=out_channels).to(device)

    # 4. 生成测试数据并移到设备上
    input_data = torch.randn(batch_size, in_channels, img_size[0], img_size[1]).to(device)
    print(f"输入数据形状: {input_data.shape} (设备: {input_data.device})")

    # 5. 前向传播
    skips, encoded = encoder_model(input_data)
    for a in skips:
        print(a.shape)
    print(encoded.shape)
    
    print("\n编码器输出验证：")
    for i, skip in enumerate(skips, 1):
        print(f"skip{i} 形状: {skip.shape} (设备: {skip.device})")
    print(f"encoded 形状: {encoded.shape} (设备: {encoded.device})")

    output = decoder_model(input_data, skips, encoded)
    print(f"\n解码器输出形状: {output.shape} (设备: {output.device})")

    # 6. 验证输出尺寸
    assert output.shape == input_data.shape, f"尺寸不匹配！预期 {input_data.shape}，实际 {output.shape}"
    print("\n测试通过：输出尺寸与输入一致！")
