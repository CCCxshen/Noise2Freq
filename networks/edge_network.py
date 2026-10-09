import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from utils.matplotlib_util import *

class CTHighFreqExtractor(nn.Module):
    def __init__(self, input_channels=1):
        super(CTHighFreqExtractor, self).__init__()
        
        # 收缩路径：捕获特征并抑制噪声
        self.encoder = nn.Sequential(
            # 第一层：初步特征提取与噪声抑制
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            
            # 第二层：增强特征表达
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            # 第三层：深入特征提取
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )
        
        # 边缘增强模块：使用可学习的边缘检测核
        self.edge_enhancer = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),  # 深度可分离卷积
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )
        
        # 扩展路径：恢复分辨率并输出高频细节
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            
            # 输出层：单通道高频细节图
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh()  # 输出范围[-1, 1]，便于后续处理
        )
        
        # 噪声抑制分支：识别并抑制噪声区域
        self.noise_suppressor = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()  # 输出噪声掩码，范围[0, 1]
        )
        
        # 初始化边缘检测卷积核
        self._init_edge_kernels()
        
    def _init_edge_kernels(self):
        """初始化边缘检测卷积核，使用类Sobel算子作为初始值"""
        # 水平和垂直方向边缘检测核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) / 8.0
        
        # 为edge_enhancer的第一个卷积层初始化权重
        with torch.no_grad():
            conv_layer = self.edge_enhancer[0]
            # 交替初始化水平和垂直边缘检测核
            for i in range(conv_layer.weight.size(0)):
                if i % 2 == 0:
                    conv_layer.weight[i, 0] = sobel_x
                else:
                    conv_layer.weight[i, 0] = sobel_y
    
    def forward(self, x):
        # 保存输入用于残差计算
        input_x = x
        
        # 特征提取与噪声抑制
        features = self.encoder(x)
        
        # 增强边缘特征
        edge_features = self.edge_enhancer(features)
        
        # 生成噪声掩码 (1表示噪声区域，0表示细节区域)
        noise_mask = self.noise_suppressor(features)
        
        # 应用噪声抑制：通过掩码弱化噪声区域的特征
        edge_features = edge_features * (1 - noise_mask)
        
        # 恢复分辨率并输出高频细节
        high_freq_details = self.decoder(edge_features)
        
        return high_freq_details

# 测试模型
if __name__ == "__main__":
    # 创建模型实例（CT图像通常为单通道）
    model = CTHighFreqExtractor(input_channels=1).cuda()
    
    # 打印模型结构
    print(model)
    
    # 打印模型参数摘要（假设输入CT切片大小为512x512）
    print("\n模型参数摘要：")
    summary(model, (1, 512, 512))
    
    # 测试前向传播
    # test_input = torch.randn(1, 1, 512, 512).cuda()  # 随机生成一张单通道512x512的CT切片
    a = np.load("/data0/xcshen/research/xcshen_research_topic/TP1_Noise2Freq/Noise2Freq_explore/inference/mayo_1mm_ldct_mamba/npy/step_0_input.npy")
    data = np.zeros((1, 1, 512, 512))
    data[0][0] = a
    output = model(torch.from_numpy(data).float().cuda())
    
    a = np.load("/data0/xcshen/research/xcshen_research_topic/TP1_Noise2Freq/Noise2Freq_explore/inference/mayo_1mm_ldct_mamba/npy/step_0_target.npy")
    data = np.zeros((1, 1, 512, 512))
    data[0][0] = a
    output_1 = model(torch.from_numpy(data).float().cuda())
    # print(f"\n输入形状: {data.shape}")
    # print(f"输出形状: {output.shape}")
    show_2Dimages([data[0][0], output[0][0].detach().cpu().numpy(), output_1[0][0].detach().cpu().numpy()], save_path="./")
    