import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, in_channels=1, alpha = 0):
        super(FusionModule, self).__init__()
        
        self.alpha = alpha
        # 特征提取网络 - 处理第一张图像
        self.feature_extractor1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # 特征提取网络 - 处理第二张图像
        self.feature_extractor2 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # 特征融合与权重生成网络
        self.fusion_network = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()  # 输出范围[0,1]的融合权重图
        )
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
                
    def forward(self, img1, img2):
        # 确保输入图像尺寸相同
        assert img1.size() == img2.size(), "输入图像尺寸必须相同"
        
        # 特征提取
        feat1 = self.feature_extractor1(img1)
        feat2 = self.feature_extractor2(img2)
        
        # 特征拼接
        concat_feat = torch.cat([feat1, feat2], dim=1)
        
        # 生成融合权重图
        weight_map = self.fusion_network(concat_feat)
        weight_map = self.alpha + (1 - self.alpha) * weight_map
        # 使用权重图融合原始输入图像
        fused_img = weight_map * img1 + (1 - weight_map) * img2
        
        return fused_img, weight_map

# 使用示例
if __name__ == "__main__":
    # 创建模型实例
    model = FusionModule(in_channels=1)
    
    # 生成随机测试图像 (批次大小=1, 通道=3, 高度=224, 宽度=224)
    img1 = torch.randn(24, 1, 128, 128)
    img2 = torch.randn(24, 1, 128, 128)
    
    # 前向传播
    fused_output, weight_map = model(img1, img2)
    
    print(f"输入图像尺寸: {img1.shape}")
    print(f"融合权重图尺寸: {weight_map.shape}")
    print(f"输出图像尺寸: {fused_output.shape}")
    print(f"权重图值范围: [{weight_map.min():.4f}, {weight_map.max():.4f}]")    