import torch
import torch.nn as nn
import torch.nn.functional as F

# 可分离卷积块 (用于效率)
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=nn.SiLU):
        super().__init__()
        # 深度卷积 (每个输入通道一个卷积核)
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        #逐点卷积 (1x1 卷积混合通道)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = activation()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.act(x)
        return x

class CrossScaleFusion(nn.Module):
    def __init__(self, num_levels, channels):
        super().__init__()
        self.num_levels = num_levels
        
        # 上采样和下采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, features, net_i):
        assert len(features) == self.num_levels
        
        # 生成所有特征图的调整版本
        adjusted_features = []
        for j in range(self.num_levels):
            if net_i == j:
                # 保持原样
                feat = features[j]
            elif net_i < j:
                # 需要上采样
                feat = F.interpolate(
                    features[j], 
                    size=features[net_i].shape[-2:], 
                    mode='nearest'
                )
            else:
                # 需要下采样
                feat = self.downsample(features[j])
                if feat.shape[-2:] != features[net_i].shape[-2:]:
                    feat = F.adaptive_avg_pool2d(feat, features[net_i].shape[-2:])

            adjusted_features.append(feat)
        
        # 计算加权融合
        weighted_sum = 0.7 * adjusted_features[j]
        for j in range(self.num_levels):
            if net_i != j:
                weighted_sum += 0.1 * adjusted_features[j]
        features[net_i] = weighted_sum
        return features

# BiFPN 层
class BiFPNLayer(nn.Module):
    def __init__(self, num_channels, num_levels=5, epsilon=1e-4, channels=None):
        """
        Args:
            num_channels (int): BiFPN内部特征图的通道数
            num_levels (int): 特征金字塔的层级数 (例如 P3-P7 就是 5 层)
            epsilon (float): 用于加权融合时防止除零的小常数
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_levels = num_levels
        self.epsilon = epsilon
        self.conv_blocks = nn.ModuleList([SeparableConvBlock(num_channels, num_channels) for _ in range(num_levels)])
        self.w1 = nn.Parameter(torch.ones(num_levels - 1, 2))
        num_weights_w2 = 2 + 3 * (num_levels - 2) + 2
        self.w2 = nn.Parameter(torch.ones(num_weights_w2, 1))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fused_net = CrossScaleFusion(num_levels, channels)

    def forward(self, features, net_i):
        """
        Args:
            features (list[torch.Tensor]): 输入的多尺度特征图列表 [P3_in, P4_in, ..., P7_in]

        Returns:
            list[torch.Tensor]: 输出的融合后的多尺度特征图列表 [P3_out, P4_out, ..., P7_out]
        """
        assert len(features) == self.num_levels, "输入特征层级数量与定义不符"
        td_features = [None] * self.num_levels
        out_features = [None] * self.num_levels
        td_features[self.num_levels - 1] = features[self.num_levels - 1]
        w1_normalized = torch.softmax(self.w1, dim=1) 
        for i in range(self.num_levels - 2, -1, -1):
            input1 = features[i]
            input2 = F.interpolate(td_features[i+1], size=input1.shape[2:], mode='bilinear')
            fusion = w1_normalized[i, 0] * input1 + w1_normalized[i, 1] * input2
            td_features[i] = self.conv_blocks[i](fusion) # 应用卷积提炼
        w2_normalized = torch.nn.functional.relu(self.w2) # 保证非负
        w2_normalized = w2_normalized / (w2_normalized.sum(dim=0, keepdim=True) + self.epsilon)
        fusion_p3 = w2_normalized[0] * features[0] + w2_normalized[1] * td_features[0]
        out_features[0] = self.conv_blocks[0](fusion_p3) # 复用卷积块
        w_idx = 2 # 当前使用的 w2 权重的索引
        for i in range(1, self.num_levels - 1):
            input1 = features[i]
            input2 = td_features[i]
            input3 = self.downsample(out_features[i-1])
            fusion = w2_normalized[w_idx] * input1 + w2_normalized[w_idx+1] * input2 + w2_normalized[w_idx+2] * input3
            out_features[i] = self.conv_blocks[i](fusion) # 复用卷积块
            w_idx += 3
        input1 = features[self.num_levels - 1]
        input2 = self.downsample(out_features[self.num_levels - 2])
        fusion = w2_normalized[w_idx] * input1 + w2_normalized[w_idx+1] * input2
        out_features[self.num_levels - 1] = self.conv_blocks[self.num_levels - 1](fusion)
        return self.fused_net(out_features, net_i)

# --- 使用示例 ---
if __name__ == '__main__':
    NUM_LEVELS = 5 # P3-P7
    NUM_CHANNELS = 64 # BiFPN 内的通道数
    BATCH_SIZE = 4

    # 创建一个 BiFPN 层
    bifpn_layer = BiFPNLayer(num_channels=NUM_CHANNELS, num_levels=NUM_LEVELS)

    # 创建假的输入特征图 (模仿骨干网输出)
    # 分辨率通常是 P3 > P4 > P5 > P6 > P7
    input_features = []
    base_size = 64
    for i in range(NUM_LEVELS):
        size = base_size // (2**i)
        # 输入通道数可能与 NUM_CHANNELS 不同，实际应用中需要 1x1 conv 调整
        # 这里简化为相同通道数
        dummy_feature = torch.randn(BATCH_SIZE, NUM_CHANNELS, size, size)
        input_features.append(dummy_feature)

    print(f"Input feature shapes:")
    for i, f in enumerate(input_features):
        print(f"  P{i+3}: {f.shape}")

    # 通过 BiFPN 层
    output_features = bifpn_layer(input_features)

    print(f"\nOutput feature shapes:")
    for i, f in enumerate(output_features):
        print(f"  P{i+3}_out: {f.shape}")

    # 检查输出尺寸是否与输入对应尺寸一致
    for i in range(NUM_LEVELS):
        assert input_features[i].shape == output_features[i].shape

    print("\nBiFPN layer executed successfully!")

    # 在实际的 EfficientDet 中，你会堆叠多个这样的 BiFPNLayer
    # num_repeats = 3 # 例如，重复 3 次
    # bifpn_network = nn.Sequential(*[BiFPNLayer(NUM_CHANNELS, NUM_LEVELS) for _ in range(num_repeats)])
    # current_features = input_features
    # for layer in bifpn_network:
    #     current_features = layer(current_features)
    # final_output_features = current_features