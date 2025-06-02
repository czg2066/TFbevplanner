import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import BiFPNs, BevViews_fused

class RegNetBiFPN(nn.Module):
    def __init__(self, num_bifpn_layers=1, backbone_name='regnety_032', pretrained=True, output_levels=4, ori_size=(160, 90)):
        """
        Args:
            bifpn_channels (int): BiFPN 内部的通道数
            num_bifpn_layers (int): 重复堆叠的 BiFPN 层数
            backbone_name (str): 使用的 RegNet 名称 (timm 支持的)
            pretrained (bool): 是否加载预训练的 RegNet 权重
            output_levels (int): BiFPN 输出的层级数 (通常是 5, 对应 P3-P7)
        """
        super().__init__()
        self.output_levels = output_levels
        self.out_size = []
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        feature_info = self.backbone.feature_info.get_dicts(keys=('num_chs', 'reduction'))
        self.backbone_output_channels = [info['num_chs'] for info in feature_info]
        backbone_strides = [info['reduction'] for info in feature_info]
        print(f"RegNet Backbone output channels: {self.backbone_output_channels}")
        print(f"RegNet Backbone strides: {[info['reduction'] for info in feature_info]}")
        self.input_projs = []
        # self.bifpns = []
        self.fused_nets = []
        for out_channels in self.backbone_output_channels:
            stride = backbone_strides[self.backbone_output_channels.index(out_channels)]
            input_proj = nn.ModuleList()
            for in_channels in self.backbone_output_channels:
                input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        # ReLU 或 SiLU 可选，通常在这里加个激活
                        # nn.ReLU(inplace=True)
                        nn.SiLU(inplace=True)
                    )
                )
            self.input_projs.append(input_proj)
        #     self.bifpn = nn.ModuleList([
        #         BiFPNs.BiFPNLayer(num_channels=out_channels, num_levels=output_levels, channels=self.backbone_output_channels)
        #         for _ in range(num_bifpn_layers)
        #     ])
        #     self.bifpns.append(self.bifpn)
            h, w = ori_size
            h = 1+h//stride if h%stride != 0 else h//stride
            w = 1+w//stride if w%stride != 0 else w//stride
            self.fused_nets.append(BevViews_fused.CrossViewTransformer(out_channels, (h, w), num_heads=2))
        pass
    def forward(self, x):
        backbone_features = self.backbone(x)
        c3, c4, c5, c6 = backbone_features
        out_features = []
        for net_i in range(len(self.input_projs)):
            input_proj = self.input_projs[net_i].to(x.device)
            # bifpn = self.bifpns[net_i].to(x.device)
            fused_net = self.fused_nets[net_i].to(x.device)
            p_inputs = []
            p_inputs.append(input_proj[0](c3))
            p_inputs.append(input_proj[1](c4))
            p_inputs.append(input_proj[2](c5))
            p_inputs.append(input_proj[3](c6))
            features = p_inputs
        #     for bifpn_layer in bifpn:
        #         features = bifpn_layer(features, net_i)
            out_features.append(fused_net(features[net_i]))
        return out_features

# --- 使用示例 ---
if __name__ == '__main__':
    # 参数设置
    BIFPN_CHANNELS = 128 # BiFPN 内部通道数 (可以根据需要调整, e.g., 64, 88, 112, 160, 224, 288...)
    NUM_BIFPN_LAYERS = 4  # BiFPN 重复次数 (e.g., 3 到 7)
    OUTPUT_LEVELS = 5     # 输出 P3-P7
    IMAGE_SIZE = 256      # 输入图像尺寸 (RegNet 通常用 224, 但这里用 256 示例)
    BATCH_SIZE = 2

    # 创建模型
    # 使用 RegNetY-3.2GF ('regnety_032')
    model = RegNetBiFPN(
        bifpn_channels=BIFPN_CHANNELS,
        num_bifpn_layers=NUM_BIFPN_LAYERS,
        backbone_name='regnety_032', # 对应 3.2 GF
        pretrained=True,             # 加载预训练权重
        output_levels=OUTPUT_LEVELS
    )
    model.eval() # 设置为评估模式

    # 创建假的输入图像
    dummy_input = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

    # 前向传播
    with torch.no_grad(): # 不需要计算梯度
        output_features = model(dummy_input)

    print(f"\nBiFPN Output Feature Shapes (P3 to P{OUTPUT_LEVELS+2}):")
    for i, f in enumerate(output_features):
        print(f"  P{i+3}_out: {f.shape}") # P3, P4, P5, P6, P7

    # 检查输出通道数是否正确
    for f in output_features:
        assert f.shape[1] == BIFPN_CHANNELS

    # 检查输出分辨率是否符合预期 (相对于输入图像尺寸)
    expected_strides = [8, 16, 32, 64, 128] # P3 to P7 的下采样倍数
    for i, f in enumerate(output_features):
        expected_size = IMAGE_SIZE // expected_strides[i]
        assert f.shape[-1] == expected_size and f.shape[-2] == expected_size, \
               f"P{i+3} shape mismatch: expected ({expected_size},{expected_size}), got {f.shape[-2:]}"

    print("\nModel built and forward pass successful!")

    # --- 后续步骤 ---
    # 这些输出的特征图 (output_features) 接下来会送入目标检测的 Class/Box Prediction Heads
    # 例如：
    # class_head = ...
    # box_head = ...
    # class_outputs = [class_head(f) for f in output_features]
    # box_outputs = [box_head(f) for f in output_features]