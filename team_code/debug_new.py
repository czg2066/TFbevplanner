import data, torch
from torch.utils.data import DataLoader
from config import GlobalConfig
import Backone

if __name__ == '__main__':
    root_path = "/media/czg/vec_env/data_collect/carla_garage/collection_data"
    NUM_BIFPN_LAYERS = 1  # BiFPN 重复次数 (e.g., 3 到 7)
    OUTPUT_LEVELS = 4     # 输出 P3-P7
    BATCH_SIZE = 1
    config = GlobalConfig()
    config.initialize(config.root_dir)
    # 创建模型
    # 使用 RegNetY-3.2GF ('regnety_032')
    model = Backone.RegNetBiFPN(
        num_bifpn_layers=NUM_BIFPN_LAYERS,
        backbone_name='regnety_032', # 对应 3.2 GF
        pretrained=True,             # 加载预训练权重
        output_levels=OUTPUT_LEVELS
    )
    model.cuda(device=0)
    model.eval()
    train_set = data.CARLA_Data(root=config.train_data,
                        config=config,
                        estimate_class_distributions=config.estimate_class_distributions,
                        estimate_sem_distribution=config.estimate_semantic_distribution,
                        shared_dict=None,
                        rank=0)
    dataloader_train = DataLoader(dataset=train_set,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=18,
                        pin_memory=True,
                        drop_last=True,
                        persistent_workers=True)
    for i, data in enumerate(dataloader_train):
        print(i)
        rgb_front = data['rgb_front'].to('cuda:0', dtype=torch.float32)
        rgb_front_left = data['rgb_front_left'].to('cuda:0', dtype=torch.float32)
        rgb_front_right = data['rgb_front_right'].to('cuda:0', dtype=torch.float32)
        rgb_back = data['rgb_back'].to('cuda:0', dtype=torch.float32)
        rgb_back_left = data['rgb_back_left'].to('cuda:0', dtype=torch.float32)
        rgb_back_right = data['rgb_back_right'].to('cuda:0', dtype=torch.float32)
        six_view_combined = torch.cat([rgb_front, rgb_front_left, rgb_front_right, rgb_back, rgb_back_left, rgb_back_right], dim=0)
        with torch.no_grad():
            model(six_view_combined)

        # print(f"\nBiFPN Output Feature Shapes (P3 to P{OUTPUT_LEVELS+2}):")
        # for i, f in enumerate(output_features):
        #     print(f"  P{i+3}_out: {f.shape}") # P3, P4, P5, P6, P7

        # # 检查输出通道数是否正确
        # for f in output_features:
        #     assert f.shape[1] == BIFPN_CHANNELS