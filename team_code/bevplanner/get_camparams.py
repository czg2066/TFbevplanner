import math, torch
import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_rad_to_deg(cam_trans):
    for i in range(len(cam_trans)):
        if cam_trans[i]:
            x, y, z, roll_rad, pitch_rad, yaw_rad = cam_trans[i]
            roll_deg = math.degrees(roll_rad)
            pitch_deg = math.degrees(pitch_rad)
            yaw_deg = math.degrees(yaw_rad)
            cam_trans[i] = [x, y, z, roll_deg, pitch_deg, yaw_deg]
    return cam_trans

class Cam_params():
    def __init__(self):
        self.cam = ["front_left", "front_right", "front_wide", "rear_left", "rear_right", "rear"] 
        self.rgb = ["rgb_"+i for i in self.cam] 
        self.semantics = ["semantic_"+i for i in self.cam] 
        self.depth = ["depth_"+i for i in self.cam]
        self.cam_trans = [[] for i in self.cam] #x,y,z,roll,pitch,yaw
        self.cam_trans[0] = [2.549, 0.933, 0.823, -0.044, 0.008, 0.841]
        self.cam_trans[1] = [2.549, 0.933, 0.823, 0.009, -0.021, -0.836]
        self.cam_trans[2] = [1.933, -0.003, 1.311, -0.000, -0.036, -0.007]
        self.cam_trans[3] = [2.457, 0.937, 0.827, -0.004, -0.011, 2.363]
        self.cam_trans[4] = [2.457, -0.937, 0.827, 0.038, 0.004, -2.358]
        self.cam_trans[5] = [0.006, 0.0, 2.025, 0.005, 0.026, 3.119]
        self.cam_trans = convert_rad_to_deg(self.cam_trans)
        self.rgb_wh = [[1281, 854] for i in self.rgb]
        self.cam_fov = [100, 100, 30, 120, 100, 100, 100]


def get_sensor_params():
    """
    从传感器数据中提取并计算所有需要的参数
    返回每个相机的参数字典列表
    """
    sensor_data = Cam_params()
    
    # 单位变换定义
    identity_rot = np.eye(3)  # 3x3单位旋转矩阵
    identity_post_tran = np.zeros(3)  # 零平移向量
    identity_bda = np.eye(3)  # 4x4单位BEV变换矩阵
    
    trans = []
    rots = []
    intrins = []
    post_rots = []
    post_trans = []
    bdas = []

    for i, cam_name in enumerate(sensor_data.cam):
        # 1. 提取平移向量 tran
        tran = np.array(sensor_data.cam_trans[i][:3])
        
        # 2. 计算旋转矩阵 rot
        roll_deg, pitch_deg, yaw_deg = sensor_data.cam_trans[i][3:]
        # 创建旋转对象 (使用 'xyz' 欧拉角顺序)
        rotation = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
        rot = rotation.as_matrix()
        
        # 3. 估计内参矩阵 intrin
        width, height = sensor_data.rgb_wh[i]
        fov_deg = sensor_data.cam_fov[i]
        
        # 计算焦距
        fov_rad = math.radians(fov_deg)
        fx = (width / 2.0) / math.tan(fov_rad / 2.0)
        fy = fx  # 假设像素是正方形
        cx = width / 2.0
        cy = height / 2.0
        
        # 构建内参矩阵
        intrin = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # 4. 设置单位变换
        post_rot = identity_rot.copy()
        post_tran = identity_post_tran.copy()
        
        # 5. 添加到列表
        trans.append(tran)
        rots.append(rot)
        intrins.append(intrin)
        post_rots.append(post_rot)
        post_trans.append(post_tran)
    bdas.append(identity_bda)
        # 收集所有参数
        # params = {
        #     'name': cam_name,
        #     'tran': tran,
        #     'rot': rot,
        #     'intrin': intrin,
        #     'post_rot': post_rot,
        #     'post_tran': post_tran,
        #     'bda': bda
        # }
    camera_params = [rots, trans, intrins, post_rots, post_trans, bdas]
    return camera_params


# 使用示例
if __name__ == "__main__":
    # 获取所有相机参数
    bs = 8
    all_camera_params = get_sensor_params()
    for i in range(len(all_camera_params)):
        all_camera_params[i] = torch.tensor([all_camera_params[i] for _ in range(bs)], dtype=torch.float32)
    pass
