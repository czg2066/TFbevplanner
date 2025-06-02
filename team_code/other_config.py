import os
import numpy as np
import math

def convert_rad_to_deg(cam_trans):
    for i in range(len(cam_trans)):
        if cam_trans[i]:
            x, y, z, roll_rad, pitch_rad, yaw_rad = cam_trans[i]
            roll_deg = math.degrees(roll_rad)
            pitch_deg = math.degrees(pitch_rad)
            yaw_deg = math.degrees(yaw_rad)
            cam_trans[i] = [x, y, z, roll_deg, pitch_deg, yaw_deg]
    return cam_trans

class Geely_sensors():
    def __init__(self, rgb=True, semantics=True, depth=True, radar=True):
        self.cam = ["front_left", "front_right", "front_tele", "front_wide", "rear_left", "rear_right", "rear"] 
        self.rgb = ["rgb_"+i for i in self.cam] 
        self.semantics = ["semantic_"+i for i in self.cam] 
        self.depth = ["depth_"+i for i in self.cam]
        self.cam_trans = [[] for i in self.cam] #x,y,z,roll,pitch,yaw
        self.cam_trans[0] = [2.549, 0.933, 0.823, -0.044, 0.008, 0.841]
        self.cam_trans[1] = [2.549, 0.933, 0.823, 0.009, -0.021, -0.836]
        self.cam_trans[2] = [1.933, -0.033, 1.311, -0.003, -0.033, -0.005]
        self.cam_trans[3] = [1.933, -0.003, 1.311, -0.000, -0.036, -0.007]
        self.cam_trans[4] = [2.457, 0.937, 0.827, -0.004, -0.011, 2.363]
        self.cam_trans[5] = [2.457, -0.937, 0.827, 0.038, 0.004, -2.358]
        self.cam_trans[6] = [0.006, 0.0, 2.025, 0.005, 0.026, 3.119]
        self.cam_trans = convert_rad_to_deg(self.cam_trans)
        self.rgb_wh = [[3840, 2160] if "tele" in i or "wide" in i else [1920, 1280] for i in self.rgb]
        self.cam_fov = [100, 100, 30, 120, 100, 100, 100]
        self.radar = ["radar_front", "radar_rear_left", "radar_rear_right"]
        self.radar_trans = [[] for i in self.radar] #x,y,z,roll,pitch,yaw
        self.radar_trans[0] = [3.802, 0.116, 0.59, 0, 0, 0]
        self.radar_trans[1] = [-0.859, 0.777, 0.684, 0, 0, -2.358]
        self.radar_trans[2] = [-0.859, 0.777, 0.684, 0, 0, 2.358]
        self.radar_trans = convert_rad_to_deg(self.radar_trans)
        self.radar_fov = [[60, 30], [150, 30], [150, 30]]
        self.radar_range = 50

        self.sensors_list = self.rgb + self.radar + self.semantics + self.depth
        self.sensors_set = self.set_sensors(rgb=rgb, semantics=semantics, depth=depth, radar=radar)
    
    def  set_sensors(self, rgb=True, semantics=True, depth=True, radar=True):
        sensors = []
        if rgb:
            for i in range(len(self.rgb)):
                    sensors.append({
                        'type': 'sensor.camera.rgb',
                        'x': self.cam_trans[i][0], 'y': self.cam_trans[i][1], 'z': self.cam_trans[i][2],
                        'roll': self.cam_trans[i][3], 'pitch': self.cam_trans[i][4], 'yaw': self.cam_trans[i][5],
                        'width':  self.rgb_wh[i][0], 'height': self.rgb_wh[i][1], 'fov': self.cam_fov[i],
                        'id': self.rgb[i]
                    })
        if semantics:
            for i in range(len(self.semantics)):
                    sensors.append({
                        'type': 'sensor.camera.semantic_segmentation',
                        'x': self.cam_trans[i][0], 'y': self.cam_trans[i][1], 'z': self.cam_trans[i][2],
                        'roll': self.cam_trans[i][3], 'pitch': self.cam_trans[i][4], 'yaw': self.cam_trans[i][5],
                        'width':  self.rgb_wh[i][0], 'height': self.rgb_wh[i][1], 'fov': self.cam_fov[i],
                        'id': self.semantics[i]
                })
        if depth:
            for i in range(len(self.depth)):
                    sensors.append({
                        'type': 'sensor.camera.depth',
                        'x': self.cam_trans[i][0], 'y': self.cam_trans[i][1], 'z': self.cam_trans[i][2],
                        'roll': self.cam_trans[i][3], 'pitch': self.cam_trans[i][4], 'yaw': self.cam_trans[i][5],
                        'width':  self.rgb_wh[i][0], 'height': self.rgb_wh[i][1], 'fov': self.cam_fov[i],
                        'id': self.depth[i]
                    })
        if radar:
            for i in range(len(self.radar)):
                    sensors.append({
                        'type': 'sensor.other.radar',
                        'x': self.radar_trans[i][0], 'y': self.radar_trans[i][1], 'z': self.radar_trans[i][2], 
                        'roll': self.radar_trans[i][3], 'pitch': self.radar_trans[i][4], 'yaw': self.radar_trans[i][5],
                        'horizontal_fov': self.radar_fov[i][0], 'vertical_fov': self.radar_fov[i][1],
                        'range': self.radar_range,
                        'id': self.radar[i]
                    })
        return sensors

class nuScenes_set():
    '''
    来自github.com Thinklab-SJTU Bench2DriveZoo的VAD分支的设置, VAD分支采用nuSenes数据集训练, 设置适配nuSenes数据集的传感器配置, 但是传感器略有位置不同
    这个设置就是6个摄像头、5个毫米波雷达和1个激光雷达的配置
    '''
    def __init__(self, rgb=1, semantic=1, depth=1, lidar=1, radar=1):
        # 传感器配置
        ## 摄像头配置
        self.rgb = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"] # 6个摄像头
        self.segment = ["SEMANTIC_"+i for i in self.rgb] # 6个摄像头
        self.depth = ["DEPTH_"+i for i in self.rgb] # 6个摄像头
        self.rgb_trans = [[] for i in range(len(self.rgb))] # 摄像头位置
        self.rgb_trans[0] = [0.80, 0.0, 1.60, 0.0, 0.0, 0.0]   #x,y,z,roll,pitch,yaw    CAM_FRONT
        self.rgb_trans[1] = [0.27, -0.55, 1.60, 0.0, 0.0, -55.0]   #x,y,z,roll,pitch,yaw     CAM_FRONT_LEFT
        self.rgb_trans[2] = [0.27, 0.55, 1.60, 0.0, 0.0, 55.0]   #x,y,z,roll,pitch,yaw     CAM_FRONT_RIGHT
        self.rgb_trans[3] = [-2.0, 0.0, 1.60, 0.0, 0.0, 180.0]   #x,y,z,roll,pitch,yaw    CAM_BACK
        self.rgb_trans[4] = [-.32, -0.55, 1.60, 0.0, 0.0, -110.0]   #x,y,z,roll,pitch,yaw   CAM_BACK_LEFT
        self.rgb_trans[5] = [-.32, 0.55, 1.60, 0.0, 0.0, 110.0]   #x,y,z,roll,pitch,yaw   CAM_BACK_RIGHT
        self.rgb_width = 1600 # 摄像头分辨率宽
        self.rgb_height = 900 # 摄像头分辨率高
        self.rgb_normal_fov = 70 # 其它视角的fov
        self.rgb_back_fov = 110  # 后视角的fov
        ## 激光雷达配置
        self.lidar = ["LIDAR_TOP"] # 1个激光雷达
        self.lidar_trans = [[0.0, 0.0, 2.50, 0.0, 0.0, -90.0]] # 激光雷达位置
        self.lidar_channels = 32    # 激光雷达通道数
        self.lidar_range = 70.0     # 激光雷达探测范围
        self.lidar_points_per_second = 1400000 # 激光雷达每秒点数
        self.lidar_rotation_frequency = 20.0 # 激光雷达旋转频率
        self.lidar_horizontal_fov = 360.0 # 激光雷达水平视角
        self.lidar_upper_fov = 10.0 # 激光雷达垂直抬角
        self.lidar_lower_fov = -30.0 # 激光雷达垂直俯角
        ## 毫米波雷达配置
        self.radar = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"] # 5个毫米波雷达
        self.radar_trans = [[] for i in range(len(self.radar))]
        self.radar_trans[0] = [1.20, 0.0, 1.0, 0.0, 0.0, 0.0] #x,y,z,roll,pitch,yaw    RADAR_FRONT
        self.radar_trans[1] = [1.20, -0.55, 1.0, 0.0, 0.0, -90.0]   #x,y,z,roll,pitch,yaw     RADAR_FRONT_LEFT
        self.radar_trans[2] = [1.20, 0.55, 1.0, 0.0, 0.0, 90.0]   #x,y,z,roll,pitch,yaw     RADAR_FRONT_RIGHT
        self.radar_trans[3] = [-1.20, -0.55, 1.0, 0.0, 0.0, 180.0]   #x,y,z,roll,pitch,yaw   RADAR_BACK_LEFT
        self.radar_trans[4] = [-1.20, 0.55, 1.0, 0.0, 0.0, 180.0]   #x,y,z,roll,pitch,yaw   RADAR_BACK_RIGHT
        self.radar_range = 250.0 # 毫米波雷达探测范围
        # sensors
        self.sensors_list = []
        if rgb: self.sensors_list += self.rgb
        if semantic: self.sensors_list += self.segment
        if depth: self.sensors_list += self.depth
        if lidar: self.sensors_list += self.lidar
        if radar: self.sensors_list += self.radar
        self.sensors = self.get_sensors(rgb, semantic, depth, lidar, radar)
    def get_sensors(self, rgb, semantic, depth, lidar, radar):
        sensors = []
        ## rgb摄像头
        if rgb:
            for i in range(len(self.rgb)):
                if i != len(self.rgb)-1: fov = self.rgb_normal_fov
                else: fov = self.rgb_back_fov
                sensors.append({
                    'type': 'sensor.camera.rgb',
                    'x': self.rgb_trans[i][0], 'y': self.rgb_trans[i][1], 'z': self.rgb_trans[i][2],
                    'roll': self.rgb_trans[i][3], 'pitch': self.rgb_trans[i][4], 'yaw': self.rgb_trans[i][5],
                    'width': self.rgb_width, 'height': self.rgb_height, 'fov': fov,
                    'id': self.rgb[i]
                })
        ## semantic摄像头
        if semantic:
            for i in range(len(self.segment)):
                sensors.append({
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': self.rgb_trans[i][0], 'y': self.rgb_trans[i][1], 'z': self.rgb_trans[i][2], 
                    'roll': self.rgb_trans[i][3], 'pitch': self.rgb_trans[i][4], 'yaw': self.rgb_trans[i][5],
                    'width': self.rgb_width, 'height': self.rgb_height, 'fov': self.rgb_normal_fov,
                    'id': self.segment[i]
                })
        ## depth摄像头
        if depth:
            for i in range(len(self.depth)):
                sensors.append({
                    'type': 'sensor.camera.depth',
                    'x': self.rgb_trans[i][0], 'y': self.rgb_trans[i][1], 'z': self.rgb_trans[i][2],
                    'roll': self.rgb_trans[i][3], 'pitch': self.rgb_trans[i][4], 'yaw': self.rgb_trans[i][5],
                    'width': self.rgb_width, 'height': self.rgb_height, 'fov': self.rgb_normal_fov,
                    'id': self.depth[i]
                })
        ## 激光雷达
        if lidar:
            for i in range(len(self.lidar)):
                sensors.append({
                    'type': 'sensor.lidar.ray_cast',
                    'x': self.lidar_trans[i][0], 'y': self.lidar_trans[i][1], 'z': self.lidar_trans[i][2],
                    'roll': self.lidar_trans[i][3], 'pitch': self.lidar_trans[i][4], 'yaw': self.lidar_trans[i][5],
                    'channels': self.lidar_channels, 'range': self.lidar_range, 'rotation_frequency': self.lidar_rotation_frequency,
                    'points_per_second': self.lidar_points_per_second, 'upper_fov': self.lidar_upper_fov, 'lower_fov': self.lidar_lower_fov,
                    'id': self.lidar[i]
                })
        if radar:
            ## 毫米波雷达
            for i in range(len(self.radar)):
                sensors.append({
                    'type': 'sensor.other.radar',
                    'x': self.radar_trans[i][0], 'y': self.radar_trans[i][1], 'z': self.radar_trans[i][2], 
                    'roll': self.radar_trans[i][3], 'pitch': self.radar_trans[i][4], 'yaw': self.radar_trans[i][5],
                    'horizontal_fov': 30.0, 'vertical_fov': 30.0,
                    'range': self.radar_range,
                    'id': self.radar[i]
                })
        return sensors

class TFpp_set():
    '''论文TF++的传感器复现，包含一个前视角的rgb、depth和semantic及它们的augmented摄像头、一个lidar'''
    def __init__(self, Aaugmentation=True):
        # 传感器配置
        ## 摄像头通用设置
        self.carmera_width = 1024   # 摄像头分辨率宽
        self.carmera_height = 512   # 摄像头分辨率高
        self.carmera_fov = 110     # 摄像头水平视角
        self.carmera_trans = [-1.5, 0.0, 2.0, 0.0, 0.0, 0.0]  # 摄像头位置
        ### rgb摄像头
        self.rgb = ["rgb", "rgb_augmented"]
        ### 深度摄像头
        self.depth = ["depth", "depth_augmented"]
        ### 语义分割摄像头
        self.semantic = ["semantic", "semantic_augmented"]
        ## 激光雷达
        self.lidar = ["lidar"] # 1个激光雷达
        self.lidar_trans = [[0.0, 0.0, 2.5, 0.0, 0.0, -90.0]] # 激光雷达位置
        self.lidar_rotation_frequency = 10.0 # 激光雷达旋转频率
        self.lidar_points_per_second = 600000 # 激光雷达每秒点数
        self.lidar_resolution_width = 256 # 激光雷达分辨率宽
        self.lidar_resolution_height = 256 # 激光雷达分辨率高
        # sensors
        self.sensors_list = self.rgb+self.depth+self.semantic+self.lidar
        self.sensors = self.get_sensors()
        self.Aaugmentation = Aaugmentation
        self.camera_rotation_augmentation_min = -5.0
        self.camera_rotation_augmentation_max = 5.0
        self.camera_translation_augmentation_min = -1.0
        self.camera_translation_augmentation_max = 1.0
        self.augmentation_translation = np.random.uniform(low=self.config.camera_translation_augmentation_min,
                                                            high=self.config.camera_translation_augmentation_max)
        self.augmentation_rotation = np.random.uniform(low=self.config.camera_rotation_augmentation_min,
                                                        high=self.config.camera_rotation_augmentation_max)
    def get_sensors(self):
        sensors = []
        ## 摄像头
        for i in range(len(self.rgb)):
            sensors.append({
                'type': 'sensor.camera.rgb',
                'x': self.carmera_trans[0], 'y': self.carmera_trans[1]+self.augmentation_translation \
                    if self.Aaugmentation and "augmented" in self.rgb[i] else self.carmera_trans[1]
                , 'z': self.carmera_trans[2],
                'roll': self.carmera_trans[3], 'pitch': self.carmera_trans[4], 'yaw': self.carmera_trans[5]+self.augmentation_rotation \
                if self.Aaugmentation and "augmented" in self.rgb[i] else self.carmera_trans[5], 
                'width': self.carmera_width, 'height': self.carmera_height, 'fov': self.carmera_fov,
                'id': self.rgb[i]
            })
        ## 深度摄像头
        for i in range(len(self.depth)):
            sensors.append({
                'type': 'sensor.camera.depth',
                'x': self.carmera_trans[0], 'y': self.carmera_trans[1]+self.augmentation_translation \
                    if self.Aaugmentation and "augmented" in self.depth[i] else self.carmera_trans[1]
                , 'z': self.carmera_trans[2],
                'roll': self.carmera_trans[3], 'pitch': self.carmera_trans[4], 'yaw': self.carmera_trans[5]+self.augmentation_rotation \
                if self.Aaugmentation and "augmented" in self.depth[i] else self.carmera_trans[5], 
                'width': self.carmera_width, 'height': self.carmera_height, 'fov': self.carmera_fov,
                'id': self.depth[i]
            })
        ## 语义分割摄像头
        for i in range(len(self.semantic)):
            sensors.append({
                'type': 'sensor.camera.semantic_segmentation',
                'x': self.carmera_trans[0], 'y': self.carmera_trans[1] + self.augmentation_translation \
                    if self.Aaugmentation and "augmented" in self.semantic[i] else self.carmera_trans[1]
                , 'z': self.carmera_trans[2],
                'roll': self.carmera_trans[3], 'pitch': self.carmera_trans[4], 'yaw': self.carmera_trans[5]+self.augmentation_rotation \
                if self.Aaugmentation and "augmented" in self.semantic[i] else self.carmera_trans[5],
                'width': self.carmera_width, 'height': self.carmera_height, 'fov': self.carmera_fov,
                'id': self.semantic[i]
            })
        ## 激光雷达
        for i in range(len(self.lidar)):
            sensors.append({
                'type': 'sensor.lidar.ray_cast',
                'x': self.lidar_trans[0][0], 'y': self.lidar_trans[0][1], 'z': self.lidar_trans[0][2],
                'roll': self.lidar_trans[0][3], 'pitch': self.lidar_trans[0][4], 'yaw': self.lidar_trans[0][5],
                'rotation_frequency': self.lidar_rotation_frequency,
                'points_per_second': self.lidar_points_per_second,
                'id': self.lidar[i]
            })
        return sensors
          
class Trans_or_inter_fuser_set():
    '''论文Transfuser和interfuser的传感器复现，Transfuser包含三个前视角的rgb、depth和semantic摄像头、一个lidar
    interfuser在Transfuser的基础上增加了一个后视角的rgb、depth和semantic摄像头,它俩的分辨率和fov不一样'''
    def __init__(self, used_model='Transfuser'):
        # 传感器配置
        ## 摄像头通用设置
        if used_model == 'Transfuser':
            self.carmera_width = 960
            self.carmera_height = 480
            self.carmera_fov = 120
        else:
            self.carmera_width = 800
            self.carmera_height = 600
            self.carmera_fov = 100
        self.carmera_trans = [[1.3, 0.0, 2.3, 0.0, 0.0, 0.0],
                              [1.3, 0.0, 2.3, 0.0, 0.0, -60.0],
                              [1.3, 0.0, 2.3, 0.0, 0.0, 60.0]]
        if used_model == 'Interfuser': self.carmera_trans.append([-1.3, 0.0, 2.3, 0.0, 0.0, 0.0])
        ### rgb摄像头
        self.rgb = ["rgb_front", "rgb_left", "rgb_right"]
        if used_model == 'Interfuser': 
            self.rgb = ["irgb_front", "irgb_left", "irgb_right", "irgb_rear"]
        ### 深度摄像头
        self.depth = ["depth_front", "depth_left", "depth_right"]
        if used_model == 'Interfuser': self.depth = ["dep_front", "dep_left", "dep_right"]
        ### 语义分割摄像头
        self.semantic = ["semantic_front", "semantic_left", "semantic_right"]
        if used_model == 'Interfuser':  self.semantic = ["seg_front", "seg_left", "seg_right"]
        ## 激光雷达
        self.lidar = ["lidar_fuser"] # 1个激光雷达
        self.lidar_trans = [[1.3, 0.0, 2.5, 0.0, 0.0, -90.0]] # 激光雷达位置
        self.lidar_rotation_frequency = 20.0 # 激光雷达旋转频率
        self.lidar_points_per_second = 1200000  # 激光雷达每秒点数
        self.lidar_resolution_width = 256 # 激光雷达分辨率宽
        self.lidar_resolution_height = 256 # 激光雷达分辨率高
        self.lidar_pixels_per_meter = 8.0 # 激光雷达像素/米，用于映射现实距离和雷达点云
        # sensors
        self.sensors_list = self.rgb+self.depth+self.semantic+self.lidar
        self.sensors = self.get_sensors()
    def get_sensors(self):
        sensors = []
        ## 摄像头
        for i in range(len(self.rgb)):
            sensors.append({
                'type': 'sensor.camera.rgb',
                'x': self.carmera_trans[i][0], 'y': self.carmera_trans[i][1], 'z': self.carmera_trans[i][2],
                'roll': self.carmera_trans[i][3], 'pitch': self.carmera_trans[i][4], 'yaw': self.carmera_trans[i][5],
                'width': self.carmera_width, 'height': self.carmera_height, 'fov': self.carmera_fov,
                'id': self.rgb[i]
            })
        ## 深度摄像头
        for i in range(len(self.depth)):
            sensors.append({
                'type': 'sensor.camera.depth',
                'x': self.carmera_trans[i][0], 'y': self.carmera_trans[i][1], 'z': self.carmera_trans[i][2],
                'roll': self.carmera_trans[i][3], 'pitch': self.carmera_trans[i][4], 'yaw': self.carmera_trans[i][5],
                'width': self.carmera_width, 'height': self.carmera_height, 'fov': self.carmera_fov,
                'id': self.depth[i]
            })
        ## 语义分割摄像头
        for i in range(len(self.semantic)):
            sensors.append({
                'type': 'sensor.camera.semantic_segmentation',
                'x': self.carmera_trans[i][0], 'y': self.carmera_trans[i][1], 'z': self.carmera_trans[i][2],
                'roll': self.carmera_trans[i][3], 'pitch': self.carmera_trans[i][4], 'yaw': self.carmera_trans[i][5],
                'width': self.carmera_width, 'height': self.carmera_height, 'fov': self.carmera_fov,
                'id': self.semantic[i]
            })
        ## 激光雷达
        for i in range(len(self.lidar)):
            sensors.append({
                'type': 'sensor.lidar.ray_cast',
                'x': self.lidar_trans[i][0], 'y': self.lidar_trans[i][1], 'z': self.lidar_trans[i][2],
                'roll': self.lidar_trans[i][3], 'pitch': self.lidar_trans[i][4], 'yaw': self.lidar_trans[i][5],
                'points_per_second': self.lidar_points_per_second,
                'rotation_frequency': self.lidar_rotation_frequency,
                'id': self.lidar[i]
            })
        return sensors

class DataConfig:
    """ 
    用于数据采集的config
    """
    def __init__(self, model_list='all'): 
        '''model_list为可选的模型列表, 默认使用所有模型'''
        #全局参数
        self.Tfpp_augmented = True # 是否使用TFpp的增强数据，仅对Tfpp有用
        self.Haveset_model = ["nuScenes", "TFpp", "Interfuser", "Transfuser"] # 可选的模型
        ###根据选择的模型列表，设置模型列表和模型列表对应的模型'''
        if model_list == 'all': model_list = self.Haveset_model
        else: self.model_list = model_list
        self.Uesd_model = []
        Finish_model = []
        for i in self.model_list:
            if i not in self.Haveset_model: raise ValueError(f"已设置的模型中不包含的模型{i}")
            if i == "Transfuser" and "Interfuser" not in Finish_model: 
                self.Uesd_model.append(Trans_or_inter_fuser_set(used_model='Transfuser'))
                Finish_model.append("Transfuser")
            if i == "Interfuser" and "Transfuser" not in Finish_model:
                self.Uesd_model.append(Trans_or_inter_fuser_set(used_model='Interfuser'))
                Finish_model.append("Interfuser")
            if i == "TFpp" and "TFpp" not in Finish_model:
                self.Uesd_model.append(TFpp_set(self.Tfpp_augmented))
                Finish_model.append("TFpp")
            if i == "nuScenes" and "nuScenes" not in Finish_model:
                self.Uesd_model.append(nuScenes_set())
                Finish_model.append("nuScenes")

    def get_rgb(self):
        rgb = []
        for i in self.Uesd_model: 
            rgb += i.rgb
        return rgb
    def get_depth(self):
        dep = []
        for i in self.Uesd_model: 
            dep += i.depth
        return dep
    def get_semantic(self):
        sem = []
        for i in self.Uesd_model: 
            sem += i.get_semantic()
        return sem
    def get_lidar(self):
        lid = []
        for i in self.Uesd_model:
            lid += i.get_lidar()
        return lid
    def get_sensors(self):
        sensors = []
        for i in self.Uesd_model:
            sensors += i.sensors
        return sensors

if __name__ == '__main__':
    data_config = nuScenes_set()
    print(data_config.sensors_list)