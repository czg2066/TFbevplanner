import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

class Lidar2ImgCalculator:
    def __init__(self, config):
        # 相机相对于车辆的位姿 [x, y, z (m), roll, pitch, yaw (degrees)]
        # 注意：根据您的描述，roll, pitch, yaw 已经是度数了
        # 您的原始 self.cam_trans[0] = [2.549, 0.933, 0.823, -0.044, 0.008, 0.841] (角度是弧度)
        # 然后您做了 convert_rad_to_deg，所以现在是度数
        # 我将直接使用度数值，如果您的 self.cam_trans 已经是度数，请直接用它
        # 如果您的 self.cam_trans 仍然是弧度，请取消下面 rad2deg 的注释
        self.cam_extrinsic_params_vehicle_coord = [
            # [x, y, z, roll_deg, pitch_deg, yaw_deg]
            [2.549, 0.933, 0.823, np.rad2deg(-0.044), np.rad2deg(0.008), np.rad2deg(0.841)], # Cam 0
            [2.549, 0.933, 0.823, np.rad2deg(0.009), np.rad2deg(-0.021), np.rad2deg(-0.836)], # Cam 1
            # [1.933, -0.033, 1.311, np.rad2deg(-0.003), np.rad2deg(-0.033), np.rad2deg(-0.005)], # Cam 2
            [1.933, -0.003, 1.311, np.rad2deg(-0.000), np.rad2deg(-0.036), np.rad2deg(-0.007)], # Cam 3
            [2.457, 0.937, 0.827, np.rad2deg(-0.004), np.rad2deg(-0.011), np.rad2deg(2.363)], # Cam 4
            [2.457, -0.937, 0.827, np.rad2deg(0.038), np.rad2deg(0.004), np.rad2deg(-2.358)], # Cam 5
            [0.006, 0.0, 2.025, np.rad2deg(0.005), np.rad2deg(0.026), np.rad2deg(3.119)]  # Cam 6
        ]
        # 如果您的 self.cam_trans 已经是度数，则可以直接赋值：
        # self.cam_extrinsic_params_vehicle_coord = self.cam_trans # 假设 self.cam_trans 已经是处理好的度数列表

        # self.rgb_wh = [[3840, 2160] if idx < 2 else [1920, 1280] for idx in range(7)] # 简化版，对应 "tele" or "wide"
        self.rgb_wh = [[config.img_w, config.img_h] for _ in range(6)] # 图像压缩后的尺寸
        # self.rgb_wh = [[320, 180] for _ in range(6)] # 图像压缩后的尺寸
        self.cam_fov_horizontal_deg = [100, 100, 120, 100, 100, 100]

        # 从CARLA相机坐标系 (X fwd, Y left, Z up) 到 OpenCV相机坐标系 (Z fwd, X right, Y down)
        self.R_opencv_carla_cam = np.array([
            [0, -1,  0], # OpenCV X = -CARLA Y
            [0,  0, -1], # OpenCV Y = -CARLA Z
            [1,  0,  0]  # OpenCV Z =  CARLA X
        ])
        self.T_opencv_carla_cam = np.eye(4)
        self.T_opencv_carla_cam[:3, :3] = self.R_opencv_carla_cam

    def get_lidar2img_matrices(self):
        lidar2img_all_cams = []

        for i in range(len(self.cam_extrinsic_params_vehicle_coord)):
            params = self.cam_extrinsic_params_vehicle_coord[i]
            x, y, z, roll_deg, pitch_deg, yaw_deg = params

            # 1. 计算 T_camera_vehicle (从车辆坐标系到CARLA相机坐标系)
            #    首先得到 T_vehicle_camera (相机在车辆坐标系下的位姿)
            #    CARLA中 carla.Rotation(pitch, yaw, roll) 的旋转顺序是 Y, Z, X (intrinsic)
            #    对应于 scipy.spatial.transform.Rotation.from_euler('yzx', [pitch, yaw, roll], degrees=True)
            R_vehicle_camera_obj = ScipyRotation.from_euler('yzx', [pitch_deg, yaw_deg, roll_deg], degrees=True)
            R_vehicle_camera = R_vehicle_camera_obj.as_matrix()
            t_vehicle_camera = np.array([x, y, z])

            T_vehicle_camera = np.eye(4)
            T_vehicle_camera[:3, :3] = R_vehicle_camera
            T_vehicle_camera[:3, 3] = t_vehicle_camera

            #    T_camera_vehicle = (T_vehicle_camera)^-1
            #    R_camera_vehicle = R_vehicle_camera.T
            #    t_camera_vehicle = -R_vehicle_camera.T @ t_vehicle_camera
            R_camera_vehicle = R_vehicle_camera.T
            t_camera_vehicle = -R_camera_vehicle @ t_vehicle_camera

            T_camera_carla_vehicle = np.eye(4) # 从车辆坐标系到CARLA相机坐标系
            T_camera_carla_vehicle[:3, :3] = R_camera_vehicle
            T_camera_carla_vehicle[:3, 3] = t_camera_vehicle

            # 2. 转换到OpenCV相机坐标系
            # T_camera_opencv_vehicle = T_opencv_carla_cam @ T_camera_carla_vehicle
            T_camera_opencv_vehicle = self.T_opencv_carla_cam @ T_camera_carla_vehicle
            
            # (雷达到车辆的变换 T_vehicle_lidar 是单位阵，所以 T_camera_opencv_lidar = T_camera_opencv_vehicle)
            T_extrinsic_lidar_to_opencv_cam = T_camera_opencv_vehicle

            # 3. 计算相机内参 K 并填充到 K_padded
            W, H = self.rgb_wh[i]
            fov_h_deg = self.cam_fov_horizontal_deg[i]
            fov_h_rad = np.deg2rad(fov_h_deg)

            cx = W / 2.0
            cy = H / 2.0
            fx = (W / 2.0) / np.tan(fov_h_rad / 2.0)
            fy = fx # 假设方形像素

            K_padded = np.array([
                [fx,  0, cx, 0],
                [ 0, fy, cy, 0],
                [ 0,  0,  1, 0], # 这个1是为了在乘以点之后，保持点的z_cam值，以便后续归一化
                [ 0,  0,  0, 1]
            ])

            # 4. 计算 lidar2img
            lidar2img = K_padded @ T_extrinsic_lidar_to_opencv_cam
            lidar2img_all_cams.append(lidar2img)

        return lidar2img_all_cams

    def build_projection_matrix(self, w, h, fov, is_behind_camera=False):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)

        if is_behind_camera:
            K[0, 0] = K[1, 1] = -focal
        else:
            K[0, 0] = K[1, 1] = focal

        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    def get_matrix(self):
        """
        Creates matrix from carla transform.
        根据CARLA的约定（通常是Z-Y-X欧拉角顺序应用：yaw, pitch, roll 来从父坐标系转到自身坐标系）
        计算传感器相对于车辆的变换矩阵 T_vehicle_from_sensor。
        这个矩阵可以将传感器坐标系下的点变换到车辆坐标系下。
        输入角度为度数。
        """
        matrixs = []
        for i in range(len(self.cam_extrinsic_params_vehicle_coord)):
            params = self.cam_extrinsic_params_vehicle_coord[i]
            x, y, z, roll, pitch, yaw = params
   
            # CARLA的变换通常是：
            # 1. 平移到车辆坐标系中的 (x,y,z) 位置
            # 2. 然后应用旋转 (yaw, pitch, roll) 来确定传感器的朝向
            # 旋转顺序通常是 yaw (Z轴), pitch (新的Y轴), roll (新的X轴)
            # 将度数转换为弧度
            yaw = np.radians(yaw)
            pitch = np.radians(pitch)
            roll = np.radians(roll)
            # 计算旋转矩阵 R_vehicle_from_sensor
            # 这是标准的 ZYX 欧拉角旋转矩阵 (先绕Z轴yaw, 再绕Y轴pitch, 最后绕X轴roll)
            # R = Rz(yaw) * Ry(pitch) * Rx(roll)
            c_y = np.cos(yaw)
            s_y = np.sin(yaw)
            c_p = np.cos(pitch)
            s_p = np.sin(pitch)
            c_r = np.cos(roll)
            s_r = np.sin(roll)
            # Rotation matrix for ZYX order
            # # Rz(yaw)
            # Rz = np.array([[c_y, -s_y, 0],
            #             [s_y,  c_y, 0],
            #             [  0,    0, 1]])
            # # Ry(pitch)
            # Ry = np.array([[ c_p, 0, s_p],
            #             [   0, 1,   0],
            #             [-s_p, 0, c_p]])
            # # Rx(roll)
            # Rx = np.array([[1,   0,   0],
            #             [0, c_r, -s_r],
            #             [0, s_r,  c_r]])
            # R_vehicle_from_sensor = Rz @ Ry @ Rx
            # 注意：你提供的 get_matrix 代码中的旋转矩阵构造方式是直接给出了组合后的结果，
            # 它的形式对应于将一个在“自身坐标系”的点，通过这个旋转，变换到“父坐标系”。
            # 我们需要确认这个旋转的含义。
            # 如果 (roll, pitch, yaw) 定义了从车辆坐标系到传感器坐标系的旋转，
            # 那么 R_sensor_from_vehicle = Rz(yaw)Ry(pitch)Rx(roll)
            # 则 R_vehicle_from_sensor = (R_sensor_from_vehicle).T
            # 我们采用你 get_matrix 中的直接构造（它代表 R_parent_from_body）
            rotation_matrix = np.identity(3)
            rotation_matrix[0, 0] = c_p * c_y
            rotation_matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
            rotation_matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r # CARLA Z轴向前的话，这里可能是正号
            rotation_matrix[1, 0] = s_y * c_p
            rotation_matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
            rotation_matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r # CARLA Z轴向前的话，这里可能是正号
            rotation_matrix[2, 0] = s_p
            rotation_matrix[2, 1] = -c_p * s_r
            rotation_matrix[2, 2] = c_p * c_r
            # 构建4x4齐次变换矩阵 T_vehicle_from_sensor
            T_vehicle_from_sensor = np.identity(4)
            T_vehicle_from_sensor[:3, :3] = rotation_matrix
            T_vehicle_from_sensor[:3, 3] = [x, y, z]
            matrixs.append(T_vehicle_from_sensor) # T_vehicle_from_sensor
        return matrixs

    def get_lidar2img(self):
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        mats = self.get_matrix()
        for i in range(len(mats)):
            mat = mats[i]
            w_cam,h_cam = self.rgb_wh[i]
            fov_cam = self.cam_fov_horizontal_deg[i]
            # print(f"cam{i}: w_cam={w_cam}, h_cam={h_cam}, fov_cam={fov_cam}")
            cam_intrinsic = self.build_projection_matrix(w_cam, h_cam, fov_cam)
            sensor2lidar_rotation = mat[:3, :3]
            sensor2lidar_translation = mat[:3, 3]
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
            lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_intrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)

            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
        lidar2cam=lidar2cam_rts
        return lidar2cam

# --- 使用示例 ---
if __name__ == '__main__':
    calculator = Lidar2ImgCalculator()
    
    # 假设您的类中 self.cam_trans 已经是角度为度的列表
    # 例如：
    # calculator.cam_extrinsic_params_vehicle_coord = your_instance.cam_trans
    # calculator.rgb_wh = your_instance.rgb_wh
    # calculator.cam_fov_horizontal_deg = your_instance.cam_fov

    lidar2img_matrices = calculator.get_lidar2img_matrices()

    for i, mat in enumerate(lidar2img_matrices):
        print(f"--- Camera {i} lidar2img ---")
        print(mat)
        # 验证：取一个雷达点，例如车辆前方10米，在雷达/车辆中心线上 (10,0,0,1)
        # 由于雷达在车辆中心，所以雷达点 (10,0,0,1) 也是车辆坐标系点 (10,0,0,1)
        # 如果雷达不在车辆中心，比如雷达在车辆坐标系 (lx,ly,lz)，则 T_vehicle_lidar 的平移部分是 [lx,ly,lz]
        # 在本例中，T_vehicle_lidar 是单位阵
        
        # 车辆坐标系点 (X=向前, Y=向右, Z=向上)
        # lidar_point_vehicle_frame = np.array([10, 0, 0, 1]) # 车辆前方10米
        # lidar_point_vehicle_frame = np.array([5, 2, 1, 1]) # 车辆前方5米，右方2米，上方1米

        # # 变换到OpenCV相机坐标系
        # T_ext = calculator.T_opencv_carla_cam @ np.linalg.inv(calculator.T_vehicle_camera_matrices_for_debug[i]) # Just for one cam
        # point_in_opencv_cam_coord = T_ext @ lidar_point_vehicle_frame
        
        # print(f"Point in OpenCV Cam {i} coords: {point_in_opencv_cam_coord}")

        # # 应用内参投影 (K_padded @ point_in_opencv_cam_coord)
        # # 或者直接用 lidar2img
        # projected_homo = mat @ lidar_point_vehicle_frame
        # print(f"Projected homogeneous point: {projected_homo}")

        # if projected_homo[2] > 0: # Z值（深度）必须为正才能在相机前方
        #     px = projected_homo[0] / projected_homo[2]
        #     py = projected_homo[1] / projected_homo[2]
        #     print(f"Pixel coordinates (Cam {i}): u={px:.2f}, v={py:.2f}, depth_in_cam={projected_homo[2]:.2f}")
            
        #     W, H = calculator.rgb_wh[i]
        #     if 0 <= px < W and 0 <= py < H:
        #         print("Point is within image bounds.")
        #     else:
        #         print("Point is outside image bounds.")
        # else:
        #     print("Point is behind the camera or on the focal plane.")
