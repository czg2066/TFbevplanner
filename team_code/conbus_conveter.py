import numpy as np
from pyquaternion import Quaternion # 需要安装: pip install pyquaternion
import copy, gzip, ujson
import math

class canbus_dataconverter:
    def __init__(self):
        self.prev_pos = None
        self.prev_angle_deg = None
    # Helper function from original logic (if not already defined elsewhere)
    def quaternion_yaw(self, q: Quaternion) -> float:
        """
        Calculate the yaw angle from a quaternion.
        Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
        It does not work for a box in the camera frame.
        :param q: Quaternion of interest.
        :return: Yaw angle in radians.
        """
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0])) # Get heading vector
        yaw = np.arctan2(v[1], v[0]) # q.yaw_pitch_roll[0] might be more direct if q is 
        return yaw

    def _get_can_bus_info_from_custom_json(self, data_dict):
        """
        Generates an 18-element CAN bus array from the custom JSON data structure.
        The structure aims to be compatible with the original nuScenes CAN bus processing.
        """
        can_bus = []
        ego_matrix = np.array(data_dict['ego_matrix'])
        pos_global = ego_matrix[0:3, 3].tolist() # x, y, z
        can_bus.extend(pos_global) # [0:3]
        # rotation_matrix_global_ego = ego_matrix[0:3, 0:3]
        rotation_matrix_input = ego_matrix[0:3, 0:3]
        U, S, Vt = np.linalg.svd(rotation_matrix_input)
        rotation_matrix_orthogonal = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        orientation_quat = Quaternion(matrix=rotation_matrix_orthogonal)
        can_bus.extend(orientation_quat.elements) # [w, x, y, z] -> [3:7]
        if 'accelerometer' in data_dict:
            accel = data_dict['accelerometer'] # Should be [ax, ay, az]
            if len(accel) == 3:
                can_bus.extend(accel) # [7:10]
            else:
                # Fallback or error if format is unexpected
                print(f"Warning: 'accelerometer' data has unexpected length: {len(accel)}. Using zeros.")
                can_bus.extend([0.0, 0.0, 0.0])
        else:
            print("Warning: 'accelerometer' not found in data_dict. Using zeros for acceleration.")
            can_bus.extend([0.0, 0.0, 0.0])
        if 'gyroscope' in data_dict:
            rotation_rate = data_dict['gyroscope'] # Should be [gx, gy, gz]
            if len(rotation_rate) == 3:
                can_bus.extend(rotation_rate) # [10:13]
            else:
                print(f"Warning: 'gyroscope' data has unexpected length: {len(rotation_rate)}. Using zeros.")
                can_bus.extend([0.0, 0.0, 0.0])
        else:
            print("Warning: 'gyroscope' not found in data_dict. Using zeros for rotation_rate.")
            can_bus.extend([0.0, 0.0, 0.0])
        if 'speed' in data_dict:
            speed = data_dict['speed']
            vel_ego = [float(speed), 0.0, 0.0]
            can_bus.extend(vel_ego) # [13:16]
        else:
            print("Warning: 'speed' not found in data_dict. Using zeros for velocity.")
            can_bus.extend([0.0, 0.0, 0.0])

        # 6. Reserved for patch_angle calculations
        can_bus.extend([0.0, 0.0]) # [16:18]

        if len(can_bus) != 18:
            raise ValueError(f"Generated can_bus has length {len(can_bus)}, expected 18. Content: {can_bus}")

        return np.array(can_bus)
    def enhance_can_bus_custom(self, input_dict_custom, queue_custom, metas_map_custom):
        """
        Enhances the can_bus data using the custom JSON structure.
        `input_dict_custom` is the current frame's data (your JSON).
        `queue_custom` is a list of previous frames' data (dicts like input_dict_custom).
        `metas_map_custom` will store the processed 'can_bus' and other meta info.
        """
        current_can_bus = self._get_can_bus_info_from_custom_json(input_dict_custom)
        ego_matrix_np = np.array(input_dict_custom['ego_matrix'])
        accurate_translation = ego_matrix_np[0:3, 3]
        # accurate_rotation_matrix = ego_matrix_np[0:3, 0:3]
        rotation_matrix_input = ego_matrix_np[0:3, 0:3]
        U, S, Vt = np.linalg.svd(rotation_matrix_input)
        rotation_matrix_orthogonal = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        accurate_rotation_quat = Quaternion(matrix=rotation_matrix_orthogonal)
        current_can_bus[:3] = accurate_translation
        current_can_bus[3:7] = accurate_rotation_quat.elements # [w, x, y, z]
        patch_angle_rad = self.quaternion_yaw(accurate_rotation_quat) # Yaw in radians
        patch_angle_deg = patch_angle_rad / np.pi * 180         # Yaw in degrees
        
        current_can_bus[-2] = patch_angle_rad # Store radians
        current_can_bus[-1] = patch_angle_deg # Store degrees (original code used this for delta)
        input_dict_custom['can_bus_processed'] = current_can_bus
                
        return input_dict_custom # Or metas_map_custom if processing a queue

    # --- Refined enhancement part to process a queue ---
    def process_frames_with_can_enhancement(self, current_frame_data, idx):
        # 1. Get 'raw' can_bus for current_frame_data
        can_bus_for_frame = self._get_can_bus_info_from_custom_json(current_frame_data)
        
        # 2. Enhance with accurate ego pose from ego_matrix
        ego_m = np.array(current_frame_data['ego_matrix'])
        accurate_translation = ego_m[0:3, 3]
        rotation_matrix_input = ego_m[0:3, 0:3]
        U, S, Vt = np.linalg.svd(rotation_matrix_input)
        rotation_matrix_orthogonal = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        accurate_rotation_q = Quaternion(matrix=rotation_matrix_orthogonal)
        
        can_bus_for_frame[:3] = accurate_translation
        can_bus_for_frame[3:7] = accurate_rotation_q.elements
        
        patch_angle_rad = self.quaternion_yaw(accurate_rotation_q)
        patch_angle_deg = patch_angle_rad / np.pi * 180
        
        can_bus_for_frame[-2] = patch_angle_rad
        can_bus_for_frame[-1] = patch_angle_deg # Store absolute degrees here
        
        # Prepare meta data for this frame
        current_meta = {'can_bus_abs': copy.deepcopy(can_bus_for_frame), 'prev_bev': False}
        # The 'can_bus' field in meta will store the *relative* can_bus info

        if idx == 0 or self.prev_pos is None:
            current_meta['prev_bev'] = False
            self.prev_pos = copy.deepcopy(can_bus_for_frame[:3]) # Absolute position
            self.prev_angle_deg = copy.deepcopy(can_bus_for_frame[-1])    # Absolute angle in degrees
            
            # For the first frame, relative to itself is zero
            # We modify a copy for the 'relative' can_bus
            relative_can_bus = copy.deepcopy(can_bus_for_frame)
            relative_can_bus[:3] = np.array([0.0, 0.0, 0.0])
            relative_can_bus[-1] = 0.0 # Relative angle (degrees)
            current_meta['can_bus'] = relative_can_bus
        else:
            current_meta['prev_bev'] = True
            
            tmp_pos_abs = copy.deepcopy(can_bus_for_frame[:3])
            tmp_angle_abs_deg = copy.deepcopy(can_bus_for_frame[-1])
            
            relative_can_bus = copy.deepcopy(can_bus_for_frame)
            relative_can_bus[:3] -= self.prev_pos # Relative position
            relative_can_bus[-1] -= self.prev_angle_deg # Relative angle in degrees
            current_meta['can_bus'] = relative_can_bus
            
            self.prev_pos = tmp_pos_abs
            self.prev_angle_deg = tmp_angle_abs_deg
            
        return current_meta

if __name__ == "__main__":
    with gzip.open("/media/czg/DriveLab_Datastorage/geely_data/rl/Town03_rl_route1_04_29_22_37_00/measurements/0001.json.gz", 'rt', encoding='utf-8') as f1:
        measurements_i = ujson.load(f1)
    # queue = [measurements_i, measurements_i] # Example with two identical frames for testing queue logic
    # metas_map = [{}, {}] # Initialize metas_map
    # Process the queue
    a = canbus_dataconverter()
    for i in range(2):
        processed_data = a.process_frames_with_can_enhancement(measurements_i, i)

    print("\nProcessed data for queue (first item):")
    print("Absolute CAN bus (from 'can_bus_abs'):")
    print(processed_data['can_bus_abs'])
    print("Relative CAN bus (from 'can_bus'):") # pos and angle should be 0 for the first frame
    print(processed_data['can_bus'])


    if len(processed_data) > 1:
        print("\nProcessed data for queue (second item):")
        print("Absolute CAN bus (from 'can_bus_abs'):")
        print(processed_data['can_bus_abs'])
        print("Relative CAN bus (from 'can_bus'):") # pos and angle relative to the first
        print(processed_data['can_bus'])

