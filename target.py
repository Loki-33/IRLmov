import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R

class Retargeter:
 
    
    def __init__(self):
        # G1 joint names in order (29 DOF)
        self.g1_joints = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
        ]
        
        # Joint limits (from XML)
        self.joint_limits = {
            'left_hip_pitch_joint': (-2.5307, 2.8798),
            'left_hip_roll_joint': (-0.5236, 2.9671),
            'left_hip_yaw_joint': (-2.7576, 2.7576),
            'left_knee_joint': (-0.087267, 2.8798),
            'left_ankle_pitch_joint': (-0.87267, 0.5236),
            'left_ankle_roll_joint': (-0.2618, 0.2618),
            'right_hip_pitch_joint': (-2.5307, 2.8798),
            'right_hip_roll_joint': (-2.9671, 0.5236),
            'right_hip_yaw_joint': (-2.7576, 2.7576),
            'right_knee_joint': (-0.087267, 2.8798),
            'right_ankle_pitch_joint': (-0.87267, 0.5236),
            'right_ankle_roll_joint': (-0.2618, 0.2618),
            'waist_yaw_joint': (-2.618, 2.618),
            'waist_roll_joint': (-0.52, 0.52),
            'waist_pitch_joint': (-0.52, 0.52),
            'left_shoulder_pitch_joint': (-3.0892, 2.6704),
            'left_shoulder_roll_joint': (-1.5882, 2.2515),
            'left_shoulder_yaw_joint': (-2.618, 2.618),
            'left_elbow_joint': (-1.0472, 2.0944),
            'left_wrist_roll_joint': (-1.97222, 1.97222),
            'left_wrist_pitch_joint': (-1.61443, 1.61443),
            'left_wrist_yaw_joint': (-1.61443, 1.61443),
            'right_shoulder_pitch_joint': (-3.0892, 2.6704),
            'right_shoulder_roll_joint': (-2.2515, 1.5882),
            'right_shoulder_yaw_joint': (-2.618, 2.618),
            'right_elbow_joint': (-1.0472, 2.0944),
            'right_wrist_roll_joint': (-1.97222, 1.97222),
            'right_wrist_pitch_joint': (-1.61443, 1.61443),
            'right_wrist_yaw_joint': (-1.61443, 1.61443),
        }
        
        # MediaPipe landmark indices
        self.mp_idx = {
            'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
    
    def get_limb_angles_3d(self, p_parent, p_joint, p_child):
        """
        Calculate joint angles in 3D space
        Returns pitch, roll, yaw angles
        """
        # Vectors
        v1 = p_joint - p_parent  # Parent to joint
        v2 = p_child - p_joint   # Joint to child
        
        # Normalize
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # Pitch (up-down rotation in sagittal plane)
        pitch = np.arctan2(-v2[1], np.sqrt(v2[0]**2 + v2[2]**2))
        
        # Roll (side-to-side rotation)
        roll = np.arctan2(v2[0], -v2[2])
        
        # Knee angle (flexion)
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        knee_angle = np.pi - np.arccos(cos_angle)
        
        return pitch, roll, knee_angle
    
    def compute_joint_angles(self, landmarks):
        """
        Compute G1 joint angles from MediaPipe landmarks
        FIXED version with proper coordinate transformations
        """
        angles = {}
        
        # Get key points
        left_hip = landmarks[self.mp_idx['left_hip'], :3]
        right_hip = landmarks[self.mp_idx['right_hip'], :3]
        left_knee = landmarks[self.mp_idx['left_knee'], :3]
        right_knee = landmarks[self.mp_idx['right_knee'], :3]
        left_ankle = landmarks[self.mp_idx['left_ankle'], :3]
        right_ankle = landmarks[self.mp_idx['right_ankle'], :3]
        left_shoulder = landmarks[self.mp_idx['left_shoulder'], :3]
        right_shoulder = landmarks[self.mp_idx['right_shoulder'], :3]
        left_elbow = landmarks[self.mp_idx['left_elbow'], :3]
        right_elbow = landmarks[self.mp_idx['right_elbow'], :3]
        left_wrist = landmarks[self.mp_idx['left_wrist'], :3]
        right_wrist = landmarks[self.mp_idx['right_wrist'], :3]
        
        # Hip center
        hip_center = (left_hip + right_hip) / 2
        
        # === LEFT LEG ===
        pitch, roll, knee = self.get_limb_angles_3d(hip_center, left_knee, left_ankle)
        
        # Hip angles - SCALE DOWN for stability
        angles['left_hip_pitch_joint'] = pitch * 0.8
        angles['left_hip_roll_joint'] = roll * 0.5
        angles['left_hip_yaw_joint'] = 0.0
        
        # Knee - ensure positive flexion
        angles['left_knee_joint'] = max(0.0, knee * 0.9)
        
        # Ankle
        ankle_vec = left_ankle - left_knee
        angles['left_ankle_pitch_joint'] = -np.arctan2(ankle_vec[1], -ankle_vec[2]) * 0.3
        angles['left_ankle_roll_joint'] = 0.0
        
        # === RIGHT LEG ===
        pitch, roll, knee = self.get_limb_angles_3d(hip_center, right_knee, right_ankle)
        
        angles['right_hip_pitch_joint'] = pitch * 0.8
        angles['right_hip_roll_joint'] = -roll * 0.5  # Mirror
        angles['right_hip_yaw_joint'] = 0.0
        angles['right_knee_joint'] = max(0.0, knee * 0.9)
        
        ankle_vec = right_ankle - right_knee
        angles['right_ankle_pitch_joint'] = -np.arctan2(ankle_vec[1], -ankle_vec[2]) * 0.3
        angles['right_ankle_roll_joint'] = 0.0
        
        # === TORSO ===
        shoulder_center = (left_shoulder + right_shoulder) / 2
        torso_vec = shoulder_center - hip_center
        
        # Simplified torso angles
        angles['waist_pitch_joint'] = np.arctan2(torso_vec[1], np.sqrt(torso_vec[0]**2 + torso_vec[2]**2)) * 0.2
        angles['waist_roll_joint'] = np.arctan2(torso_vec[0], -torso_vec[2]) * 0.3
        angles['waist_yaw_joint'] = 0.0
        
        # === LEFT ARM ===
        shoulder_vec = left_elbow - left_shoulder
        elbow_vec = left_wrist - left_elbow
        
        # Shoulder pitch (forward/back)
        angles['left_shoulder_pitch_joint'] = np.arctan2(-shoulder_vec[1], shoulder_vec[2]) - np.pi/2
        
        # Shoulder roll (up/down)
        angles['left_shoulder_roll_joint'] = np.arctan2(shoulder_vec[0], -shoulder_vec[2])
        
        angles['left_shoulder_yaw_joint'] = 0.0
        
        # Elbow flexion
        cos_angle = np.clip(np.dot(
            shoulder_vec / (np.linalg.norm(shoulder_vec) + 1e-8),
            elbow_vec / (np.linalg.norm(elbow_vec) + 1e-8)
        ), -1.0, 1.0)
        angles['left_elbow_joint'] = max(0.0, np.pi - np.arccos(cos_angle))
        
        # Wrists - keep neutral
        angles['left_wrist_roll_joint'] = 0.0
        angles['left_wrist_pitch_joint'] = 0.0
        angles['left_wrist_yaw_joint'] = 0.0
        
        # === RIGHT ARM (mirror) ===
        shoulder_vec = right_elbow - right_shoulder
        elbow_vec = right_wrist - right_elbow
        
        angles['right_shoulder_pitch_joint'] = np.arctan2(-shoulder_vec[1], shoulder_vec[2]) - np.pi/2
        angles['right_shoulder_roll_joint'] = -np.arctan2(shoulder_vec[0], -shoulder_vec[2])
        angles['right_shoulder_yaw_joint'] = 0.0
        
        cos_angle = np.clip(np.dot(
            shoulder_vec / (np.linalg.norm(shoulder_vec) + 1e-8),
            elbow_vec / (np.linalg.norm(elbow_vec) + 1e-8)
        ), -1.0, 1.0)
        angles['right_elbow_joint'] = max(0.0, np.pi - np.arccos(cos_angle))
        
        angles['right_wrist_roll_joint'] = 0.0
        angles['right_wrist_pitch_joint'] = 0.0
        angles['right_wrist_yaw_joint'] = 0.0
        
        # Clamp to joint limits
        for joint_name, angle in angles.items():
            limits = self.joint_limits[joint_name]
            angles[joint_name] = np.clip(angle, limits[0], limits[1])
        
        return angles
    
    def retarget_video(self, mediapipe_pkl_path, output_path='g1_motion.npz'):
        """
        Retarget full video from MediaPipe landmarks to G1 joint angles
        """
        # Load MediaPipe data
        with open(mediapipe_pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        poses = data['poses']
        fps = data['fps']
        
        all_joint_angles = []
        
        print(f"Retargeting {len(poses)} frames...")
        
        for i, pose_data in enumerate(poses):
            if pose_data is None:
                if all_joint_angles:
                    all_joint_angles.append(all_joint_angles[-1])
                else:
                    # Standing pose with bent elbows
                    default = {joint: 0.0 for joint in self.g1_joints}
                    default['left_elbow_joint'] = 0.5
                    default['right_elbow_joint'] = 0.5
                    all_joint_angles.append([default[j] for j in self.g1_joints])
            else:
                landmarks = pose_data['landmarks_3d']
                angles = self.compute_joint_angles(landmarks)
                all_joint_angles.append([angles[j] for j in self.g1_joints])
            
            if (i + 1) % 30 == 0:
                print(f"Processed {i + 1}/{len(poses)} frames")
        
        # Convert to numpy and save
        joint_angles_array = np.array(all_joint_angles)
        
        # Print stats for debugging
        print(f"\nJoint angle statistics:")
        print(f"  Shape: {joint_angles_array.shape}")
        print(f"  Mean: {np.mean(np.abs(joint_angles_array)):.4f}")
        print(f"  Max: {np.max(np.abs(joint_angles_array)):.4f}")
        print(f"  Std: {np.std(joint_angles_array):.4f}")
        
        np.savez(output_path,
                 joint_angles=joint_angles_array,
                 joint_names=self.g1_joints,
                 fps=fps)
        
        print(f"\nRetargeting complete!")
        print(f"Output saved to: {output_path}")
        
        return joint_angles_array


def smooth_motion(joint_angles, window_size=5):
    """Apply smoothing filter"""
    from scipy.ndimage import gaussian_filter1d
    smoothed = np.copy(joint_angles)
    for i in range(joint_angles.shape[1]):
        smoothed[:, i] = gaussian_filter1d(joint_angles[:, i], sigma=window_size/3)
    return smoothed


if __name__ == '__main__':
    retargeter = Retargeter()
    input_path = 'smpl_files/mediapipe_poses.pkl'
    output_path = 'smpl_files/salute_retarget.npz'

    joint_angles = retargeter.retarget_video(input_path, output_path)
    
    # Apply smoothing
    smooth_window = 7
    if smooth_window > 0:
        print(f"\nApplying Gaussian smoothing...")
        smoothed = smooth_motion(joint_angles, smooth_window)
        smoothed_output = output_path.replace('.npz', '_smoothed.npz')
        np.savez(
            smoothed_output,
            joint_angles=smoothed,
            joint_names=retargeter.g1_joints
        )
        print(f"Smoothed motion saved to: {smoothed_output}")
