import cv2
import mediapipe as mp
import numpy as np
import pickle
from pathlib import Path
import os 

class MediaPipeSMPLExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 0, 1, or 2 (2 is most accurate)
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process_video(self, video_path, output_path='output'):
        """Extract pose from video and convert to SMPL-compatible format"""
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {frame_count}, FPS: {fps}")
        
        all_poses = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(image_rgb)
            
            if results.pose_world_landmarks:
                # Extract 3D landmarks (33 keypoints)
                landmarks_3d = []
                for landmark in results.pose_world_landmarks.landmark:
                    landmarks_3d.append([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ])
                
                # Extract 2D landmarks for reference
                landmarks_2d = []
                if results.pose_landmarks:
                    h, w = frame.shape[:2]
                    for landmark in results.pose_landmarks.landmark:
                        landmarks_2d.append([
                            landmark.x * w,
                            landmark.y * h,
                            landmark.visibility
                        ])
                
                frame_data = {
                    'frame': frame_idx,
                    'landmarks_3d': np.array(landmarks_3d),
                    'landmarks_2d': np.array(landmarks_2d) if landmarks_2d else None,
                    'timestamp': frame_idx / fps
                }
                all_poses.append(frame_data)
            else:
                print(f"No pose detected in frame {frame_idx}")
                all_poses.append(None)
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames")
        
        cap.release()
        
        # Save results
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path) / 'mediapipe_poses.pkl'
        
        with open(output_file, 'wb') as f:
            pickle.dump({
                'poses': all_poses,
                'fps': fps,
                'frame_count': frame_count,
                'video_path': str(video_path)
            }, f)
        
        print(f"\nResults saved to: {output_file}")
        print(f"Successfully processed {len([p for p in all_poses if p is not None])}/{frame_count} frames")
        
        return all_poses
    
    def convert_to_smpl_format(self, poses):
        """
        Convert MediaPipe landmarks to SMPL-compatible format
        MediaPipe gives 33 keypoints, SMPL uses 24 joints
        Simplified 
        """
        # MediaPipe to SMPL joint mapping 
        mp_to_smpl = {
            0: [11, 12],  # Pelvis (avg of hips)
            1: 24,        # Left hip
            2: 26,        # Left knee  
            3: 28,        # Left ankle
            4: 23,        # Right hip
            5: 25,        # Right knee
            6: 27,        # Right ankle
            7: [11, 12],  # Spine (avg of hips)
            8: [11, 12],  # Spine1 (avg of hips)
            9: [11, 12],  # Spine2 (avg of hips)
            10: [0],      # Neck (nose approx)
            11: [0],      # Head (nose)
            12: [0],      # Head top (nose)
            13: 11,       # Left shoulder
            14: 13,       # Left elbow
            15: 15,       # Left wrist
            16: 12,       # Right shoulder
            17: 14,       # Right elbow
            18: 16,       # Right wrist
        }
        
        smpl_poses = []
        for pose_data in poses:
            if pose_data is None:
                smpl_poses.append(None)
                continue
                
            landmarks = pose_data['landmarks_3d']
            
            # Create SMPL-like structure (simplified)
            smpl_joints = np.zeros((24, 3))
            
            # Map available joints
            for smpl_idx, mp_indices in mp_to_smpl.items():
                if smpl_idx >= 24:
                    continue
                if isinstance(mp_indices, list):
                    # Average multiple MediaPipe joints
                    smpl_joints[smpl_idx] = np.mean([landmarks[i, :3] for i in mp_indices], axis=0)
                else:
                    smpl_joints[smpl_idx] = landmarks[mp_indices, :3]
            
            smpl_poses.append({
                'joints': smpl_joints,
                'frame': pose_data['frame']
            })
        
        return smpl_poses

    def visualize_results(self, video_path, poses, output_path='output/visualization.mp4'):

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            pose_data = poses[frame_idx]
            if pose_data and pose_data['landmarks_2d'] is not None:
                # Draw landmarks
                landmarks_2d = pose_data['landmarks_2d']
                for i, (x, y, vis) in enumerate(landmarks_2d):
                    if vis > 0.5:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                
                # Draw connections (simplified)
                connections = [
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                    (11, 23), (12, 24), (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
                ]
                for conn in connections:
                    if conn[0] < len(landmarks_2d) and conn[1] < len(landmarks_2d):
                        pt1 = (int(landmarks_2d[conn[0]][0]), int(landmarks_2d[conn[0]][1]))
                        pt2 = (int(landmarks_2d[conn[1]][0]), int(landmarks_2d[conn[1]][1]))
                        if landmarks_2d[conn[0]][2] > 0.5 and landmarks_2d[conn[1]][2] > 0.5:
                            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        print(f"Visualization saved to: {output_path}")


if __name__ == '__main__':

    video_path = 'Video/salute.mp4'
    output_path = 'smpl_files'
    vis_output = 'output'
    extractor = MediaPipeSMPLExtractor()
    poses = extractor.process_video(video_path, output_path)
    smpl_poses = extractor.convert_to_smpl_format(poses)
    smpl_output = os.path.join(output_path, 'salute_sample.pkl')
    vis_output = os.path.join(vis_output, 'vis4.mp4')
    extractor.visualize_results(video_path,poses, vis_output)




