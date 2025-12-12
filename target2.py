import numpy as np

# Create proper standing/waving trajectory for G1 robot
n_frames = 3000
demo_trajectory = np.zeros((n_frames, 29))

# Use the standing pose from XML keyframe as base
standing_pose = np.array([
    0, 0, 0, 0, 0, 0,        # Left leg (neutral)
    0, 0, 0, 0, 0, 0,        # Right leg (neutral)
    0, 0, 0,                  # Waist (neutral)
    0.2, 0.2, 0, 1.28, 0, 0, 0,   # Left arm (standing with bent elbow)
    0.2, -0.2, 0, 1.28, 0, 0, 0   # Right arm (standing with bent elbow)
])

for i in range(n_frames):
    t = i / n_frames
    
    # Start with standing pose
    demo_trajectory[i] = standing_pose.copy()
    
    # Add slight knee bend for stability
    demo_trajectory[i, 3] = 0.1   # left_knee_joint (slight bend)
    demo_trajectory[i, 9] = 0.1   # right_knee_joint (slight bend)
    
    # Add wave motion to arms
    wave = np.sin(2 * np.pi * t)
    
    # Left arm wave
    demo_trajectory[i, 15] = 0.2 + 0.5 * wave  # left_shoulder_pitch
    demo_trajectory[i, 16] = 0.2 + 0.3 * abs(wave)  # left_shoulder_roll (lift up)
    demo_trajectory[i, 18] = 1.28 + 0.3 * np.sin(4 * np.pi * t)  # left_elbow
    
    # Right arm wave (opposite phase)
    demo_trajectory[i, 22] = 0.2 - 0.5 * wave  # right_shoulder_pitch
    demo_trajectory[i, 23] = -0.2 - 0.3 * abs(wave)  # right_shoulder_roll
    demo_trajectory[i, 25] = 1.28 + 0.3 * np.sin(4 * np.pi * t + np.pi)  # right_elbow

# Verify joint limits (from XML)
joint_limits = {
    'left_hip_pitch': (-2.5307, 2.8798),
    'left_hip_roll': (-0.5236, 2.9671),
    'left_hip_yaw': (-2.7576, 2.7576),
    'left_knee': (-0.087267, 2.8798),
    'left_ankle_pitch': (-0.87267, 0.5236),
    'left_ankle_roll': (-0.2618, 0.2618),
    'right_hip_pitch': (-2.5307, 2.8798),
    'right_hip_roll': (-2.9671, 0.5236),
    'right_hip_yaw': (-2.7576, 2.7576),
    'right_knee': (-0.087267, 2.8798),
    'right_ankle_pitch': (-0.87267, 0.5236),
    'right_ankle_roll': (-0.2618, 0.2618),
    'waist_yaw': (-2.618, 2.618),
    'waist_roll': (-0.52, 0.52),
    'waist_pitch': (-0.52, 0.52),
    'left_shoulder_pitch': (-3.0892, 2.6704),
    'left_shoulder_roll': (-1.5882, 2.2515),
    'left_shoulder_yaw': (-2.618, 2.618),
    'left_elbow': (-1.0472, 2.0944),
    'left_wrist_roll': (-1.97222, 1.97222),
    'left_wrist_pitch': (-1.61443, 1.61443),
    'left_wrist_yaw': (-1.61443, 1.61443),
    'right_shoulder_pitch': (-3.0892, 2.6704),
    'right_shoulder_roll': (-2.2515, 1.5882),
    'right_shoulder_yaw': (-2.618, 2.618),
    'right_elbow': (-1.0472, 2.0944),
    'right_wrist_roll': (-1.97222, 1.97222),
    'right_wrist_pitch': (-1.61443, 1.61443),
    'right_wrist_yaw': (-1.61443, 1.61443),
}

limits_array = np.array([
    (-2.5307, 2.8798), (-0.5236, 2.9671), (-2.7576, 2.7576),  # Left leg
    (-0.087267, 2.8798), (-0.87267, 0.5236), (-0.2618, 0.2618),
    (-2.5307, 2.8798), (-2.9671, 0.5236), (-2.7576, 2.7576),  # Right leg
    (-0.087267, 2.8798), (-0.87267, 0.5236), (-0.2618, 0.2618),
    (-2.618, 2.618), (-0.52, 0.52), (-0.52, 0.52),  # Waist
    (-3.0892, 2.6704), (-1.5882, 2.2515), (-2.618, 2.618),  # Left arm
    (-1.0472, 2.0944), (-1.97222, 1.97222), (-1.61443, 1.61443), (-1.61443, 1.61443),
    (-3.0892, 2.6704), (-2.2515, 1.5882), (-2.618, 2.618),  # Right arm
    (-1.0472, 2.0944), (-1.97222, 1.97222), (-1.61443, 1.61443), (-1.61443, 1.61443),
])

# Clip to limits
for i in range(29):
    demo_trajectory[:, i] = np.clip(demo_trajectory[:, i], limits_array[i, 0], limits_array[i, 1])

# Save
np.save('g1_standing_wave.npy', demo_trajectory)

print("Created G1 standing wave motion!")
print(f"Shape: {demo_trajectory.shape}")
print(f"Frame 0 (standing): {demo_trajectory[0]}")
print(f"\nJoint ranges used:")
print(f"  Legs: {np.min(demo_trajectory[:, :12]):.3f} to {np.max(demo_trajectory[:, :12]):.3f}")
print(f"  Arms: {np.min(demo_trajectory[:, 15:]):.3f} to {np.max(demo_trajectory[:, 15:]):.3f}")
print(f"  Waist: {np.min(demo_trajectory[:, 12:15]):.3f} to {np.max(demo_trajectory[:, 12:15]):.3f}")
