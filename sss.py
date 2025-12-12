import numpy as np

def stabilize_salute(trajectory):
    """
    Stabilize salute motion - keep legs planted, only upper body moves
    """
    n_frames = len(trajectory)
    stabilized = np.zeros((n_frames, 29))
    
    print(f"Stabilizing salute: {n_frames} frames")
    
    for i in range(n_frames):
        # === STABLE STANDING BASE (never changes) ===
        
        # Left leg - standing with slight bend
        stabilized[i, 0] = -0.1    # left_hip_pitch_joint (slight back)
        stabilized[i, 1] = 0.05    # left_hip_roll_joint (slight out)
        stabilized[i, 2] = 0.0     # left_hip_yaw_joint
        stabilized[i, 3] = 0.2     # left_knee_joint (slight bend for stability)
        stabilized[i, 4] = 0.1     # left_ankle_pitch_joint (compensation)
        stabilized[i, 5] = 0.0     # left_ankle_roll_joint
        
        # Right leg - mirror
        stabilized[i, 6] = -0.1    # right_hip_pitch_joint
        stabilized[i, 7] = -0.05   # right_hip_roll_joint
        stabilized[i, 8] = 0.0     # right_hip_yaw_joint
        stabilized[i, 9] = 0.2     # right_knee_joint
        stabilized[i, 10] = 0.1    # right_ankle_pitch_joint
        stabilized[i, 11] = 0.0    # right_ankle_roll_joint
        
        # Waist - mostly stable, allow slight movement for naturalness
        stabilized[i, 12] = 0.0    # waist_yaw_joint
        stabilized[i, 13] = 0.0    # waist_roll_joint
        stabilized[i, 14] = 0.05   # waist_pitch_joint (slight forward lean)
        
        # === ARMS - COPY FROM SALUTE MOTION ===
        # This is where the salute happens!
        stabilized[i, 15:29] = trajectory[i, 15:29]
    
    # Print stats
    print(f"✓ Leg movement (should be 0): {stabilized[:, 0].std():.6f}")
    print(f"✓ Arm movement (should be >0): {stabilized[:, 15].std():.3f}")
    
    return stabilized


def create_smooth_salute(trajectory, smoothing_window=7):
    """
    Optional: Smooth the arm motion for more natural movement
    """
    from scipy.ndimage import gaussian_filter1d
    
    smoothed = trajectory.copy()
    
    # Only smooth arms (indices 15-28), leave legs as-is
    for joint_idx in range(15, 29):
        smoothed[:, joint_idx] = gaussian_filter1d(
            trajectory[:, joint_idx], 
            sigma=smoothing_window/3
        )
    
    return smoothed


def verify_stability(trajectory, demo_name="salute"):
    """
    Check if the trajectory is actually stable
    """
    print(f"\n=== Verifying {demo_name} stability ===")
    
    # Check leg movement (should be minimal)
    leg_joints = trajectory[:, :12]  # First 12 joints are legs
    leg_std = np.std(leg_joints, axis=0)
    
    print(f"Leg joint std devs:")
    for i, std in enumerate(leg_std):
        status = "✓" if std < 0.01 else "⚠️"
        print(f"  Joint {i}: {std:.6f} {status}")
    
    # Check arm movement (should exist)
    arm_joints = trajectory[:, 15:29]  # Arms
    arm_std = np.std(arm_joints, axis=0)
    
    print(f"\nArm joint std devs:")
    has_movement = False
    for i, std in enumerate(arm_std):
        if std > 0.05:
            has_movement = True
            print(f"  Arm joint {i+15}: {std:.3f} ✓ (moving)")
    
    if not has_movement:
        print("⚠️  WARNING: No significant arm movement detected!")
    
    # Check if any joints exceed limits
    print(f"\nValue ranges:")
    print(f"  Min: {trajectory.min():.3f}")
    print(f"  Max: {trajectory.max():.3f}")
    
    return leg_std.max() < 0.01 and has_movement


# === MAIN USAGE ===
if __name__ == '__main__':
    # Load your salute trajectory
    salute_path = 'smpl_files/salute_retarget_smoothed.npz'
    salute = np.load(salute_path)['joint_angles']
    
    print(f"Original salute shape: {salute.shape}")
    print(f"Duration: ~{len(salute)/30:.1f} seconds at 30fps")
    
    # Stabilize it
    stable_salute = stabilize_salute(salute)
    
    # Optional: Smooth for more natural motion
    stable_salute = create_smooth_salute(stable_salute, smoothing_window=5)
    
    # Verify it's stable
    is_stable = verify_stability(stable_salute, "stable_salute")
    
    if is_stable:
        # Save
        np.save('stable_salute.npy', stable_salute)
        print(f"\n✓ Saved stable_salute.npy ({stable_salute.shape})")
        print(f"  Ready to use for training!")
    else:
        print("\n⚠️  Trajectory may not be stable enough")
        print("  Consider adjusting knee bend or checking retargeting")
