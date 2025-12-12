import gymnasium as gym 
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from env import Env  # Import your custom environment

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# Custom reward wrapper to integrate discriminator rewards
class GAILRewardWrapper(gym.Wrapper):
    def __init__(self, env, discriminator):
        super().__init__(env)
        self.discriminator = discriminator
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Compute discriminator reward
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            disc_output = self.discriminator(state_tensor, action_tensor)
            gail_reward = -torch.log(1 - disc_output + 1e-8).item()
        
        # Combine with pose error reward
        pose_error = info['pose_error']
        total_reward = gail_reward - 0.1 * pose_error
        
        return obs, total_reward, terminated, truncated, info

# Extract expert demonstrations from your trajectory
def extract_expert_demonstrations(demo_trajectory, xml_path, n_episodes=1):
    """
    Extract state-action pairs from expert trajectory
    Assumes demo_trajectory is your SMPL+IK joint positions
    """
    states = []
    actions = []
    
    env = Env(xml_path, demo_trajectory, render_mode='rgb_array')
    
    # For each episode, replay the expert trajectory
    for episode in range(n_episodes):
        obs, _ = env.reset()
        
        for timestep in range(len(demo_trajectory) - 1):
            # Current and next pose from demo
            current_pose = demo_trajectory[timestep]
            next_pose = demo_trajectory[timestep + 1]
            
            # Compute action as difference (velocity control)
            action = (next_pose - current_pose) * 30  # Scale by fps
            action = np.clip(action, -3.0, 3.0)
            
            states.append(obs)
            actions.append(action)
            
            obs, _, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                break
    
    env.close()
    return np.array(states), np.array(actions)

# GAIL Training
class GAIL:
    def __init__(self, xml_path, demo_trajectory, expert_states, expert_actions):
        self.xml_path = xml_path
        self.demo_trajectory = demo_trajectory
        self.expert_states = torch.FloatTensor(expert_states)
        self.expert_actions = torch.FloatTensor(expert_actions)
        
        # Get dimensions
        state_dim = 88  # Your observation space
        action_dim = 29  # Your action space
        
        # Initialize discriminator
        self.discriminator = Discriminator(state_dim, action_dim, hidden_dim=512)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=3e-4)
        
        # Create environment with GAIL reward
        def make_env():
            env = Env(xml_path, demo_trajectory, render_mode='rgb_array')
            env = GAILRewardWrapper(env, self.discriminator)
            return env
        
        self.env = DummyVecEnv([make_env])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        
        # Initialize policy
        self.policy = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="./gail_tensorboard/"
        )
        
        # Storage for policy rollouts
        self.policy_states = []
        self.policy_actions = []
        
    def train_discriminator(self, batch_size=256, n_epochs=5):
        """Train discriminator to distinguish expert from policy"""
        if len(self.policy_states) < batch_size:
            return 0.0
        
        policy_states = torch.FloatTensor(np.array(self.policy_states[-5000:]))
        policy_actions = torch.FloatTensor(np.array(self.policy_actions[-5000:]))
        
        total_loss = 0.0
        n_batches = 0
        
        for epoch in range(n_epochs):
            # Sample batches
            n_samples = min(len(self.expert_states), len(policy_states))
            expert_idx = np.random.choice(len(self.expert_states), batch_size, replace=True)
            policy_idx = np.random.choice(len(policy_states), batch_size, replace=True)
            
            expert_batch_states = self.expert_states[expert_idx]
            expert_batch_actions = self.expert_actions[expert_idx]
            policy_batch_states = policy_states[policy_idx]
            policy_batch_actions = policy_actions[policy_idx]
            
            # Train discriminator
            self.disc_optimizer.zero_grad()
            
            expert_pred = self.discriminator(expert_batch_states, expert_batch_actions)
            policy_pred = self.discriminator(policy_batch_states, policy_batch_actions)
            
            # Binary cross-entropy loss
            expert_loss = -torch.log(expert_pred + 1e-8).mean()
            policy_loss = -torch.log(1 - policy_pred + 1e-8).mean()
            disc_loss = expert_loss + policy_loss
            
            disc_loss.backward()
            self.disc_optimizer.step()
            
            total_loss += disc_loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def collect_rollouts(self, n_steps):
        """Collect rollouts from current policy"""
        obs = self.env.reset()
        
        for _ in range(n_steps):
            action, _ = self.policy.predict(obs, deterministic=False)
            self.policy_states.append(obs[0])
            self.policy_actions.append(action[0])
            obs, _, _, _ = self.env.step(action)
    
    def train(self, total_timesteps=500000, disc_update_freq=2048):
        """Main GAIL training loop"""
        n_updates = total_timesteps // disc_update_freq
        
        for update in range(n_updates):
            print(f"\n=== Update {update + 1}/{n_updates} ===")
            
            # Collect rollouts
            self.collect_rollouts(disc_update_freq)
            
            # Train discriminator
            if len(self.policy_states) >= 1000:
                disc_loss = self.train_discriminator(batch_size=256, n_epochs=5)
                print(f"Discriminator Loss: {disc_loss:.4f}")
            
            # Train policy with GAIL rewards
            self.policy.learn(total_timesteps=disc_update_freq, reset_num_timesteps=False)
            
            # Log statistics
            if len(self.policy_states) > 0:
                recent_states = self.policy_states[-disc_update_freq:]
                print(f"Collected {len(recent_states)} transitions")
            
            # Save checkpoint
            if (update + 1) % 10 == 0:
                self.policy.save(f"gail_policy_update_{update + 1}")
                print(f"Saved checkpoint at update {update + 1}")
        
        return self.policy

# Usage
if __name__ == "__main__":
    # Your paths
    xml_path = "g1.xml"
    
    # Load your expert trajectory (SMPL + IK results)
    # Shape should be (num_frames, 29) for joint positions
    demo_trajectory = np.load("g1_dance_demo_smoothed.npz")
    demo_trajectory = demo_trajectory['joint_angles']
    print(f"Demo trajectory shape: {demo_trajectory.shape}")
    print(f"Demo length: {len(demo_trajectory)} frames")
    
    # Extract expert demonstrations
    print("Extracting expert state-action pairs...")
    expert_states, expert_actions = extract_expert_demonstrations(
        demo_trajectory, xml_path, n_episodes=3
    )
    
    print(f"Expert states: {expert_states.shape}")
    print(f"Expert actions: {expert_actions.shape}")
    
    # Train GAIL
    print("\nStarting GAIL training...")
    gail = GAIL(xml_path, demo_trajectory, expert_states, expert_actions)
    trained_policy = gail.train(total_timesteps=500000, disc_update_freq=2048)
    trained_policy = PPO.load('gail_policy_update_240.zip')
    # Save final policy
    trained_policy.save("gail_humanoid_final")
    print("\nTraining complete! Policy saved.")
    
    # Evaluate
    print("\nEvaluating trained policy...")
    test_env = Env(xml_path, demo_trajectory, render_mode='human')
    obs, _ = test_env.reset()
    
    while True:
    for _ in range(len(demo_trajectory)):
        action, _ = trained_policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        
        if terminated or truncated:
            break
    
    test_env.close()
