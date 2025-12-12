import numpy as np
from imitation.policies.serialize import load_policy
import torch
import time
from imitation.util.networks import RunningNorm
from imitation.algorithms.adversarial import gail
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.algorithms import bc
from imitation.data import types
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import gymnasium as gym
from env import Env 
import mujoco 
import mujoco.viewer

def create_trajectories(demo_trajectory, xml_path, n_episodes=15):
    env = Env(xml_path, demo_trajectory, render_mode='rgb_array')
    trajectories = []
    succ_episodes = 0

    for episode in range(n_episodes):
        obs_list = []
        acts_list = []
        infos_list = []

        obs, info = env.reset()
        noise_scale = 0.02 + (episode/n_episodes) * 0.05 
        noise = np.random.randn(29) * noise_scale 

        env.data.qpos[7:36] = demo_trajectory[0] + noise
        env.data.qpos[2] = 0.79

        start_offset = np.random.randint(0, min(20, len(demo_trajectory) //20))

        mujoco.mj_forward(env.model, env.data)
        obs = env._get_obs()
        obs_list.append(obs)

        for timestep in range(start_offset, len(demo_trajectory)-1):
            if timestep >= len(demo_trajectory) - 1:
                break 

            current_pose = demo_trajectory[timestep]
            next_pose = demo_trajectory[timestep + 1]
            action = (next_pose - current_pose) * 30

            action_noise_scale = 0.01 * (1 - episode / (n_episodes * 2))

            action += np.random.randn(29) * max(action_noise_scale, 0.01)
            action = np.clip(action, -3.0, 3.0)

            acts_list.append(action)

            obs, _, terminated, truncated, info = env.step(action)
            obs_list.append(obs)
            infos_list.append(info)

            if terminated or truncated:
                print(f"  Episode {episode+1} ended early at step {timestep-start_offset}")
                break
        assert len(obs_list) == len(acts_list)+1
        min_length = len(demo_trajectory) * 0.3
        if len(acts_list) >= min_length:
            trajectory = types.Trajectory(
                obs=np.array(obs_list[:-1], dtype=np.float32),
                acts=np.array(acts_list, dtype=np.float32),
                infos=np.array(infos_list) if infos_list else None,
                terminal=True
            )

            trajectories.append(trajectory)
            succ_episodes += 1
            print(f"  Episode {episode+1}: {len(acts_list)} steps ✓")
        else:
            print(f"  Episode {episode+1}: Too short ({len(acts_list)} steps), skipped ✗")
    env.close()
    
    
    return trajectories

def traj(demo_trajectory, xml_path, n_copies=10):
    print(f"\nCreating expert demonstrations...")
    print(f"Demo trajectory: {demo_trajectory.shape}")
    
    env = Env(xml_path, demo_trajectory, render_mode='rgb_array')
    
    # We'll create multiple copies with slight variations
    all_trajectories = []
    
    for copy_idx in range(n_copies):
        obs_list = []
        acts_list = []
        
        # Reset environment
        obs, info = env.reset()
        
        # Set to starting pose with small noise
        if copy_idx == 0:
            # First copy: exact demo
            noise = np.zeros(29)
        else:
            # Other copies: small variations
            noise = np.random.randn(29) * 0.02
        
        env.data.qpos[7:36] = demo_trajectory[0] + noise
        env.data.qpos[2] = 0.79
        mujoco.mj_forward(env.model, env.data)
        obs = env._get_obs()
        
        obs_list.append(obs)  # Initial observation
        
        # Replay the demo
        for t in range(len(demo_trajectory) - 1):
            current_pose = demo_trajectory[t]
            next_pose = demo_trajectory[t + 1]
            
            # Compute action from demo
            action = (next_pose - current_pose) * 30
            
            # Add tiny noise to non-first copies
            if copy_idx > 0:
                action += np.random.randn(29) * 0.01
            
            action = np.clip(action, -3.0, 3.0)
            acts_list.append(action)
            
            # Take step
            obs, _, terminated, truncated, info = env.step(action)
            obs_list.append(obs)  # Observation after action
            
            if terminated or truncated:
                print(f"  Copy {copy_idx+1}: Terminated at step {t+1}")
                break
        
        # Only keep if we got a reasonable number of steps
        if len(acts_list) >= 50:  # At least 50 steps
            trajectory = types.Trajectory(
                obs=np.array(obs_list, dtype=np.float32),
                acts=np.array(acts_list, dtype=np.float32),
                infos=None,  # Skip infos to avoid issues
                terminal=True
            )
            all_trajectories.append(trajectory)
            print(f"  ✓ Copy {copy_idx+1}: {len(acts_list)} steps")
        else:
            print(f"  ✗ Copy {copy_idx+1}: Too short ({len(acts_list)} steps)")
    
    env.close()
    
    if len(all_trajectories) == 0:
        raise ValueError("Failed to create any valid trajectories!")
    
    print(f"\n✓ Created {len(all_trajectories)} expert trajectories")
    total_steps = sum(len(t.acts) for t in all_trajectories)
    print(f"  Total steps: {total_steps}")
    print(f"  Avg steps per trajectory: {total_steps / len(all_trajectories):.1f}")
    
    return all_trajectories

def make_env(xml_path, demo_trajectory):
    def _init():
        env = Env(xml_path, demo_trajectory, render_mode='rgb_array')
        env = RolloutInfoWrapper(env)
        return env 
    return _init

def train_gail(xml_path, demo_trajectory, expert_trajectories, total_timesteps=300000):
    print("===============TRAINING GAIL===================")
    venv = DummyVecEnv([make_env(xml_path, demo_trajectory)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(xml_path, demo_trajectory)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,training=False, clip_obs=10.0)

    learner = PPO(
        policy='MlpPolicy',
        env=venv,
        batch_size=64,
        n_steps=512,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log='./gail_logs/',
        device='cpu',
        policy_kwargs=dict(
            net_arch=[dict(pi=[256,256], vf=[256,256])]
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq = 5000,
        save_path='./checkpoints/',
        name_prefix='gail_',
        save_vecnormalize=True
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_model/',
        log_path='/gail_eval/',
        eval_freq = 2500,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
        hid_sizes=[128, 128]  # Hidden layer sizes for discriminator
    )
    print('\nInitializing GAIL trainer...')
    trainer = gail.GAIL(
        demonstrations=expert_trajectories,
        demo_batch_size=32,
        gen_replay_buffer_capacity=20000,
        n_disc_updates_per_round=2,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        disc_opt_cls=torch.optim.Adam,
        disc_opt_kwargs={
            'lr':1e-4,
        }
    )

    print('starting training...')
    trainer.train(
        total_timesteps=total_timesteps,
    )

    return trainer, learner, venv 

def train_bc(xml_path, demo_trajectory, expert_trajectories):
    print("===============TRAINING BC===================")
    venv = DummyVecEnv([make_env(xml_path, demo_trajectory)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=expert_trajectories,
        rng=np.random.default_rng(42),
        batch_size=128,
        ent_weight=1e-3,
        l2_weight=1e-5
    )

    print('starting BC....')
    bc_trainer.train(
        n_epochs=200,
        log_interval=10,
        progress_bar=True
    )

    print('\bTRAINING COMPLETE...')
    return bc_trainer, venv

def evaluate_policy(learner, xml_path, demo_trajectory, n_episodes=10):
    print('==============EVALUATIONS==================')
    eval_env = Env(xml_path, demo_trajectory, render_mode='rgb_array')
    all_lengths = []
    completion_rates= []

    for ep in range(n_episodes):
        obs, _ = eval_env.reset()
        ep_reward = 0
        ep_length = 0
        for step in range(len(demo_trajectory)):
            action, _ = learner.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_length += 1

            if terminated or truncated:
                break 

        all_lengths.append(ep_length)
        completion_rates.append(ep_length / len(demo_trajectory))

        print(f"Episode {ep+1:2d}: "
              f"Length={ep_length:3d}/{len(demo_trajectory)} "
              f"({completion_rates[-1]*100:.1f}%), ")
    
    eval_env.close()
        
    print(f"  Avg Episode Length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
    print(f"  Avg Completion: {np.mean(completion_rates)*100:.1f}% ± {np.std(completion_rates)*100:.1f}%")

def visualize_policy(policy, xml_path, demo_trajectory):
    print('=======VISUALIZATION============')
    vis_env = Env(xml_path, demo_trajectory,render_mode='human')
    obs, _ = vis_env.reset()

    for step in range(len(demo_trajectory)):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = vis_env.step(action)
        vis_env.render()
        
        time.sleep(0.03)
        if terminated or truncated:
            print('Episode Eneded')
            break 
    vis_env.close()


def ss(xml_path):
    demo_trajectory= np.load('stable_salute.npy')
    env = Env(xml_path, demo_trajectory, render_mode='human')

    obs, _ = env.reset()

    for t in range(min(200, len(demo_trajectory)-1)):
        action = (demo_trajectory[t+1] - demo_trajectory[t]) * 30
        action = np.clip(action, -3.0, 3.0)
        print(action)

        obs, _, terminated, truncated, info = env.step(action)
        env.render()

        time.sleep(0.03)


if __name__ == '__main__':
    xml_path = 'g1.xml'
    trajectory_path = 'stable_salute.npy'
    demo_trajectory = np.load(trajectory_path)
#    expert_trajectories = traj(
#        demo_trajectory, xml_path, n_copies=10
#    )
#
#    # GAIL 
#    trainer, learner, venv = train_gail(
#       xml_path, demo_trajectory, expert_trajectories, total_timesteps=300000
#    )
#
#    # Behaviour Cloning 
#    bc_trainer, bc_venv = train_bc(
#        xml_path, demo_trajectory, expert_trajectories
#    )
#    bc_learner = bc_trainer.policy
#    print('SAVING MODELS!!!!!!!!!!!!!!!!!!!')
#    learner.save('./trained_models/salute_learner_gail')
#    torch.save(bc_trainer.policy, './trained_models/salute_bc_policy.pth')
#    venv.save("./trained_models/salute_venv_normalize.pkl")
    
    venv = DummyVecEnv([make_env(xml_path, demo_trajectory)])
    venv = VecNormalize.load('trained_models/salute_venv_normalize.pkl', venv)
    venv.training = False  
    venv.norm_reward = False
    learner= PPO.load('trained_models/salute_learner_gail', env=venv)
    bc_learner = torch.load('./trained_models/salute_bc_policy.pth', weights_only=False)
    bc_learner.eval()
    # EVALUATE GAIL 
    evaluate_policy(learner, xml_path, demo_trajectory, n_episodes=10)

    # EVALUATE BC 
    evaluate_policy(bc_learner, xml_path, demo_trajectory, n_episodes=10)
    
    visualize_policy(bc_learner, xml_path, demo_trajectory)
#    ss(xml_path)
    print("DONE MGF")
