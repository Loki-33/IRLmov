import pygame
import mujoco 
import gymnasium as gym 
from gymnasium import spaces 
import numpy as np 
import mujoco.viewer

class Env(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps':30}
    def __init__(self,xml_path, demo_trajectory, render_mode='rgb_array'):
        super().__init__()
        pygame.init()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.renderer = None 

        self.demo_trajectory = demo_trajectory
        self.demo_length = len(demo_trajectory)

        self.action_space = spaces.Box(
            low=-3.0,
            high=3.0,
            shape=(29,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(88+3+4,),
            dtype=np.float32
        )
        self.viewer = None 
        self.max_timesteps = self.demo_length
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        noise = np.random.randn(29) * 0.05

        self.data.qpos[7:36] = self.demo_trajectory[0] + noise
        self.data.qpos[2] = 0.79


        self.timestep = 0

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}

        return obs, info 

    def step(self, action):
        action = np.clip(action, -3.0, 3.0)
        self.data.ctrl[:29] = action 
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self.timestep += 1 


        obs = self._get_obs()

        terminated = self._is_terminated()

        truncated = self.timestep >= self.max_timesteps

        info ={
            'timestep': self.timestep,
            'pose_error': self._pose_error()
        }

        return obs, 0, terminated, truncated, info 

    
    def _get_obs(self):
        qpos = self.data.qpos[7:36].copy()
        qvel = self.data.qvel[6:35].copy()

        demo_idx = min(self.timestep, self.demo_length-1)
        target_pose = self.demo_trajectory[demo_idx].copy()


        phase = np.array([self.timestep/ self.demo_length])

        com = self.data.subtree_com[0].copy()  # Center of mass
        quat = self.data.qpos[3:7].copy()  # Root orientation
        
        obs = np.concatenate([qpos, qvel, target_pose, phase, com, quat])
        return obs.astype(np.float32)

    
    def _pose_error(self):
        demo_idx = min(self.timestep, self.demo_length-1)
        target_pose = self.demo_trajectory[demo_idx]
        current_pose = self.data.qpos[7:36]
        return np.mean((current_pose - target_pose) ** 2)

    def _is_terminated(self):
        height = self.data.qpos[2]
        if height < 0.3:
            return True
        
        # Tipped over
        quat = self.data.qpos[3:7]
        up_vec = self._quat_to_upvec(quat)
        if up_vec[2] < 0.2:
            return True
        if abs(self.data.qvel[2]) > 2.0:  # Fast vertical movement
            return True
        
        return False

    def _quat_to_upvec(self, quat):
        quat = quat/ np.linalg.norm(quat)
        w, x, y, z = quat 
    
        up = np.array([
            2 * (x*z + w*y),
            2 * (y*z - w*x),
            1 - 2 * (x*x + y*y)
        ])
        return up
    
    def render(self):
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            self.viewer.sync()

        elif self.render_mode == 'rgb_array':
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=400, width=640)

            self.renderer.update_scene(self.data)

            return self.renderer.render()
    
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None 
        elif self.viewer is not None:
            self.viewer.close() 
            self.viewer = None 

