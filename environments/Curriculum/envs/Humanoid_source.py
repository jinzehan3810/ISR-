import numpy as np
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from gymnasium.utils.ezpickle import EzPickle
from gymnasium import utils

class HumanoidCurriculumEnv(HumanoidEnv, EzPickle):
    def __init__(self, **kwargs):
        # Humanoid-v4 默认参数
        super().__init__(**kwargs)
        EzPickle.__init__(self, **kwargs)

    # --- API for LLM ---
    # LLM will use these functions to construct reward functions

    def get_torso_pos(self):
        """Return torso position (x, y, z)"""
        return self.data.qpos[:3]

    def get_torso_vel(self):
        """Return torso linear velocity (vx, vy, vz)"""
        return self.data.qvel[:3]

    def get_joint_angles(self):
        """Return all joint angles (excluding root)"""
        return self.data.qpos[7:]

    def get_joint_velocities(self):
        """Return all joint velocities (excluding root)"""
        return self.data.qvel[6:]

    def obs(self):
        """
        Returns a dictionary of observations that LLM can use.
        LLM generated code will typically call this first.
        """
        pos = self.get_torso_pos()
        vel = self.get_torso_vel()
        
        return {
            "torso_z": pos[2],          # Height (z)
            "torso_x": pos[0],          # Forward position (x)
            "torso_y": pos[1],          # Sideways position (y)
            "forward_vel": vel[0],      # Forward velocity
            "sideways_vel": vel[1],     # Sideways velocity
            "upward_vel": vel[2],       # Upward velocity
            "joint_angles": self.get_joint_angles(),
            "joint_vels": self.get_joint_velocities()
        }

    # The compute_reward_curriculum method will be appended here by LLM
