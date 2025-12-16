Task: Humanoid Walk
Goal: The robot must walk forward as fast as possible without falling over.
Success Condition: Average forward velocity > 1.0 m/s and torso height > 1.0 m.
Environment:
- Action Space: 17 continuous actions controlling joint torques.
- Observation Space: 376 dimensions (joint positions, velocities, forces, etc.)
- Dynamics: Mujoco physics engine.
- Termination: Episode ends if torso height < 1.0 or > 2.0 (falling).
