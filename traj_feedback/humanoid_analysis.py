import numpy as np

def analyze_trajectory_humanoid(obs_trajectory, goal_trajectory):
    """
    Analyze the trajectory of the Humanoid robot.
    Returns a dictionary of statistics.
    """
    obs_trajectory = np.array(obs_trajectory)
    
    # Assuming obs structure from Humanoid_source.py:
    # [0]: torso_z
    # [1]: torso_x
    # [2]: torso_y
    # [3]: forward_vel
    # [4]: sideways_vel
    # [5]: upward_vel
    
    # Extract relevant metrics
    torso_z = obs_trajectory[:, 0]
    forward_vel = obs_trajectory[:, 3]
    
    # Calculate statistics
    mean_height = np.mean(torso_z)
    mean_velocity = np.mean(forward_vel)
    max_velocity = np.max(forward_vel)
    
    # Check if robot fell (height < 1.0 is a common threshold for falling)
    fell_count = np.sum(torso_z < 1.0)
    fall_rate = fell_count / len(torso_z)
    
    return {
        "mean_height": f"{mean_height:.2f}",
        "mean_forward_velocity": f"{mean_velocity:.2f}",
        "max_forward_velocity": f"{max_velocity:.2f}",
        "fall_rate": f"{fall_rate:.2%}"
    }
