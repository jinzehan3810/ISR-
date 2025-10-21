import os
import gymnasium as gym
from stable_baselines3 import SAC
import Curriculum

os.environ['MUJOCO_GL'] = 'egl' 

# ==================== 全局配置变量 ====================
ENV_NAME = "Curriculum/FetchSlide"  # 自定义环境名称
WEIGHTS_PATH = "logs/gpt_FetchSlide/FetchSlide/curriculum_0/[Basic End Effector Control]/sample_1/best_model.zip"  # SAC权重文件路径
VIDEO_PATH = "./videos/"  # 视频保存目录



def main():
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, VIDEO_PATH, 
                                  episode_trigger=lambda x: True)
    
    model = SAC.load(WEIGHTS_PATH)
    

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    
    
    env.close()
    print(f"视频已保存到: {VIDEO_PATH}")

if __name__ == "__main__":
    main()