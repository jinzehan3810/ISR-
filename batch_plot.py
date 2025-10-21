#!/usr/bin/env python3
"""
批量绘图脚本
为多个实验生成学习曲线图：reward_main.png、reward_task.png、success.png
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# 添加项目路径
sys.path.append('/home/gyhai/CurricuLLM')
from gpt.utils import file_to_string

# ==================== 配置区域 ====================
# 在这里修改模型名称
MODEL_NAME = "qwen"  # 可以修改为其他模型名称，如 "gpt4", "claude" 等

# 是否在文件名中包含时间戳
INCLUDE_TIMESTAMP = True

# 生成时间戳字符串
def get_timestamp():
    """获取当前时间戳字符串"""
    if INCLUDE_TIMESTAMP:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    return ""

# 生成文件名前缀
def get_filename_prefix():
    """生成包含模型名称和时间戳的文件名前缀"""
    timestamp = get_timestamp()
    if timestamp:
        return f"{MODEL_NAME}_{timestamp}_"
    return f"{MODEL_NAME}_"
# ================================================

def load_curriculum_training_log(logger_path):
    """加载课程训练日志"""
    training_log = np.load(logger_path + "/evaluations.npz", allow_pickle=True)

    reward_dict = training_log["results_dict"]
    success = training_log["successes"].mean(axis=1)

    averaged_dicts = []

    for row in reward_dict:
        sum_dict = {}
        for col in row:
            for key in col:
                sum_dict[key] = sum_dict.get(key, 0) + col[key]

        avg_dict = {key: value/len(row) for key, value in sum_dict.items()}
        averaged_dicts.append(avg_dict)

    reward_df = pd.DataFrame(averaged_dicts)

    return reward_df, success

def extract_curriculum(logger_path):
    """提取课程信息"""
    curriculum_txt = file_to_string(logger_path + "curriculum.md")
    # Split the string into individual task sections
    task_sections = re.split(r'\n\n(?=Task)', curriculum_txt)

    # Function to extract details from each task section
    def extract_task_details(task_section):
        details = {}
        lines = task_section.split('\n')
        for line in lines:
            if line.startswith('Task'):
                details['Task'] = line.split(' ')[1]
            elif line.startswith('Name:'):
                details['Name'] = line.split(': ')[1]
            elif line.startswith('Description:'):
                details['Description'] = line.split(': ')[1]
            elif line.startswith('Reason:'):
                details['Reason'] = ': '.join(line.split(': ')[1:])
        return details

    # Extract details for all tasks
    curriculum_info = [extract_task_details(section) for section in task_sections]
    curriculum_length = len(curriculum_info)
    
    return curriculum_info, curriculum_length

def extract_best_agent(logger_path, curriculum_info, curriculum_length):
    """提取最佳智能体信息"""
    task_list = []
    best_agent_list = []
    for idx in range(curriculum_length):
        curriculum_name = curriculum_info[idx]['Name']
        task_list.append(curriculum_name)
        try:
            decision = file_to_string(logger_path + curriculum_name + '.md')
            decision = decision.split('\n')[0]
            numbers = re.findall(r'\d+', decision)
        except:
            numbers = [0]
        if numbers:
            best_agent_list.append(int(numbers[0]))
        else:
            print(f"No number found in the decision {idx}")
            best_agent_list.append(0)
            
    return task_list, best_agent_list

def plot_experiment(experiment_name):
    """为单个实验生成所有图表"""
    print(f"正在处理实验: {experiment_name} (使用模型: {MODEL_NAME})")
    
    # 设置日志路径
    logger_path = f"./logs/{MODEL_NAME}_{experiment_name}/{experiment_name}/curriculum_0/"
    
    # 检查路径是否存在
    if not os.path.exists(logger_path):
        print(f"警告: 路径不存在 {logger_path}")
        return
    
    try:
        # 提取课程信息
        curriculum_info, curriculum_length = extract_curriculum(logger_path)
        task_list, best_sample_idx = extract_best_agent(logger_path, curriculum_info, curriculum_length)
        
        # 初始化数据列表
        reward_main = []
        reward_task = []
        success_list = []
        task_length = []
        
        # 加载每个任务的数据
        for idx, task in enumerate(task_list):
            path = logger_path + task + f"/sample_{best_sample_idx[idx]}"
            
            if not os.path.exists(path + "/evaluations.npz"):
                print(f"警告: 评估文件不存在 {path}/evaluations.npz")
                continue
                
            reward_df, success = load_curriculum_training_log(path)
            
            reward_main.append(reward_df["main"])
            reward_task.append(reward_df["task"])
            success_list.append(success)
            task_length.append(len(reward_df["main"]))
        
        if not reward_main:
            print(f"警告: 没有找到有效数据用于实验 {experiment_name}")
            return
        
        # 合并数据
        reward_main = np.concatenate(reward_main, axis=0)
        reward_task = np.concatenate(reward_task, axis=0)
        success = np.concatenate(success_list, axis=0)
        
        # 设置绘图参数
        n_tasks = len(task_length)
        color_list = ['blue', 'orange', 'green', 'purple', 'pink', 'brown', 'olive']
        filename_prefix = get_filename_prefix()
        
        # 1. 绘制主奖励图
        plt.figure(figsize=(10, 6))
        plt.plot(reward_main, label='curriculum')
        for i in range(n_tasks):
            idx_start = sum(task_length[:i])
            idx_end = sum(task_length[:i+1])
            task_string = 'task ' + str(i)
            plt.axvspan(idx_start, idx_end, color=color_list[i], alpha=0.3, label=task_string)
        plt.title(f'{experiment_name} - Main Reward (Model: {MODEL_NAME})')
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(logger_path + f"{filename_prefix}reward_main.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 绘制任务奖励图
        plt.figure(figsize=(10, 6))
        plt.plot(reward_task, label='curriculum')
        for i in range(n_tasks):
            idx_start = sum(task_length[:i])
            idx_end = sum(task_length[:i+1])
            task_string = 'task ' + str(i)
            plt.axvspan(idx_start, idx_end, color=color_list[i], alpha=0.3, label=task_string)
        plt.title(f'{experiment_name} - Task Reward (Model: {MODEL_NAME})')
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(logger_path + f"{filename_prefix}reward_task.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 绘制成功率图
        plt.figure(figsize=(10, 6))
        success_moving_avg = pd.Series(success).rolling(10).mean()
        plt.plot(success_moving_avg, label='curriculum')
        for i in range(n_tasks):
            idx_start = sum(task_length[:i])
            idx_end = sum(task_length[:i+1])
            task_string = 'task ' + str(i)
            plt.axvspan(idx_start, idx_end, color=color_list[i], alpha=0.3, label=task_string)
        plt.title(f'{experiment_name} - Success Rate (Model: {MODEL_NAME})')
        plt.xlabel('Training Steps')
        plt.ylabel('Success Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(logger_path + f"{filename_prefix}success.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 已完成实验 {experiment_name} 的图表生成")
        print(f"  - {logger_path}{filename_prefix}reward_main.png")
        print(f"  - {logger_path}{filename_prefix}reward_task.png")
        print(f"  - {logger_path}{filename_prefix}success.png")
        
    except Exception as e:
        print(f"错误: 处理实验 {experiment_name} 时出现问题: {str(e)}")

def plot_combined_experiments():
    """生成3x3的组合图表"""
    print(f"开始生成3x3组合学习曲线图... (使用模型: {MODEL_NAME})")
    print("=" * 50)
    
    # 切换到项目目录
    os.chdir('/home/gyhai/CurricuLLM')
    
    # 定义实验列表
    experiments = ['FetchPush', 'FetchSlide', 'AntMaze_UMaze']
    
    # 创建3x3的子图
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'CurricuLLM Learning Curves Comparison (Model: {MODEL_NAME})', fontsize=16, fontweight='bold')
    
    # 设置颜色列表
    color_list = ['blue', 'orange', 'green', 'purple', 'pink', 'brown', 'olive']
    
    for exp_idx, experiment_name in enumerate(experiments):
        print(f"正在处理实验: {experiment_name}")
        
        # 设置日志路径
        logger_path = f"./logs/{MODEL_NAME}_{experiment_name}/{experiment_name}/curriculum_0/"
        
        # 检查路径是否存在
        if not os.path.exists(logger_path):
            print(f"警告: 路径不存在 {logger_path}")
            continue
        
        try:
            # 提取课程信息
            curriculum_info, curriculum_length = extract_curriculum(logger_path)
            task_list, best_sample_idx = extract_best_agent(logger_path, curriculum_info, curriculum_length)
            
            # 初始化数据列表
            reward_main = []
            reward_task = []
            success_list = []
            task_length = []
            
            # 加载每个任务的数据
            for idx, task in enumerate(task_list):
                path = logger_path + task + f"/sample_{best_sample_idx[idx]}"
                
                if not os.path.exists(path + "/evaluations.npz"):
                    print(f"警告: 评估文件不存在 {path}/evaluations.npz")
                    continue
                    
                reward_df, success = load_curriculum_training_log(path)
                
                reward_main.append(reward_df["main"])
                reward_task.append(reward_df["task"])
                success_list.append(success)
                task_length.append(len(reward_df["main"]))
            
            if not reward_main:
                print(f"警告: 没有找到有效数据用于实验 {experiment_name}")
                continue
            
            # 合并数据
            reward_main = np.concatenate(reward_main, axis=0)
            reward_task = np.concatenate(reward_task, axis=0)
            success = np.concatenate(success_list, axis=0)
            
            # 设置绘图参数
            n_tasks = len(task_length)
            
            # 1. 绘制主奖励图 (第一列)
            ax1 = axes[exp_idx, 0]
            ax1.plot(reward_main, label='curriculum', linewidth=1.5)
            for i in range(n_tasks):
                idx_start = sum(task_length[:i])
                idx_end = sum(task_length[:i+1])
                task_string = f'task {i}'
                ax1.axvspan(idx_start, idx_end, color=color_list[i], alpha=0.3, label=task_string if i < 3 else "")
            ax1.set_title(f'{experiment_name} - Main Reward', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Reward')
            ax1.grid(True, alpha=0.3)
            if exp_idx == 0:  # 只在第一行显示图例
                ax1.legend(fontsize=8)
            
            # 2. 绘制任务奖励图 (第二列)
            ax2 = axes[exp_idx, 1]
            ax2.plot(reward_task, label='curriculum', linewidth=1.5)
            for i in range(n_tasks):
                idx_start = sum(task_length[:i])
                idx_end = sum(task_length[:i+1])
                task_string = f'task {i}'
                ax2.axvspan(idx_start, idx_end, color=color_list[i], alpha=0.3, label=task_string if i < 3 else "")
            ax2.set_title(f'{experiment_name} - Task Reward', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Reward')
            ax2.grid(True, alpha=0.3)
            if exp_idx == 0:  # 只在第一行显示图例
                ax2.legend(fontsize=8)
            
            # 3. 绘制成功率图 (第三列)
            ax3 = axes[exp_idx, 2]
            success_moving_avg = pd.Series(success).rolling(10).mean()
            ax3.plot(success_moving_avg, label='curriculum', linewidth=1.5)
            for i in range(n_tasks):
                idx_start = sum(task_length[:i])
                idx_end = sum(task_length[:i+1])
                task_string = f'task {i}'
                ax3.axvspan(idx_start, idx_end, color=color_list[i], alpha=0.3, label=task_string if i < 3 else "")
            ax3.set_title(f'{experiment_name} - Success Rate', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('Success Rate')
            ax3.grid(True, alpha=0.3)
            if exp_idx == 0:  # 只在第一行显示图例
                ax3.legend(fontsize=8)
            
            print(f"✓ 已完成实验 {experiment_name} 的子图绘制")
            
        except Exception as e:
            print(f"错误: 处理实验 {experiment_name} 时出现问题: {str(e)}")
    
    # 调整子图间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # 为主标题留出空间
    
    # 保存组合图
    filename_prefix = get_filename_prefix()
    output_path = f'./{filename_prefix}combined_learning_curves_3x3.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("=" * 50)
    print("3x3组合图表生成完成！")
    print(f"保存路径: {output_path}")
    print(f"使用模型: {MODEL_NAME}")
    print("图表布局:")
    print("  列1: Main Reward  |  列2: Task Reward  |  列3: Success Rate")
    print("  行1: FetchPush")
    print("  行2: FetchSlide") 
    print("  行3: AntMaze_UMaze")

def main():
    """主函数"""
    # 可以选择生成单独的图表还是组合图表
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--separate':
        # 生成单独的图表
        print("开始批量生成单独的学习曲线图...")
        print("=" * 50)
        
        # 切换到项目目录
        os.chdir('/home/gyhai/CurricuLLM')
        
        # 定义实验列表
        experiments = ['FetchPush', 'FetchSlide', 'AntMaze_UMaze']
        
        # 为每个实验生成图表
        for experiment in experiments:
            plot_experiment(experiment)
            print()
        
        print("=" * 50)
        print("批量绘图完成！")
        print(f"总共处理了 {len(experiments)} 个实验")
        print("每个实验生成了 3 张图表（reward_main.png, reward_task.png, success.png）")
    else:
        # 生成组合图表（默认）
        plot_combined_experiments()

if __name__ == "__main__":
    main()
