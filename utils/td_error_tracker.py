"""
全局TD误差追踪器
用于记录和可视化整个训练过程（包括多阶段课程学习）的TD误差统计信息
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
import pickle


class TDErrorTracker:
    """
    全局TD误差追踪器
    
    功能：
    1. 记录每个训练步的TD误差统计信息
    2. 支持多阶段/多任务训练
    3. 实时保存数据到磁盘
    4. 生成可视化图表
    """
    
    def __init__(self, save_dir: str, experiment_name: str = "td_tracking"):
        """
        初始化追踪器
        
        参数:
            save_dir: 保存目录
            experiment_name: 实验名称
        """
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 数据存储
        self.data = {
            'timesteps': [],           # 全局时间步
            'td_mean': [],             # TD误差均值
            'td_std': [],              # TD误差标准差
            'td_var': [],              # TD误差方差
            'reward_var': [],          # 奖励方差
            'stage_markers': [],       # 阶段标记 [(timestep, stage_name), ...]
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'total_timesteps': 0,
            }
        }
        
        # 当前阶段信息
        self.current_stage = None
        self.current_timestep = 0
        
        print(f"[TDErrorTracker] 初始化追踪器: {save_dir}/{experiment_name}")
    
    def set_stage(self, stage_name: str):
        """
        设置当前训练阶段
        
        参数:
            stage_name: 阶段名称（例如："curriculum_0_task_1"）
        """
        self.current_stage = stage_name
        self.data['stage_markers'].append((self.current_timestep, stage_name))
        print(f"[TDErrorTracker] 切换到阶段: {stage_name} (timestep: {self.current_timestep})")
    
    def record(self, td_errors: np.ndarray, rewards: Optional[np.ndarray] = None, 
               timestep_increment: int = 1):
        """
        记录一批TD误差
        
        参数:
            td_errors: TD误差数组
            rewards: 奖励数组（可选）
            timestep_increment: 时间步增量
        """
        # 确保是numpy数组
        td_errors = np.array(td_errors).flatten()
        
        # 计算统计信息
        td_mean = float(np.mean(td_errors))
        td_std = float(np.std(td_errors))
        td_var = float(np.var(td_errors))
        
        reward_var = 0.0
        if rewards is not None:
            rewards = np.array(rewards).flatten()
            reward_var = float(np.var(rewards))
        
        # 更新全局时间步
        self.current_timestep += timestep_increment
        
        # 记录数据
        self.data['timesteps'].append(self.current_timestep)
        self.data['td_mean'].append(td_mean)
        self.data['td_std'].append(td_std)
        self.data['td_var'].append(td_var)
        self.data['reward_var'].append(reward_var)
        self.data['metadata']['total_timesteps'] = self.current_timestep
    
    def save(self):
        """保存数据到磁盘"""
        # 保存为pickle（完整数据）
        pickle_path = os.path.join(self.save_dir, f"{self.experiment_name}_data.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.data, f)
        
        # 保存为JSON（可读性）
        json_path = os.path.join(self.save_dir, f"{self.experiment_name}_data.json")
        # 转换为可JSON序列化的格式
        json_data = {
            'timesteps': self.data['timesteps'],
            'td_mean': self.data['td_mean'],
            'td_std': self.data['td_std'],
            'td_var': self.data['td_var'],
            'reward_var': self.data['reward_var'],
            'stage_markers': self.data['stage_markers'],
            'metadata': self.data['metadata'],
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"[TDErrorTracker] 数据已保存: {pickle_path}")
    
    def load(self, filepath: str):
        """从磁盘加载数据"""
        with open(filepath, 'rb') as f:
            self.data = pickle.load(f)
        
        # 恢复当前时间步
        if self.data['timesteps']:
            self.current_timestep = self.data['timesteps'][-1]
        
        print(f"[TDErrorTracker] 数据已加载: {filepath}")
    
    def plot(self, save_path: Optional[str] = None, show: bool = True, 
             figsize: Tuple[int, int] = (15, 10)):
        """
        绘制TD误差统计图表
        
        参数:
            save_path: 保存路径（如果为None，使用默认路径）
            show: 是否显示图表
            figsize: 图表大小
        """
        if not self.data['timesteps']:
            print("[TDErrorTracker] 警告: 没有数据可以绘制")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        timesteps = np.array(self.data['timesteps'])
        
        # 子图1: TD误差均值
        ax1 = axes[0]
        ax1.plot(timesteps, self.data['td_mean'], label='TD Mean', color='blue', alpha=0.7)
        ax1.set_ylabel('TD Error Mean', fontsize=12)
        ax1.set_title('TD Error Statistics Over Training', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 添加阶段分隔线
        for stage_timestep, stage_name in self.data['stage_markers']:
            ax1.axvline(x=stage_timestep, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax1.text(stage_timestep, ax1.get_ylim()[1], stage_name, 
                    rotation=90, verticalalignment='top', fontsize=8)
        
        # 子图2: TD误差标准差
        ax2 = axes[1]
        ax2.plot(timesteps, self.data['td_std'], label='TD Std', color='green', alpha=0.7)
        ax2.set_ylabel('TD Error Std', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 添加阶段分隔线
        for stage_timestep, _ in self.data['stage_markers']:
            ax2.axvline(x=stage_timestep, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # 子图3: TD误差方差 和 奖励方差
        ax3 = axes[2]
        ax3.plot(timesteps, self.data['td_var'], label='TD Variance', color='purple', alpha=0.7)
        
        # 如果有奖励方差数据
        if any(self.data['reward_var']):
            ax3_twin = ax3.twinx()
            ax3_twin.plot(timesteps, self.data['reward_var'], label='Reward Variance', 
                         color='orange', alpha=0.7, linestyle=':')
            ax3_twin.set_ylabel('Reward Variance', fontsize=12, color='orange')
            ax3_twin.legend(loc='upper right')
        
        ax3.set_ylabel('TD Variance', fontsize=12, color='purple')
        ax3.set_xlabel('Training Timesteps', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        
        # 添加阶段分隔线
        for stage_timestep, _ in self.data['stage_markers']:
            ax3.axvline(x=stage_timestep, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"{self.experiment_name}_plot.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[TDErrorTracker] 图表已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison(self, other_trackers: List['TDErrorTracker'], 
                       labels: Optional[List[str]] = None,
                       save_path: Optional[str] = None,
                       show: bool = True,
                       figsize: Tuple[int, int] = (15, 8)):
        """
        比较多个追踪器的数据
        
        参数:
            other_trackers: 其他追踪器列表
            labels: 标签列表
            save_path: 保存路径
            show: 是否显示
            figsize: 图表大小
        """
        all_trackers = [self] + other_trackers
        
        if labels is None:
            labels = [f"Tracker {i}" for i in range(len(all_trackers))]
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_trackers)))
        
        # 子图1: TD均值比较
        ax1 = axes[0]
        for tracker, label, color in zip(all_trackers, labels, colors):
            if tracker.data['timesteps']:
                ax1.plot(tracker.data['timesteps'], tracker.data['td_mean'], 
                        label=label, alpha=0.7, color=color)
        
        ax1.set_ylabel('TD Error Mean', fontsize=12)
        ax1.set_title('TD Error Mean Comparison', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 子图2: TD标准差比较
        ax2 = axes[1]
        for tracker, label, color in zip(all_trackers, labels, colors):
            if tracker.data['timesteps']:
                ax2.plot(tracker.data['timesteps'], tracker.data['td_std'], 
                        label=label, alpha=0.7, color=color)
        
        ax2.set_ylabel('TD Error Std', fontsize=12)
        ax2.set_xlabel('Training Timesteps', fontsize=12)
        ax2.set_title('TD Error Std Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"td_comparison_plot.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[TDErrorTracker] 对比图表已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_statistics(self) -> Dict:
        """获取当前统计信息"""
        if not self.data['timesteps']:
            return {}
        
        return {
            'total_timesteps': self.current_timestep,
            'num_records': len(self.data['timesteps']),
            'current_stage': self.current_stage,
            'latest_td_mean': self.data['td_mean'][-1] if self.data['td_mean'] else 0.0,
            'latest_td_std': self.data['td_std'][-1] if self.data['td_std'] else 0.0,
            'avg_td_mean': np.mean(self.data['td_mean']) if self.data['td_mean'] else 0.0,
            'avg_td_std': np.mean(self.data['td_std']) if self.data['td_std'] else 0.0,
            'num_stages': len(self.data['stage_markers']),
        }
    
    def print_summary(self):
        """打印统计摘要"""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("TD Error Tracker Summary".center(60))
        print("="*60)
        for key, value in stats.items():
            print(f"  {key:25s}: {value}")
        print("="*60 + "\n")


# 全局单例实例（可选）
_global_tracker = None

def get_global_tracker(save_dir: str = "./logs/td_tracking", 
                       experiment_name: str = "global_td") -> TDErrorTracker:
    """获取全局追踪器实例"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TDErrorTracker(save_dir, experiment_name)
    return _global_tracker

def reset_global_tracker():
    """重置全局追踪器"""
    global _global_tracker
    _global_tracker = None
