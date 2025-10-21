# TD误差监控使用指南

## 📖 概述

这个系统分为两部分：
1. **TD误差追踪器** (`utils/td_error_tracker.py`) - 负责记录和可视化TD误差
2. **监控SAC** (`train/Monitored_SAC.py`) - 在训练时自动记录TD误差到追踪器

## 🚀 快速开始

### 第一步：测试追踪器功能

运行测试脚本，看看追踪器的基本功能：

```bash
python test_td_tracker.py
```

这会生成模拟的TD误差数据和图表，保存在 `./logs/td_test/` 目录。

### 第二步：在实际训练中使用

查看示例脚本：

```bash
python example_monitored_training.py
```

## 📝 详细使用方法

### 1. 基础使用（单阶段训练）

```python
from utils.td_error_tracker import TDErrorTracker
from train.Monitored_SAC import MonitoredSAC

# 创建追踪器
td_tracker = TDErrorTracker(
    save_dir="./logs/td_tracking",
    experiment_name="my_experiment"
)

# 创建模型（传入追踪器）
model = MonitoredSAC(
    "MultiInputPolicy",
    env,
    td_tracker=td_tracker,    # 关键参数
    record_freq=10,           # 记录频率
    verbose=1
)

# 训练
model.learn(total_timesteps=1000000)

# 保存数据和生成图表
td_tracker.save()
td_tracker.plot()
```

### 2. 多阶段训练（课程学习）

```python
# 创建全局追踪器
td_tracker = TDErrorTracker(
    save_dir="./logs/curriculum_td",
    experiment_name="curriculum_experiment"
)

# 阶段1
td_tracker.set_stage("curriculum_stage_1")
model1 = MonitoredSAC("MultiInputPolicy", env1, td_tracker=td_tracker)
model1.learn(total_timesteps=500000)

# 阶段2
td_tracker.set_stage("curriculum_stage_2")
model2 = MonitoredSAC("MultiInputPolicy", env2, td_tracker=td_tracker)
model2.learn(total_timesteps=500000)

# 阶段3
td_tracker.set_stage("curriculum_stage_3")
model3 = MonitoredSAC("MultiInputPolicy", env3, td_tracker=td_tracker)
model3.learn(total_timesteps=500000)

# 保存和可视化
td_tracker.save()
td_tracker.plot()  # 会显示所有阶段的TD误差变化
```

### 3. 在现有的 SAC_Module 中集成

修改 `train/SAC_Module.py`:

```python
from utils.td_error_tracker import TDErrorTracker
from train.Monitored_SAC import MonitoredSAC

class SAC_Module:
    def __init__(self, env_name, env_path, logger_path, cfg, seed=0, td_tracker=None):
        # ...现有代码...
        self.td_tracker = td_tracker  # 添加追踪器
    
    def train(self):
        # ...创建环境...
        
        # 如果有追踪器，使用MonitoredSAC
        if self.td_tracker is not None:
            model = MonitoredSAC(
                self.cfg['policy_network'],
                training_env,
                td_tracker=self.td_tracker,
                record_freq=10,
                verbose=1,
                tensorboard_log=self.logger_path + "sac/",
            )
        else:
            # 否则使用标准SAC
            model = SAC(...)
        
        model.learn(...)
```

### 4. 在课程学习中使用

修改 `train/Curriculum_Module.py`:

```python
from utils.td_error_tracker import TDErrorTracker

class Curriculum_Module:
    def __init__(self, env_name, env_path, logger_path, cfg, seed=0):
        # ...现有代码...
        
        # 创建全局TD追踪器
        self.td_tracker = TDErrorTracker(
            save_dir=logger_path + "td_tracking/",
            experiment_name=f"{env_name}_curriculum"
        )
    
    def train_single(self, curriculum_idx, task, sample_num):
        # 设置当前阶段
        stage_name = f"curriculum_{curriculum_idx}_task_{task['Name']}_sample_{sample_num}"
        self.td_tracker.set_stage(stage_name)
        
        # 创建模型时传入追踪器
        model = MonitoredSAC(
            ...,
            td_tracker=self.td_tracker,
            record_freq=10
        )
        
        # 训练...
    
    def train(self):
        self.generate_curriculum()
        
        for curriculum_idx in range(self.curriculum_length):
            for sample_num in range(self.cfg["num_samples"]):
                # 训练...
        
        # 训练结束后保存和可视化
        self.td_tracker.save()
        self.td_tracker.plot()
        self.td_tracker.print_summary()
```

## 📊 可视化功能

### 基本图表

```python
# 生成标准图表（包含3个子图）
td_tracker.plot(save_path="./my_plot.png", show=True)
```

生成的图表包含：
- TD误差均值随时间变化
- TD误差标准差随时间变化
- TD误差方差和奖励方差对比
- 阶段分隔线（红色虚线）

### 对比多个实验

```python
# 加载多个追踪器数据
tracker1 = TDErrorTracker("./logs/exp1", "exp1")
tracker1.load("./logs/exp1/exp1_data.pkl")

tracker2 = TDErrorTracker("./logs/exp2", "exp2")
tracker2.load("./logs/exp2/exp2_data.pkl")

# 对比图表
tracker1.plot_comparison(
    [tracker2],
    labels=["Baseline", "With Normalization"],
    save_path="./comparison.png"
)
```

## 🔧 配置参数

### TDErrorTracker 参数

- `save_dir`: 保存目录
- `experiment_name`: 实验名称

### MonitoredSAC 参数

- `td_tracker`: TDErrorTracker实例（必需）
- `record_freq`: 记录频率（默认10，表示每10个gradient steps记录一次）
- 其他参数同标准SAC

## 📁 输出文件

训练后会生成以下文件：

```
logs/td_tracking/
├── my_experiment_data.pkl      # 完整数据（pickle格式）
├── my_experiment_data.json     # 数据（JSON格式，可读）
└── my_experiment_plot.png      # 可视化图表
```

## 💡 使用建议

1. **记录频率**: 
   - 对于快速任务：`record_freq=5`
   - 对于正常任务：`record_freq=10`（默认）
   - 对于长时间任务：`record_freq=20`

2. **内存管理**: 追踪器会保存所有记录点的数据，如果训练非常长，可以增大 `record_freq` 以减少数据量

3. **多实验对比**: 建议为每个实验使用不同的 `experiment_name`，便于后续对比

4. **定期保存**: 在长时间训练中，可以定期调用 `td_tracker.save()` 避免数据丢失

## 🐛 故障排除

### 问题1：图表没有显示阶段分隔线
**原因**: 没有调用 `set_stage()`  
**解决**: 在每个训练阶段开始时调用 `td_tracker.set_stage("stage_name")`

### 问题2：数据为空
**原因**: `record_freq` 设置太大，或训练步数太少  
**解决**: 降低 `record_freq` 或增加训练步数

### 问题3：内存占用过大
**原因**: 记录点太多  
**解决**: 增大 `record_freq`

## 📚 下一步

现在您已经可以监控TD误差了。下一步我们可以：

1. ✅ **当前完成**: TD误差监控和可视化
2. ⏭️ **下一步**: 添加TD误差归一化功能
3. ⏭️ **未来**: 自动调整归一化参数

您可以先运行测试看看效果，然后我们再继续实现归一化功能！
