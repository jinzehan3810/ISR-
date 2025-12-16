# CurricuLLM (DeepSeek & Humanoid Edition)

This repository is a modified version of CurricuLLM, adapted to support **DeepSeek** models and **Humanoid** robot tasks. It leverages Large Language Models (LLMs) to automatically design curricula and reward functions for reinforcement learning agents.

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jinzehan3810/ISR-.git
   cd ISR-
   ```

2. **Create and activate Conda environment**
   ```bash
   conda create -n CurricuLLM python=3.10
   conda activate CurricuLLM
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e environments
   pip install gymnasium  # Ensure gymnasium is installed
   ```

## 🔑 API Configuration

**Crucial Step**: You must configure your LLM API key before running the code, as the key file is not included in the repository for security reasons.

1. Create a new file named `ds_key.yaml` inside the `gpt/` directory.
2. Add your DeepSeek API key and base URL to the file:

   **File:** `gpt/ds_key.yaml`
   ```yaml
   OPENAI_API_KEY: "your-deepseek-api-key-here"
   base_url: "https://api.deepseek.com"
   ```

   > **Note:** The code is currently configured to use the `DeepSeek-V3.1-Terminus` model. If you wish to use a different model, you can modify the `GPT_MODEL` variable in `gpt/curriculum_api_chain_ant.py` and `gpt/curriculum_api_chain_fetch.py`.

## 🚀 Running the Code

### 1. Quick Test (Recommended for first run)
Run a quick test to verify that the LLM connection, code generation, and environment loading are working correctly. This runs a very short training session (200 steps) and finishes in minutes.

```bash
python main.py --task=AntMaze_Test --exp=curriculum --logdir=./logs_test --seed=0
```

### 2. Train Humanoid Robot to Walk (New Task)
Start the curriculum learning process for the Humanoid robot. This will generate tasks, reward functions, and train the agent to walk. 
*Note: This is a full training session and requires significant time and compute resources.*

```bash
python main.py --task=HumanoidWalk --exp=curriculum --logdir=./logs_humanoid --seed=0
```

### 3. Original Tasks
You can also run the original tasks supported by CurricuLLM:

```bash
# AntMaze
python main.py --task=AntMaze_UMaze --exp=curriculum --logdir=./logs --seed=0

# FetchPush
python main.py --task=FetchPush --exp=curriculum --logdir=./logs --seed=0
```

## 📂 Project Structure

- `configs/`: Configuration files for training (steps, environments, etc.).
- `environments/`: Custom Gym environments and source files for code injection.
- `gpt/`: LLM interaction logic and prompt templates.
- `train/`: Main training loop (Curriculum generation -> Code generation -> RL training).
- `traj_feedback/`: Tools for analyzing robot trajectories to provide feedback to the LLM.

## 📝 Notes

- **Training Time**: Real training (like HumanoidWalk) takes a significant amount of time (hours to days) and computational resources.
- **Logs**: Training logs, generated code, and curriculum files are saved in the directory specified by `--logdir`.
