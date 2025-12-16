from openai import OpenAI
import yaml
import os
import numpy as np
import pandas as pd
import re

from gpt.utils import *

GPT_MODEL = "DeepSeek-V3.1-Terminus" # gpt-4-turbo-preview  deepseek-reasoner

class CurriculumAPI_Ant:
    def __init__(self, env_name, prompt_path, log_path):
        self.env = env_name
        self.client = get_client()
        self.prompt_path = prompt_path
        self.log_path = log_path

    def generate_curriculum(self):
        initial_system = file_to_string(self.prompt_path + self.env + '/curriculum_system.txt')
        initial_user = file_to_string(self.prompt_path + self.env + '/curriculum_user.txt')

        tasks_string = gpt_interaction(self.client, GPT_MODEL, initial_system, initial_user)

        # Ensure the directory exists and write the curriculum to a file
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path + 'curriculum.md', 'w', encoding='utf-8') as file:
            file.write(tasks_string)

        # Split the string into individual task sections
        task_sections = re.split(r'\n\n(?=Task)', tasks_string)

        # Function to extract details from each task section
        def extract_task_details(task_section):

            details = {}
            lines = task_section.split('\n')
            for line in lines:
                if line.startswith('Task'):
                    details['Task'] = line.split(' ')[1]
                elif line.startswith('Name:'):
                    details['Name'] = line.split(': ')[1].replace('[', '').replace(']', '').strip()
                elif line.startswith('Description:'):
                    details['Description'] = line.split(': ')[1]
                elif line.startswith('Reason:'):
                    details['Reason'] = ': '.join(line.split(': ')[1:])
            return details

        # Extract details for all tasks
        tasks_details = [extract_task_details(section) for section in task_sections]
        self.tasks_details = tasks_details

        # Return list of dictionaries with task details
        return tasks_details

    def generate_rewards(self, curriculum_idx, reward_code_history):
        task_detail = self.tasks_details[curriculum_idx]

        reward_system = file_to_string(self.prompt_path + self.env + '/reward_system.txt')
        reward_user = file_to_string(self.prompt_path + self.env + '/reward_user.txt')

        # Concatenate the task details into the user strings
        reward_user = reward_user.replace('<<Task_Name>>', task_detail['Name'])
        reward_user = reward_user.replace('<<Task_Description>>', task_detail['Description'])
        reward_user = reward_user.replace('<<Task_Reason>>', task_detail['Reason'])

        # Add previous task and reward information
        if curriculum_idx > 0:
            for i in range(curriculum_idx):
                task_history_details = self.tasks_details[i]
                reward_code = reward_code_history[i]
                reward_history = file_to_string(self.prompt_path + self.env + '/reward_history.txt')
                reward_history = reward_history.replace('<<Task_Name>>', task_history_details['Name'])
                reward_history = reward_history.replace('<<Task_Description>>', task_history_details['Description'])
                reward_history = reward_history.replace('<<Task_Reason>>', task_history_details['Reason'])
                reward_history = reward_history.replace('<<Task_Code>>', reward_code)

                reward_user = reward_user + "\n" + reward_history

        # Get reward function from GPT
        reward_answer = gpt_interaction(self.client, GPT_MODEL, reward_system, reward_user)

        pattern = r"`python\n(.*?)\n`"
        match = re.search(pattern, reward_answer, re.DOTALL)

        pattern = r"`threshold\n(.*?)\n`"
        threshold_match = re.search(pattern, reward_answer, re.DOTALL)

        if match and threshold_match:
            code_block = match.group(1)
            threshold_block = threshold_match.group(1)
            print("Extracted Code Block:\n", code_block)
            print("Extracted goal distance threshold:\n", threshold_block)
            return code_block, threshold_block
        else:
            print("No code block found.")
        return None, None

    def _load_curriculum_from_log(self):
        """从日志文件中加载课程信息，参考batch_plot.py中的extract_curriculum函数"""
        try:
            curriculum_file = self.log_path + "curriculum.md"
            if not os.path.exists(curriculum_file):
                print(f"Curriculum file not found: {curriculum_file}")
                return None
                
            with open(curriculum_file, 'r') as file:
                curriculum_txt = file.read()
            
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
                        details['Name'] = line.split(': ')[1].replace('[', '').replace(']', '').strip()
                    elif line.startswith('Description:'):
                        details['Description'] = line.split(': ')[1]
                    elif line.startswith('Reason:'):
                        details['Reason'] = ': '.join(line.split(': ')[1:])
                return details

            # Extract details for all tasks
            curriculum_info = [extract_task_details(section) for section in task_sections]
            print(f"Successfully loaded {len(curriculum_info)} tasks from curriculum.md")
            return curriculum_info
            
        except Exception as e:
            print(f"Error loading curriculum from log: {e}")
            return None

    def extract_best_agent_reward(self, curriculum_idx):
        """从日志中提取指定阶段的最佳agent的reward_code"""
        # 如果没有tasks_details，从日志文件中提取课程信息
        if not hasattr(self, 'tasks_details') or not self.tasks_details:
            curriculum_info = self._load_curriculum_from_log()
            if not curriculum_info:
                return None, None
            self.tasks_details = curriculum_info
            
        if curriculum_idx >= len(self.tasks_details):
            return None, None
            
        task = self.tasks_details[curriculum_idx]
        task_name = task['Name']
        
        try:
            # 读取决策文件获取最佳agent索引
            decision_file = self.log_path + task_name + '.md'
            if not os.path.exists(decision_file):
                print(f"Decision file not found: {decision_file}")
                return None, None
                
            with open(decision_file, 'r') as file:
                decision = file.read().split('\n')[0]
            
            numbers = re.findall(r'\d+', decision)
            if numbers:
                best_agent_idx = int(numbers[0])
            else:
                print(f"No agent number found in decision for {task_name}")
                return None, None
            
            # 读取对应sample的reward_code
            reward_code_file = self.log_path + f"{task_name}/sample_{best_agent_idx}/reward_code.md"
            threshold_code_file = self.log_path + f"{task_name}/sample_{best_agent_idx}/threshold_code.md"
            
            if not os.path.exists(reward_code_file):
                print(f"Reward code file not found: {reward_code_file}")
                return None, None
                
            with open(reward_code_file, 'r') as file:
                reward_code = file.read()
            
            threshold_code = None
            if os.path.exists(threshold_code_file):
                with open(threshold_code_file, 'r') as file:
                    threshold_code = file.read()
            
            print(f"Successfully loaded reward code from {task_name}/sample_{best_agent_idx}")
            return reward_code, threshold_code
            
        except Exception as e:
            print(f"Error loading best agent reward for {task_name}: {e}")
            return None, None
	
    def update_env_code(self, env_code_path, curriculum_idx, previous_reward_code=None, version_number=0, use_existing_best=False):
        # Created environment with task and save as version = env_version
        # 确保tasks_details可用
        if not hasattr(self, 'tasks_details') or not self.tasks_details:
            curriculum_info = self._load_curriculum_from_log()
            if not curriculum_info:
                raise ValueError("Cannot load curriculum information from log files")
            self.tasks_details = curriculum_info
            
        task = self.tasks_details[curriculum_idx]
        
        if use_existing_best:
            # 使用已有的最佳reward code
            print(f"Using existing best reward code for {task['Name']}")
            reward_code, threshold_code = self.extract_best_agent_reward(curriculum_idx)
            
            if reward_code is None:
                print(f"Failed to load existing best reward code, falling back to generation")
                use_existing_best = False
        
        if not use_existing_best:
            # 原有的生成逻辑
            reward_code = None
            max_attempt = 5
            attempt = 0
            while reward_code is None and attempt < max_attempt:
                reward_code, threshold_code = self.generate_rewards(curriculum_idx, previous_reward_code)
                attempt += 1
                if reward_code is None:
                    print("Failed to generate reward code. Retrying...")

        # Save the reward code
        save_string_to_file(self.log_path + f"{task['Name']}/sample_{version_number}/" + "reward_code.md", reward_code)
        if threshold_code:
            save_string_to_file(self.log_path + f"{task['Name']}/sample_{version_number}/" + "threshold_code.md", threshold_code)

        with open(env_code_path, 'r') as file:
            original_code = file.read()

        # Indent the code block with 4 spaces to the beginning of each line
        reward_code = '\n'.join('    ' + line for line in reward_code.splitlines())
        new_code = original_code + '\n\n' + reward_code

        # Save as a new file with specific version number
        new_file_path = env_code_path.replace('_source.py', '.py')
        with open(new_file_path, 'w') as file:
            file.write(new_code)

        # Save the threshold code if available
        if threshold_code:
            def insert_line_in_file(file_path, string_to_insert, line_number):
                # Read the current content of the file
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                
                # Insert the string at the specified line number
                lines.insert(line_number - 1, '        ' + string_to_insert + '\n')
                
                # Write the updated content back to the file
                with open(file_path, 'w') as file:
                    file.writelines(lines)

            insert_line_in_file(new_file_path, threshold_code, 271)
        
        print(f"Updated command code saved to {new_file_path}")

        return reward_code

    def feedback(self, env_name, task, curriculum_idx, statistics):
        feedback_system = file_to_string(self.prompt_path + env_name + '/feedback_system.txt')
        feedback_user = file_to_string(self.prompt_path + env_name + '/feedback_user.txt')

        # Concatenate the task details into the user strings
        feedback_user = feedback_user.replace('<<Task_Name>>', task['Name'])
        feedback_user = feedback_user.replace('<<Task_Description>>', task['Description'])
        feedback_user = feedback_user.replace('<<Task_Reason>>', task['Reason'])

        # Add previous task information
        if curriculum_idx > 0:
            for i in range(curriculum_idx):
                task_history_details = self.tasks_details[i]
                feedback_history = file_to_string(self.prompt_path + env_name + '/feedback_history.txt')
                feedback_history = feedback_history.replace('<<Task_Name>>', task_history_details['Name'])
                feedback_history = feedback_history.replace('<<Task_Description>>', task_history_details['Description'])
                feedback_history = feedback_history.replace('<<Task_Reason>>', task_history_details['Reason'])

                feedback_user = feedback_user + "\n" + feedback_history

        # Statistics to string
        feedback_statistics = ""
        for agent in range(len(statistics)):
            feedback_statistics += f"Agent {agent}:\n"
            for key, value in statistics[agent].items():
                feedback_statistics += f"{key}: {value}\n"
            feedback_statistics += "\n"

        feedback_user = feedback_user + "\n" + feedback_statistics

        gpt_answer = gpt_interaction(self.client, GPT_MODEL, feedback_system, feedback_user)
        
        # Ensure the directory exists and write the curriculum to a file
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path + task['Name'] + '_statistics.md', 'w') as file:
            file.write(feedback_statistics)        
        with open(self.log_path + task['Name'] + '.md', 'w') as file:
            file.write(gpt_answer)

        decision = gpt_answer.split('\n')[0]
        print("For task " + task['Name'] + ", GPT decided " + decision) 
        numbers = re.findall(r'\d+', decision)
        if numbers:
            return int(numbers[0])
        else:
            print("No number found in the decision.")
            return None

