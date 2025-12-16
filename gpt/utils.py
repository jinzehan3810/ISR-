from openai import OpenAI
import yaml
import os

GPT_KEY_PATH = './gpt/ds_key.yaml' # './gpt/gpt_key.yaml' './gpt/qwen_key.yaml'

def file_to_string(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()
    
def get_client():
    with open(GPT_KEY_PATH, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)

    client = OpenAI(api_key=config['OPENAI_API_KEY'], base_url=config['base_url'])

    return client
    
def gpt_interaction(client, gpt_model, system_string, user_string):
    trial = 0
    completion = None
    
    while completion is None and trial < 5:
        completion = client.chat.completions.create(
            model=gpt_model,
            messages=[
            {"role": "system", "content": system_string},
            {"role": "user", "content": user_string}
            ]
        )
        trial += 1

    print(completion.choices[0].message.content)

    return completion.choices[0].message.content

def save_string_to_file(save_path, string_file):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(string_file)