from utils import load_config
from openai import OpenAI

config = load_config()
EVAL_MODEL_NAME = config['EVAL_MODEL_NAME']
API_KEY = config['API_KEY']
BASE_URL = config['BASE_URL']

def get_openai_api(prompt: str):
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL+"/v1"
    )

    request_params = {
        "model": "gpt-4o",  
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }

    response = client.chat.completions.create(**request_params)
    return response.choices[0].message.content,1

def get_qwen_api(prompt: str):
    pass

if __name__ == "__main__":
    prompt = "你好"
    response = get_openai_api(prompt)
    print(response)

