# coding=utf-8
import requests

import app.config


def generate_by_ollama(model: str, messages: list[dict]):
    try:
        url = f'{app.config.api_ollama_host}/api/chat'
        data = {
            "model": model,
            "messages": messages,
            "options": {
                "repeat_penalty": 1.1,
                "top_k": 40,
                "top_p": 0.95,
                "temperature": 0.8
            },
            "stream": False
        }
        response = requests.post(url, json=data)
        return response.json()['message']['content']
    except Exception as e:
        print(f"generate_by_ollama error: {e}")
        return None


def generate_by_openai(model: str, messages: list[dict], json_format: bool = False):
    try:
        url = f'{app.config.api_openai_host}/v1/chat/completions'
        data = {
            "model": model,
            "messages": messages,
            "frequency_penalty": 1.1,
            "temperature": 0.8,
            "stream": False
        }
        if json_format:
            data["response_format"] = {
                "type": "json_object"
            }
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {app.config.api_openai_key}"
            },
            json=data
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"generate_by_openai error: {e}")
        return None
