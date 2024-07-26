# coding=utf-8

import enum
import json
import random
import re
import time

import requests

import app.config
from app.tools import get_base64_image


class TaskAction(enum.Enum):
    IMAGINE = "IMAGINE"
    UPSCALE = "UPSCALE"
    VARIATION = "VARIATION"
    REROLL = "REROLL"
    DESCRIBE = "DESCRIBE"
    BLEND = "BLEND"


class TaskStatus(enum.Enum):
    NOT_START = "NOT_START"
    SUBMITTED = "SUBMITTED"
    IN_PROGRESS = "IN_PROGRESS"
    FAILURE = "FAILURE"
    SUCCESS = "SUCCESS"


class MJBot:

    def __init__(self):
        self.TAG = 'MJBot'
        self.host = app.config.mj_host
        self.api_key = app.config.mj_api_key
        self.max_attempts = 60
        self.fetch_time_interval = 10

    def __request__(self, path: str, body: dict) -> dict:
        try:
            response = requests.post(f'{self.host}{path}',
                                     headers={"mj-api-secret": f"{self.api_key}"},
                                     json=body)
            return response.json()
        except Exception as e:
            print(f'__request__ error: {e}')
            return {'code': -1}

    def __fetch_task__(self, task_id):
        try:
            response = requests.get(f'{self.host}/mj/task/{task_id}/fetch',
                                    headers={"mj-api-secret": f"{self.api_key}"})
            resp = response.json()
            task_status = resp['status']
            if task_status in [TaskStatus.SUCCESS.value, TaskStatus.FAILURE.value]:
                return resp
            return None
        except Exception as e:
            print(f'__fetch_task__ error: {e}')
            return None

    def __do_fetch__(self, task_id, interval=None):
        if interval is None:
            interval = self.fetch_time_interval
        max_attempts = self.max_attempts
        time.sleep(interval)
        for attempt in range(max_attempts):
            result = self.__fetch_task__(task_id)
            if result:
                return result
            print(f"Attempt {attempt + 1}/{max_attempts} failed, retrying in {interval} seconds...")
            time.sleep(interval)
        print(f"fetch {task_id} result failed")
        return None

    def create_image(self, prompt: str, img_path: str = None, payload: dict = None, notify_hook: str = None):
        imagine = {
            'prompt': prompt
        }
        if payload:
            imagine['state'] = json.dumps(payload)
        if notify_hook:
            imagine['notifyHook'] = notify_hook
        if img_path:
            imagine['base64Array'] = [f"data:image/png;base64,{get_base64_image(img_path)}"]
        result = self.__request__('/mj/submit/imagine', body=imagine)
        if result['code'] != 1:
            return None
        return result['result']

    def create_image_and_fetch(self, prompt: str, img_path: str = None):
        task_id = self.create_image(prompt, img_path)
        if task_id is None:
            return None
        result = self.__do_fetch__(task_id)
        if result:
            return result['imageUrl']
        return None

    def create_single_image_and_fetch(self, prompt: str, img_path: str = None):
        task_id = self.create_image(prompt, img_path)
        if task_id is None:
            return None
        result = self.__do_fetch__(task_id)
        if result is None:
            return None
        new_task_id = result['id']
        random_index = random.randint(1, 4)
        return self.change_image_and_fetch(new_task_id, TaskAction.UPSCALE, random_index)

    def change_image(self, task_id: str, action: TaskAction, index: int = 1, payload: dict = None,
                     notify_hook: str = None):
        change = {
            'action': action.value,
            'index': index,
            'taskId': task_id
        }
        if notify_hook:
            change['notifyHook'] = notify_hook
        if payload:
            change['state'] = json.dumps(payload)
        result = self.__request__('/mj/submit/change', body=change)
        if result['code'] != 1:
            return None
        return result['result']

    def change_image_and_fetch(self, task_id: str, action: TaskAction, index: int = 1):
        new_task_id = self.change_image(task_id, action, index)
        if new_task_id is None:
            return None
        result = self.__do_fetch__(new_task_id)
        if result:
            return result['imageUrl']
        return None

    def des_image(self, img_path: str, payload: dict = None, notify_hook: str = None):
        des = {
            'base64': f"data:image/png;base64,{get_base64_image(img_path)}",
        }
        if payload:
            des['state'] = json.dumps(payload)
        if notify_hook:
            des['notifyHook'] = notify_hook
        result = self.__request__('/mj/submit/describe', body=des)
        if result['code'] != 1:
            return None
        return result['result']

    def des_image_and_fetch(self, img_path: str):
        task_id = self.des_image(img_path)
        if task_id is None:
            return None
        result = self.__do_fetch__(task_id, interval=3)
        if result:
            prompt = [re.sub(r'--ar \d+:\d+', '', p[4:]).strip() for p in str(result['promptEn']).split("\n\n")]
            return prompt
        return None
