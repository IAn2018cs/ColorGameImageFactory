# coding=utf-8
import json

import app.config
from app.llm_tools import generate_by_ollama
from app.llm_tools import generate_by_openai


def create_sd_prompts(category: str, prompt_count: int) -> list[str]:
    system = """
    请根据以下输入类型，批量生成用于 Stable Diffusion 绘画模型的 prompt。每个 prompt 主要以英文单词或短语组成，之间用英文 , 分割。生成的图片将用于填色游戏，所以图片的风格应该是漫画风，色彩鲜明，线条清晰。输出以 {"result": []} 的 JSON 格式呈现。

    输入类型：Collections
    输出：
    {
        "result": [
            {"prompt": "flower, pink flower, overalls, bouquet, braid, closed eyes, shirt, brown hair, hug, white shirt, smile, holding, short sleeves, leaf, long hair, orange flower, white flower, heart, 2girls, multiple girls, open mouth, sitting, red flower, indoors, holding bouquet, yellow flower, dress, 1boy, blue flower, tulip, blue overalls, happy birthday, english text, striped, table, denim, 1girl, long sleeves"},
            {"prompt": "book, reading, library, glasses, shelves, pages, bookmark, study, lamp, cozy, chair, student, desk, open book, novel, author, literature, reading glasses, notebook, pen, writing, studying, quiet, knowledge, bookshelf, learning, quiet place, reading corner, coffee cup, armchair, reading room, soft light"},
            ...
        ]
    }

    输入类型：鸟类
    输出：
    {
        "result": [
            {"prompt": "duck, flower, no humans, bird, outdoors, sky, day, cloud, barrel, plant, grass, fence, blue sky, water, rose, duckling, tree, red flower, animal, bush, animal focus, house, bucket, vines, pink flower, yellow flower, building"},
            {"prompt": "parrot, jungle, tropical, bright colors, flying, green leaves, branch, colorful feathers, birdwatching, nature, perch, exotic, wildlife, rainforest, beak, wing, feathers, tree branch, vivid, natural habitat, animal, squawking, avian, greenery, lush, natural, foliage, tropical bird, outdoor, perched, beak open"},
            ...
        ]
    }

    输入类型：Characters
    输出：
    {
        "result": [
            {"prompt": "1girl, flower, striped shirt, shirt, bicycle, ponytail, sneakers, smiling, blue eyes, playground, sunny day, swing, bench, short hair, waving, skirt, school uniform, backpack, hand up, jumping, laughing, park, smiling, eyes closed, schoolbag, grass, running, cheerful, friends, talking, playing, walking, waving hand, waving goodbye"},
            {"prompt": "1boy, cap, shorts, t-shirt, ball, playing, grass, sun, cheerful, running, happy, sneakers, playground, friends, laughing, energetic, summer day, child, outdoor activity, fun, smile, jumping, open space, blue sky, trees, sports, exercise, joyful, game, field, bright colors"},
            ...
        ]
    }
    """.strip()

    prompt = f'现在请根据以下输入类型生成 {prompt_count}' + ' 条类似的 prompt，以 {"result": []} 的 JSON 格式输出：\n'
    prompt += f'输入类型：{category}\n'
    prompt += """输出：
    {
        "result": [
            {"prompt": ""},
            {"prompt": ""},
            ...
        ]
    }""".strip()

    messages = [
        {
            "role": "system",
            "content": system
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    if 'ollama' in app.config.default_llm_type:
        output = generate_by_ollama(app.config.default_llm_model, messages)
    else:
        output = generate_by_openai(app.config.default_llm_model, messages, json_format=True)

    result = json.loads(output)
    return [item['prompt'] for item in result['result']]
