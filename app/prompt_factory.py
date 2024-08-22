# coding=utf-8
import json

import app.config
from app.llm_tools import generate_by_ollama
from app.llm_tools import generate_by_openai
from app.tools import get_image_data_url


def extract_json(response):
    response = response.replace("JSON\n", "").replace("json\n", "").replace("```", "")
    json_start = response.index("{")
    json_end = response.rfind("}")
    return json.loads(response[json_start:json_end + 1])


def create_mj_prompts(category: str, prompt_count: int) -> list[str]:
    system = """Please generate prompts in bulk for the Midjourney drawing model based on the following input types. Each prompt should mainly consist of English phrases separated by commas. The style of the images should be vibrant in color with clear and concise lines. The output should be presented in JSON format as {"result": []}."""
    exp_1 = json.dumps({
        "result": [
            "A cute ginger cat playing with butterflies in the garden, with a house and rainbow in the background, in the style of a cartoon with bright colors, like a children's book illustration"
        ]
    })
    exp_2 = json.dumps({
        "result": [
            "coloring book illustration of a happy white cockatoo sits on the branch of an acacia tree, eating red berries and butterflies in its beak. The background is a blue sky with clouds. In the style of a cartoon coloring book, with thick lines, a black outline, colorful colors, and detailed design"
        ]
    })
    exp_3 = json.dumps({
        "result": [
            "A young couple hiking in the mountains with their dog by their side. The illustration is in a colorful cartoon style with line art. The man is wearing gray and black  while the woman wears a pink jacket over her head; they both have backpacks on their shoulders and are holding walking sticks. A golden retriever is next to them with a background of a green forest and mountain range"
        ]
    })

    messages = [
        {
            "role": "system",
            "content": system
        },
        {
            "role": "user",
            "content": "Type: Cat, Number: 1"
        },

        {
            "role": "assistant",
            "content": exp_1
        },
        {
            "role": "user",
            "content": "Type: 鸟类, Number: 1"
        },

        {
            "role": "assistant",
            "content": exp_2
        },
        {
            "role": "user",
            "content": "Type: Characters, Number: 1"
        },

        {
            "role": "assistant",
            "content": exp_3
        },
        {
            "role": "user",
            "content": f"Type: {category}, Number: {prompt_count}"
        }
    ]
    if 'ollama' in app.config.default_llm_type:
        output = generate_by_ollama(app.config.default_llm_model, messages)
    else:
        output = generate_by_openai(app.config.default_llm_model, messages, json_format=True)

    result = extract_json(output)
    return [item for item in result['result']]


def create_sd_prompts(category: str, prompt_count: int) -> list[str]:
    system = """Please generate prompts in bulk for the Stable Diffusion drawing model based on the theme type. Each prompt should mainly consist of English words or phrases separated by commas. The style of the images should be vibrant in color with clear and concise lines. The output should be presented in JSON format as {"result": []}."""

    exp_1 = json.dumps({
        "result": [
            "flower, pink flower, overalls, bouquet, braid, closed eyes, shirt, brown hair, hug, white shirt, smile, holding, short sleeves, leaf, long hair, orange flower, white flower, heart, 2girls, multiple girls, open mouth, sitting, red flower, indoors, holding bouquet, yellow flower, dress, 1boy, blue flower, tulip, blue overalls, happy birthday, english text, striped, table, denim, 1girl, long sleeves",
            "book, reading, library, glasses, shelves, pages, bookmark, study, lamp, cozy, chair, student, desk, open book, novel, author, literature, reading glasses, notebook, pen, writing, studying, quiet, knowledge, bookshelf, learning, quiet place, reading corner, coffee cup, armchair, reading room, soft light"
        ]
    })
    exp_2 = json.dumps({
        "result": [
            "duck, flower, no humans, bird, outdoors, sky, day, cloud, barrel, plant, grass, fence, blue sky, water, rose, duckling, tree, red flower, animal, bush, animal focus, house, bucket, vines, pink flower, yellow flower, building",
            "parrot, jungle, tropical, bright colors, flying, green leaves, branch, colorful feathers, birdwatching, nature, perch, exotic, wildlife, rainforest, beak, wing, feathers, tree branch, vivid, natural habitat, animal, squawking, avian, greenery, lush, natural, foliage, tropical bird, outdoor, perched, beak open"
        ]
    })
    exp_3 = json.dumps({
        "result": [
            "1girl, flower, striped shirt, shirt, bicycle, ponytail, sneakers, smiling, blue eyes, playground, sunny day, swing, bench, short hair, waving, skirt, school uniform, backpack, hand up, jumping, laughing, park, smiling, eyes closed, schoolbag, grass, running, cheerful, friends, talking, playing, walking, waving hand, waving goodbye",
            "1boy, cap, shorts, t-shirt, ball, playing, grass, sun, cheerful, running, happy, sneakers, playground, friends, laughing, energetic, summer day, child, outdoor activity, fun, smile, jumping, open space, blue sky, trees, sports, exercise, joyful, game, field, bright colors"
        ]
    })
    messages = [
        {
            "role": "system",
            "content": system
        },
        {
            "role": "user",
            "content": "Type: Collections, Number: 2"
        },

        {
            "role": "assistant",
            "content": exp_1
        },
        {
            "role": "user",
            "content": "Type: 鸟类, Number: 2"
        },

        {
            "role": "assistant",
            "content": exp_2
        },
        {
            "role": "user",
            "content": "Type: Characters, Number: 2"
        },

        {
            "role": "assistant",
            "content": exp_3
        },
        {
            "role": "user",
            "content": f"Type: {category}, Number: {prompt_count}"
        }
    ]
    if 'ollama' in app.config.default_llm_type:
        output = generate_by_ollama(app.config.default_llm_model, messages)
    else:
        output = generate_by_openai(app.config.default_llm_model, messages, json_format=True)

    result = extract_json(output)
    return [item for item in result['result']]


def des_image_prompt(image_path: str) -> str:
    system = """You are an AI specialized in creating concise yet detailed image descriptions for training image generation models. Analyze the input image and produce a description that:

1. Consists of short phrases or brief sentences
2. Separates each phrase or sentence with a comma (,) or period (.)
3. Starts with the main subject and its setting
4. Describes key visual elements, including colors, styles, and notable details
5. Uses clear, specific language
6. Emphasizes distinct colors and clear, crisp lines
7. Incorporates relevant artistic terms

Your output should be a single line of text, with phrases separated by commas or periods, similar to this example:

Vector illustration of a squirrel character in a forest setting. Golden-brown fur, large expressive eyes. Holding a red strawberry. Stylized cartoon aesthetic with bold outlines and flat colors."""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": system
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": get_image_data_url(image_path)
                    }
                }
            ]
        }
    ]
    output = generate_by_openai(app.config.default_llm_model, messages)
    return output


if __name__ == '__main__':
    prompts = des_image_prompt('./../temp/des/img_7.png')
    print(prompts)
    # print('prompts:')
    # for p in create_mj_prompts("Animals", 3):
    #     print(f"{p}\n")
