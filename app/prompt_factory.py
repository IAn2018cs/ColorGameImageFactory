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
    system = """You are an expert in creating detailed prompts for Midjourney, specializing in images with clear lines and distinct colors. Your task is to generate vivid and diverse image descriptions based on a user-provided theme, emphasizing sharp outlines and bold color palettes.

Input format:
Theme: [theme], Number: [number of prompts]

Your output should be in JSON format as follows:
{
    "result": [
        "Generated prompt 1",
        "Generated prompt 2",
        ...
    ]
}

When creating prompts, follow these guidelines:

1. Emphasize clear, well-defined lines and shapes in the image description.
2. Use bold, distinct color palettes with minimal gradients or subtle shading.
3. Focus on compositions that would translate well to images with sharp contrasts.
4. Incorporate a variety of elements such as subject matter, composition, and lighting.
5. Be specific about details and artistic techniques that enhance clarity and color distinction.
6. Consider various art styles that naturally feature clear lines and bold colors (e.g., pop art, digital illustration, graphic novel style).
7. Ensure each prompt is unique and different from the others.

Here's an example of how you should format your response:

Input:
Theme: City Life, Number: 2

Output:
{
    "result": [
        "Busy city crosswalk viewed from above. Crisp white pedestrian lines contrasting with dark asphalt. People crossing in colorful, distinctly outlined clothing. Bold red and green traffic lights. Surrounding buildings with clearly defined edges and windows. Graphic novel style with thick black outlines and flat, vibrant color fills.",
        "Sleek modern subway station interior. Shiny steel pillars and clean white walls with sharp edges. Digital information boards displaying bright text against dark backgrounds. Commuters in clearly defined silhouettes, each in a distinct bold color. Minimalist illustration style with geometric shapes and a limited color palette of primary colors plus black and white."
    ]
}

Now, please wait for the user to input a theme and number, then create the prompts accordingly."""

    messages = [
        {
            "role": "system",
            "content": system
        },
        {
            "role": "user",
            "content": f"Theme: {category}, Number: {prompt_count}"
        }
    ]
    if 'ollama' in app.config.default_llm_type:
        output = generate_by_ollama(app.config.default_llm_model, messages)
    else:
        output = generate_by_openai(app.config.default_llm_model, messages, json_format=True)

    result = extract_json(output)
    return [item for item in result['result']]


def create_mj_prompts_v2(category: str, prompt_count: int) -> list[str]:
    system = """You are a professional Midjourney prompt creator. Your task is to generate descriptive and imaginative prompts based on the user's given theme and requested number. These prompts will be used to create clear, detailed images suitable for various styles, including line art and colorful cartoon illustrations.

User input will follow this format:
Theme: [theme], Number: [number of prompts to generate]

When creating prompts, follow these guidelines:

1. Develop engaging scenes or compositions based on the given theme.
2. Describe main elements in detail, including shapes, poses, and characteristics.
3. Mention distinctive features such as expressions, textures, or unique details.
4. Describe background elements that complement the theme and enhance the overall atmosphere.
5. Emphasize clarity of lines and precision of details.
6. Describe colors (if applicable) and compositional balance.
7. Use natural, flowing sentences or phrases separated by English commas or periods.
8. Keep prompts concise yet detailed.
9. Avoid specific art style terminology to maintain versatility.
10. Do not use words like "blur", "soft focus", or "shadow" that might affect image generation.
11. Focus on describing clear, distinct visual elements and details.

Your output must strictly adhere to the following JSON format:

{
    "result": [
        "Generated prompt 1",
        "Generated prompt 2",
        ...
    ]
}

Where:
- The number of generated prompts in the "result" array must match the number specified in the user input.
- Each "Generated prompt" should be a complete, self-contained string describing an image based on the given theme.
- Do not include any numbering or additional formatting within the prompt strings.

Output only the JSON object without any additional explanations or comments."""

    messages = [
        {
            "role": "system",
            "content": system
        },
        {
            "role": "user",
            "content": f"Theme: {category}, Number: {prompt_count}"
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
    # prompts = des_image_prompt('./../temp/des/img_9.png')
    # print(prompts)
    print('prompts:')
    for p in create_mj_prompts_v2("butterfly", 2):
        print(f"{p}\n")
