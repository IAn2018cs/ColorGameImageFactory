# coding=utf-8
import requests

import app.config


def extract_line_by_gan(source_dir, dst_dir):
    result = requests.post(f'{app.config.gan_api_host}/extract_line',
                           json={'dataroot': source_dir, 'dst_dir': dst_dir})
    return result.json()['paths']
