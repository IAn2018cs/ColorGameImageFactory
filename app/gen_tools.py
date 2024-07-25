# coding=utf-8
import mimetypes
import os
from urllib.parse import urlparse

import requests
from requests_toolbelt import MultipartEncoder

import app.config
from app.tools import create_path


def is_hidden(filepath):
    name = os.path.basename(os.path.abspath(filepath))
    return name.startswith('.')


def extract_line_by_gan(source_dir, dst_dir):
    create_path(dst_dir)
    # 准备文件列表
    files = []
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if os.path.isfile(file_path) and not is_hidden(file_path):
            files.append(file_path)
    # 准备 MultipartEncoder
    fields = {}
    for i, file_path in enumerate(files):
        file_name = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        fields[f'files[{i}]'] = (file_name, open(file_path, 'rb'), mime_type or 'application/octet-stream')

    m = MultipartEncoder(fields=fields)

    # 发送请求
    headers = {'Content-Type': m.content_type}
    response = requests.post(f'{app.config.gan_api_host}/extract_line', data=m, headers=headers)

    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    # 获取返回的 URLs
    urls = response.json()['urls']

    # 下载文件到目标目录
    downloaded_paths = []
    for url in urls:
        filename = os.path.basename(urlparse(url).path)
        dst_path = os.path.join(dst_dir, filename)

        # 下载文件
        file_response = requests.get(url)
        if file_response.status_code == 200:
            with open(dst_path, 'wb') as f:
                f.write(file_response.content)
            downloaded_paths.append(dst_path)
        else:
            print(f"Failed to download {url}")

    return downloaded_paths
