# coding=utf-8
import os
import shutil
import time

from flask import Flask, request, jsonify, url_for, send_from_directory
from models import create_model
from options.test_options import TestOptions
from util import util
from werkzeug.utils import secure_filename

from data import create_dataset

app = Flask(__name__, root_path=os.path.dirname(os.path.abspath(__file__)))
app.config['CACHE_DIR'] = os.path.join(app.root_path, 'temp_dst_dir')
os.makedirs(app.config['CACHE_DIR'], exist_ok=True)


def save_images(ori_name, visuals, root_dir, aspect_ratio=1.0):
    """Save images to the disk.

    Parameters:
        ori_name (str)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        root_dir (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
    """

    ims = []
    for label, im_data in visuals.items():
        if label == 'real':
            continue
        im = util.tensor2im(im_data)
        image_name = f'{ori_name}_line.png'
        save_path = os.path.join(root_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(save_path)
    return ims


@app.route('/downloads/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['CACHE_DIR'], filename, as_attachment=True)


@app.route('/extract_line', methods=['POST'])
def extract_line():
    # 创建带时间戳的唯一临时目录
    timestamp = int(time.time())
    temp_dir = f'temp_uploads_{timestamp}'
    dataroot = os.path.join(app.root_path, 'temp_uploads', temp_dir)
    os.makedirs(dataroot, exist_ok=True)

    # 设置输出目录
    dst_dir = app.config['CACHE_DIR']

    # 保存上传的文件
    saved_files = []
    for key, file in request.files.items():
        filename = secure_filename(file.filename)
        file_path = os.path.join(dataroot, filename)
        file.save(file_path)
        saved_files.append(file_path)

    if not saved_files:
        return jsonify({'error': 'No valid files uploaded'}), 400

    print(f'dataroot: {dataroot}')
    print(f'dst_dir: {dst_dir}')

    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    opt.dataroot = dataroot
    opt.name = 'stroke_pix2pix'
    opt.model = 'test'
    opt.netG = 'unet_256'
    opt.direction = 'AtoB'
    opt.dataset_mode = 'single'
    opt.norm = 'batch'
    opt.preprocess = 'scale_width'
    opt.load_size = 1024
    opt.crop_size = 512

    cache_path = f'{dataroot}.ipynb_checkpoints'
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    result = []

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths

        file_name_split = os.path.split(img_path[0])[-1].split('.')
        name = file_name_split[0]

        saved_paths = save_images(name, visuals, dst_dir)

        # Convert file paths to URLs
        urls = [url_for('download_file', filename=os.path.basename(path), _external=True) for path in saved_paths]
        result.extend(urls)

    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)

    return jsonify({'urls': result})


app.run(host='0.0.0.0', port=9091)
