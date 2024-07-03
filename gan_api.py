# coding=utf-8
import os
import shutil

from flask import Flask, request, jsonify
from models import create_model
from options.test_options import TestOptions
from util import util

from data import create_dataset

app = Flask(__name__)


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


@app.route('/extract_line', methods=['POST'])
def extract_line():
    dataroot = request.json['dataroot']
    dst_dir = request.json['dst_dir']

    print(f'dataroot: {dataroot}')
    print(f'dst_dir: {dst_dir}')

    os.makedirs(dst_dir, exist_ok=True)

    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    opt.dataroot = dataroot
    opt.name = 'sketch_pix2pix_512'
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

        result.extend(save_images(name, visuals, dst_dir))

    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)

    return jsonify({'paths': result})


app.run(host='0.0.0.0', port=9091)
