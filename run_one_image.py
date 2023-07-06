"""
It is used to run the model on one image (and its corresponding trimap).

For default, run:
python run_one_image.py \
    --model vitmatte-s \
    --checkpoint-dir path/to/checkpoint
It will be saved in the directory ``./demo``.

If you want to run on your own image, run:
python run_one_image.py \
    --model vitmatte-s(or vitmatte-b) \
    --checkpoint-dir <your checkpoint directory> \
    --image-dir <your image directory> \
    --trimap-dir <your trimap directory> \
    --output-dir <your output directory> \
    --device <your device>
"""
import os
from PIL import Image
from os.path import join as opj
from torchvision.transforms import functional as F
from detectron2.engine import default_argument_parser
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import numpy as np
import cv2
import torch

def infer_one_image(model, input, save_dir=None):
    """
    Infer the alpha matte of one image.
    Input:
        model: the trained model
        image: the input image
        trimap: the input trimap
    """
    output = model(input)['phas'].flatten(0, 2)
    output = F.to_pil_image(output)
    
    return output

def init_model(model, checkpoint, device):
    """
    Initialize the model.
    Input:
        config: the config file of the model
        checkpoint: the checkpoint of the model
    """
    assert model in ['vitmatte-s', 'vitmatte-b']
    if model == 'vitmatte-s':
        config = 'configs/common/model.py'
        cfg = LazyConfig.load(config)
        model = instantiate(cfg.model)
        model.to('cuda')
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)
    elif model == 'vitmatte-b':
        config = 'configs/common/model.py'
        cfg = LazyConfig.load(config)
        cfg.model.backbone.embed_dim = 768
        cfg.model.backbone.num_heads = 12
        cfg.model.decoder.in_chans = 768
        model = instantiate(cfg.model)
        model.to('cuda')
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)
    return model

def get_data(image_dir, trimap_dir):
    """
    Get the data of one image.
    Input:
        image_dir: the directory of the image
        trimap_dir: the directory of the trimap
    """
    image = Image.open(image_dir).convert('RGB').resize((1024,1024))
    image = F.to_tensor(image).unsqueeze(0)
    trimap = Image.open(trimap_dir).convert('L').resize((1024,1024))
    trimap = F.to_tensor(trimap).unsqueeze(0)

    return {
        'image': image,
        'trimap': trimap
    }

def generate_checkerboard_image(height, width, num_squares):
    num_squares_h = num_squares
    square_size_h = height // num_squares_h
    square_size_w = square_size_h
    num_squares_w = width // square_size_w
    

    new_height = num_squares_h * square_size_h
    new_width = num_squares_w * square_size_w
    image = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(num_squares_h):
        for j in range(num_squares_w):
            start_x = j * square_size_w
            start_y = i * square_size_h
            color = 255 if (i + j) % 2 == 0 else 200
            image[start_y:start_y + square_size_h, start_x:start_x + square_size_w] = color

    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image

if __name__ == '__main__':
    #add argument we need:
    parser = default_argument_parser()
    parser.add_argument('--model', type=str, default='vitmatte-s')
    parser.add_argument('--checkpoint-dir', type=str, default='pretrained/ViTMatte_S_Com.pth')
    parser.add_argument('--image-dir', type=str, default='demo/bulb_rgb.png')
    parser.add_argument('--trimap-dir', type=str, default='demo/bulb_trimap.png')
    parser.add_argument('--output-dir', type=str, default='demo/result.png')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    input = get_data(args.image_dir, args.trimap_dir)
    print('Initializing model...Please wait...')
    model = init_model(args.model, args.checkpoint_dir, args.device)
    print('Model initialized. Start inferencing...')
    alpha = infer_one_image(model, input, args.output_dir)
    print('Inferencing finished.')
    
    input_x = Image.open(args.image_dir)
    
    # Resize alpha mask to fit input image
    alpha = alpha.resize(input_x.size)
    foreground_alpha = input_x.copy()
    # Put the alpha onto the image
    foreground_alpha.putalpha(alpha)
    # save image to output
    foreground_alpha.save(args.output_dir)