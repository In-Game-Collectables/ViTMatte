import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os

import tqdm
import glob
from segment_anything import sam_model_registry, SamPredictor
from detectron2.engine import default_argument_parser


def generate_trimap(mask, erode_kernel_size=2, dilate_kernel_size=2):
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    eroded = cv2.erode(mask, erode_kernel, iterations=5)
    dilated = cv2.dilate(mask, dilate_kernel, iterations=5)
    trimap = np.zeros_like(mask)
    trimap[dilated==255] = 128
    trimap[eroded==255] = 255
    return trimap

if __name__ == '__main__':
    #add argument we need:
    parser = default_argument_parser()
    parser.add_argument('--model', type=str, default='vit_h')
    parser.add_argument('--checkpoint-dir', type=str, default='pretrained/')
    parser.add_argument('--image-dir', type=str, default='dataset/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scale_x', type=float, default=0.2)  # Size of box in x-axis
    parser.add_argument('--scale_y', type=float, default=0.05)  # Size of box in the y-axis
    parser.add_argument('--erode', type=int, default=2)  # Erosion kernel size
    parser.add_argument('--dilate', type=int, default=2)  # Dilation kernel size
    
    args = parser.parse_args()

    sam_checkpoint = os.path.join(args.checkpoint_dir,"sam_vit_h_4b8939.pth")
    model_type = args.model
    device = args.device

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # create trimap directory
    trimap_dir = os.path.join(os.path.dirname(args.image_dir), f'trimap')
    os.makedirs(trimap_dir, exist_ok=True)
        
    img_paths = glob.glob(os.path.join(args.image_dir, 'images/*'))
    
    for img_path in tqdm.tqdm(img_paths):
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        predictor.set_image(image)
        
        img_width = image.shape[0]
        img_height = image.shape[1]
        
        # Bounding box
        x1, y1 = args.scale_x*img_width, args.scale_y*img_height
        x2, y2 = img_width-x1, img_height-y1
        input_box = np.array([x1, y1, x2, y2]).astype(int) # Default bounding box is the center of the screen
        
        # Additional single mask point at the center of image
        ctr_x = img_width/2
        ctr_y = img_height/2
        input_point = np.array([[ctr_x, ctr_y], [ctr_x, ctr_y-50], [ctr_x, ctr_y+50]]).astype(int)
        input_label = np.array([1, 1, 1])

        masks, _, _ = predictor.predict(
            point_coords = input_point,
            point_labels = input_label,
            box = input_box[None, :],
            multimask_output = False,
        )

        mask_all = np.ones((image.shape[0], image.shape[1], 3))
        for ann in masks:
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                mask_all[ann[0] == True, i] = color_mask[i]
        img = image / 255 * 0.3 + mask_all * 0.7

        # generate alpha matte
        mask = masks[0].astype(np.uint8)*255
        trimap = generate_trimap(mask, args.erode, args.dilate).astype(np.float32)

        out_path = os.path.join(trimap_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png')
        cv2.imwrite(out_path, trimap)
