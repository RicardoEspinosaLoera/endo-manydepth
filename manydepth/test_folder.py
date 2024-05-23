# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import json
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import os

import torch
from torchvision import transforms

import networks


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for ManyDepth models.')

    parser.add_argument('--images_path', type=str,
                        help='path to a test image to predict for', required=True)
    parser.add_argument('--load_weights_folder', type=str,
                        help='path to a folder of weights to load', required=True)
    parser.add_argument('--mode', type=str, default='multi', choices=('multi', 'mono'),
                        help='"multi" or "mono". If set to "mono" then the network is run without '
                             'the source image, e.g. as described in Table 5 of the paper.',
                        required=False)
    return parser.parse_args()


def load_and_preprocess_image(image_path, resize_width, resize_height):
    image = pil.open(image_path).convert('RGB')
    original_width, original_height = image.size
    image = image.resize((resize_width, resize_height), pil.LANCZOS)
    image = transforms.ToTensor()(image).unsqueeze(0)
    if torch.cuda.is_available():
        return image.cuda(), (original_height, original_width)
    return image, (original_height, original_width)


def test_simple(args):
    """Function to predict for a single image or folder of images
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("-> Loading model from ", args.load_weights_folder)

    # Loading pretrained model
    print("   Loading pretrained encoder-decoder")
    
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    


    # Setting states of networks
    encoder = networks.mpvit_small() #networks.ResnetEncoder(opt.num_layers, False)
    encoder.num_ch_enc = [64,128,216,288,288]  # = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoderT()

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()


    # Load input data
    HEIGHT, WIDTH = 256, 320 
    dir_list = os.listdir(args.images_path)
    for i in dir_list:
        input_image, original_size = load_and_preprocess_image(os.path.join(args.images_path,i),resize_width=WIDTH,resize_height=HEIGHT)

        with torch.no_grad():
            # Estimate depth
            output = depth_decoder(encoder(input_image))[("disp", 0)]
            print(output.shape)

        

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
