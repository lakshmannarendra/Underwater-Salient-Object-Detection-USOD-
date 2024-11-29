import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import get_loader
import transforms as trans
from torchvision import transforms
import time
from Models.USOD_Net import ImageDepthNet
from torch.utils import data
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


def test_net(args):

    # Enable benchmark mode for faster performance on GPUs
    cudnn.benchmark = True

    # Initialize the network
    net = ImageDepthNet(args)
    net.cuda()  # Move the model to GPU
    net.eval()  # Set the model to evaluation mode

    # Load model (multi-GPU)
    model_path = args.save_model_dir + 'TC_USOD.pth'
    print(model_path)
    state_dict = torch.load(model_path)  # Load the model state
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # Remove `module.` prefix from keys
        new_state_dict[name] = v
    # Load parameters into the model
    net.load_state_dict(new_state_dict)

    print('Model loaded from {}'.format(model_path))

    test_paths = args.test_paths.split('+')  # Split test paths

    for test_dir_img in test_paths:

        # Get the test dataset
        test_dataset = get_loader(test_dir_img, args.data_root, args.img_size, mode='test')

        test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
        print('''
                   Starting testing:
                       dataset: {}
                       Testing size: {}
                   '''.format(test_dir_img.split('/')[0], len(test_loader.dataset)))

        time_list = []  # List to store time taken for each batch
        for i, data_batch in enumerate(test_loader):
            images, depths, image_w, image_h, image_path = data_batch
            images, depths = Variable(images.cuda()), Variable(depths.cuda())  # Move data to GPU

            starts = time.time()  # Start timing
            outputs_saliency = net(images, depths)  # Get model outputs
            ends = time.time()  # End timing
            time_use = ends - starts  # Calculate time taken
            time_list.append(time_use)  # Append time to list

            # Unpack the outputs
            d1, d2, d3, d4, d5, db, ud2, ud3, ud4, ud5, udb = outputs_saliency
            image_w, image_h = int(image_w[0]), int(image_h[0])  # Get image dimensions
            transform = trans.Compose([
                transforms.ToPILImage(),  # Convert tensor to PIL image
                trans.Scale((image_w, image_h))  # Resize image
            ])
            output_s = d1.data.cpu().squeeze(0)  # Move output to CPU and remove singleton dimension

            output_s = transform(output_s)  # Apply transformations

            dataset = test_dir_img.split('/')[0]  # Get dataset name
            filename = image_path[0].split('/')[-1].split('.')[0]  # Get filename without extension

            # Save saliency maps
            save_test_path = args.save_test_path_root + dataset + '/USOD10K/'
            if not os.path.exists(save_test_path):
                os.makedirs(save_test_path)  # Create directory if it doesn't exist
            output_s.save(os.path.join(save_test_path, filename + '.png'))  # Save the output image
        print('dataset:{}, cost:{}'.format(test_dir_img.split('/')[0], np.mean(time_list) * 1000))  # Print average time taken






