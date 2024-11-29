# Import necessary libraries
import os
import torch
import Training
import Testing
from Evaluation import main
import argparse

# Function to convert string to boolean
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Main execution block
if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser()
    # Training arguments
    parser.add_argument('--Training', default=False, type=str2bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33111', type=str, help='init_method')
    parser.add_argument('--data_root', default='', type=str, help='data path')
    parser.add_argument('--train_steps', default=60000, type=int, help='train_steps')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--pretrained_model', default='/Users/praneethreddy/Downloads/USOD10k/pretrained_model/80.7_T2T_ViT_t_14.pth.tar', type=str, help='load pretrained model')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=60000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=60000, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--trainset', default='/Users/praneethreddy/Downloads/USOD10k/USOD10k/USOD10K_TR', type=str, help='Training set')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')
    # Testing arguments
    parser.add_argument('--Testing', default=False, type=str2bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--test_paths', type=str, default='/Users/praneethreddy/Downloads/USOD10k/USOD10k/USOD10K_TE')

    # Evaluation arguments
    parser.add_argument('--Evaluation', default=False, type=str2bool, help='Evaluation or not')
    parser.add_argument('--methods', type=str, default='USOD10K', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')

    # Parse the arguments
    args = parser.parse_args()

    # Set the environment variable for CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    # Execute training if specified
    if args.Training:
        Training.train_net(num_gpus=num_gpus, args=args)
    # Execute testing if specified
    if args.Testing:
        Testing.test_net(args)
    # Execute evaluation if specified
    if args.Evaluation:
        main.evaluate(args)
