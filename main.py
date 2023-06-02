import argparse
import torch
from torch.utils.data import DataLoader

from models import *
from functions import *
from datasets import *
from utils import *

def get_args_parser():
    parser = argparse.ArgumentParser('Multiple Instance Learning for Histopathological Image Segmentation', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--model', default='SA_MIL', type=str)
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--data_path', default='../data/colon', type=str, help="The path of dataset")
    parser.add_argument('--work_path', default='./work', type=str, help="The path to save model")
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--device_ids', default=[0, 1], type=list)
    parser.add_argument('--r', default=4, type=int, help="The hyperparameter r in Generalized Mean function")
    parser.add_argument('--test_num_pos', default=80, type=int, help="The number of positive images in test dataset")
    parser.add_argument('--pretrain', action='store_true', help="Invoke the pretrain method of the model object")
    parser.add_argument('--train', action='store_true', help="Performing training process")
    parser.add_argument('--test', action='store_true', help="Performing testing process")
    parser.add_argument('--checkpoint', default=None, type=str, help="The filename of the checkpoint loaded during testing. If not provided, the best model will be used.")
    parser.add_argument('--save_all', action='store_true', help="Save all checkpoint during training")
    return parser

def main(args):
    print("Initializing......")
    work_path = args.work_path
    if not os.path.exists(work_path):
        os.makedirs(work_path)
    print("work path: " + work_path)
    print('data path: ' + args.data_path)
    model = eval(args.model)().to(torch.device(args.device))
    print('Model: ', args.model)

    if args.train:
        dataset_train = Dataset_train(args)
        dataset_valid = Dataset_valid(args)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=0)
        print('total train data: ', dataset_train.__len__())
        print('Batch Size: ', args.batch_size)
        train(model, dataloader_train, args, valid, dataloader_valid)
    if args.test:
        print("testing......")
        dataset_test = Dataset_test(args)
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
        test(model, dataloader_test, args)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
