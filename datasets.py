import torch
import os
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import Dataset

"""
The dataset should be organized into the following structure when --data_path = .../dataset.
.../dataset/train/pos
.../dataset/train/neg
.../dataset/test/pos
.../dataset/test/neg
.../dataset/test/gt
"""

class Dataset_train(Dataset):
    def __init__(self, args):
        super(Dataset_train, self).__init__()
        self.path_pos = os.path.join(args.data_path, 'train/pos')
        self.path_neg = os.path.join(args.data_path, 'train/neg')
        self.list_pos = os.listdir(self.path_pos)
        self.list_neg = os.listdir(self.path_neg)
        self.list_pos.sort()
        self.list_neg.sort()
        self.num_pos = len(self.list_pos)
        self.num_neg = len(self.list_neg)
        self.size = args.input_size
        self.transforms = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=[164.7261, 129.4018, 176.4253], std=[43.0450, 49.9314, 32.2143])
        ])

    def __getitem__(self, index):
        if index < self.num_pos:
            image = self.read(self.path_pos, self.list_pos[index])
            label = torch.ones(1)
        else:
            image = self.read(self.path_neg, self.list_neg[index - self.num_pos])
            label = torch.zeros(1)
        return image, label

    def __len__(self):
        return self.num_pos + self.num_neg

    def read(self, path, name):
        img = io.imread(os.path.join(path, name))[:,:,0:3]
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        img = self.transforms(img)
        return img


class Dataset_valid(Dataset):
    def __init__(self, args):
        super(Dataset_valid, self).__init__()
        self.path_pos = os.path.join(args.data_path, 'test/pos')
        self.path_neg = os.path.join(args.data_path, 'test/neg')
        self.path_gdt = os.path.join(args.data_path, 'test/gt')
        self.list_pos = os.listdir(self.path_pos)
        self.list_neg = os.listdir(self.path_neg)
        self.list_gdt = os.listdir(self.path_gdt)
        self.list_pos.sort()
        self.list_neg.sort()
        self.list_gdt.sort()
        self.num_pos = len(self.list_pos)
        self.num_neg = len(self.list_neg)
        self.size = args.input_size
        self.transforms_test = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.Normalize(mean=[164.7261, 129.4018, 176.4253], std=[43.0450, 49.9314, 32.2143])
        ])
        self.transforms_grdth = transforms.Compose([
            transforms.Resize(self.size)
                        ])

    def __getitem__(self, index):
        if index < self.num_pos:
            image = self.read(self.path_pos, self.list_pos[index])
            grdth = self.read(self.path_gdt, self.list_gdt[index], False)
        else:
            image = self.read(self.path_neg, self.list_neg[index-self.num_pos])
            grdth = torch.zeros(1, self.size, self.size)
        return image, grdth

    def __len__(self):
        return self.num_pos + self.num_neg

    def read(self, path, name, isRGB=True):
        img = io.imread(os.path.join(path, name))
        if isRGB:
            img = torch.from_numpy(img).float().permute(2, 0, 1)
            img = self.transforms_test(img)
        else:
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = torch.from_numpy(img).float().unsqueeze(0)
            img = self.transforms_grdth(img)
            img = (img > 0) + 0
        return img

class Dataset_test(Dataset):
    def __init__(self, args):
        self.path_pos = os.path.join(args.data_path, 'test/pos')
        self.path_neg = os.path.join(args.data_path, 'test/neg')
        self.path_gdt = os.path.join(args.data_path, 'test/gt')
        self.list_pos = os.listdir(self.path_pos)
        self.list_neg = os.listdir(self.path_neg)
        self.list_gdt = os.listdir(self.path_gdt)
        self.list_pos.sort()
        self.list_neg.sort()
        self.list_gdt.sort()
        self.num_pos = len(self.list_pos)
        self.num_neg = len(self.list_neg)
        self.size = args.input_size
        self.transforms_test = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.Normalize(mean=[164.7261, 129.4018, 176.4253], std=[43.0450, 49.9314, 32.2143])
        ])
        self.transforms_grdth = transforms.Compose([
            transforms.Resize(self.size)
        ])

    def __getitem__(self, index):
        if index < self.num_pos:
            image = self.read(self.path_pos, self.list_pos[index], 'rgb')
            label = self.read(self.path_gdt, self.list_gdt[index], 'bln')
            image_show = self.read(self.path_pos, self.list_pos[index])
            name = self.list_pos[index]
        else:
            image = self.read(self.path_neg, self.list_neg[index-self.num_pos], 'rgb')
            label = torch.zeros(1, self.size, self.size)
            image_show = self.read(self.path_neg, self.list_neg[index-self.num_pos])
            name = self.list_neg[index-self.num_pos]
        return image, label, image_show, name

    def __len__(self):
        return self.num_pos + self.num_neg

    def read(self, path, name, mode=None):
        img = io.imread(os.path.join(path, name))
        if mode == 'rgb':
            img = torch.from_numpy(img).float().permute(2, 0, 1)
            img = self.transforms_test(img)
            return img
        if mode == 'bln':
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = torch.from_numpy(img).float().unsqueeze(0)
            img = self.transforms_grdth(img)
            img = (img > 0) + 0
            return img
        return img

