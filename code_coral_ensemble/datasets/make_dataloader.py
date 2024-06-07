import os

import numpy as np
from torch.utils.data import Dataset, DataLoader
import imageio
import datasets.imutils as imutils
import torch
import json
import argparse

def img_loader(path):
    img = imageio.imread(path).astype(np.float32)
    return img


class CoralReefDataSet(Dataset):
    def __init__(self, dataset_path, data_info_path, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        with open(data_info_path, 'r') as f:
            self.data_list = json.load(f)

        self.loader = data_loader
        self.type = type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]

    def __transforms(self, aug, img):
        if aug:
            img = imutils.random_fliplr(img)
            img = imutils.random_flipud(img)
            img = imutils.random_rot(img)
            # image = self.color_jittor(image)

        img = imutils.normalize_img(img)  # imagenet normalization
        img = np.transpose(img, (2, 0, 1))  # pytorch requires channel, head, weight
        return img

    def __getitem__(self, index):
        entry = self.data_list[index]

        img_path = os.path.join(self.dataset_path, 'img', entry['tileid'] + '.jpg')

        img = self.loader(img_path)
        # img = cv.resize(img, dsize=(224, 224), interpolation=cv.INTER_AREA) # For ViT training

        labels = [value for key, value in entry.items() if key != 'tileid']
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.type == 'train':
            img = self.__transforms(True, img)
        else:
            img = self.__transforms(False, img)
        return img, labels, entry['tileid']

    def __len__(self):
        return len(self.data_list)


def make_data_loader(
        args):  # **kwargs was the second argument and was omitted (to be tested) note that it was also 4th argument of the
    # DataLoader and it was omitted (to be tested)
    if args.dataset == 'coral_reef':
        # Creating a torch.utils.data.Dataset ready for the torch.utils.data.DataLoader
        dataset = CoralReefDataSet(args.dataset_path, args.data_info_path, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8,
                                 drop_last=False)
        # set num_workers to 4 because of potential slowing down with 16 on CERN GPU
        # drop_last (bool, optional) – set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
        # If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: False)
        return data_loader

    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coral Reef DataLoader Test")
    parser.add_argument('--dataset', type=str, default='coral_reef')
    parser.add_argument('--max_iters', type=int)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='D:/Workspace/Python/coralReef/data/test_img')
    parser.add_argument('--data_info_path', type=str,
                        default='D:/Workspace/Python/coralReef/data/test_img/train_set.json')
    parser.add_argument('--shuffle', type=bool,
                        default=True)  # shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_name_list', type=list)
    args = parser.parse_args()
    # Reading from data_list_path which is set as default to './xBD_list/train.txt'
    train_data_loader = make_data_loader(args)
    for i, data in enumerate(train_data_loader):
        img, label, _ = data
        print(i, "个inputs", img.data.size())
