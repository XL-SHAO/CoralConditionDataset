import sys
sys.path.append('/home/songjian/project/CRC')
import argparse

import torch.nn as nn
import torchvision.models as models
from torch.optim import AdamW
from dataset.make_dataloader import make_data_loader, CoralReefDataSet
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
from tqdm import tqdm
from deep_model.efficientnet import EfficientNet
import os
from deep_model.SegFormer.SegFormer import WeTr
import time


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.model = models.resnet152(pretrained=True)
        num_ftrs = self.model.fc.in_features

        # Replace the final fully connected layer
        # Here, 8 is the number of output classes
        self.model.fc = nn.Linear(num_ftrs, 8)
        self.model.cuda()

        self.optimizer = AdamW(self.model.parameters(), lr=1e-4, weight_decay=5e-4)
        self.loss_fn = binary_cross_entropy_with_logits

        self.train_data_loader = make_data_loader(args)
        self.model_save_path = os.path.join(args.model_save_path, str(time.time()))
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def train(self):
        best_info = []
        best_match_ratio = 0
        # Number of input features to the final fully connected layer
        self.model.train()
        for _iter, data in enumerate(tqdm(self.train_data_loader)):
            img, labels, _ = data

            img = img.cuda()
            labels = labels.cuda()[:, 1:]

            self.optimizer.zero_grad()
            outputs = self.model(img)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if (_iter + 1) % 1000 == 0:
                each_class_f1, micro_f1, macro_f1, exact_match_ratio = self.validation()
                if exact_match_ratio > best_match_ratio:
                    best_match_ratio = exact_match_ratio
                    best_info = [_iter + 1, loss, exact_match_ratio, micro_f1, macro_f1, each_class_f1]
                    torch.save(self.model.state_dict(),
                               os.path.join(self.model_save_path, f'{_iter + 1}_model.pth'))
                print(f'iteration is {_iter + 1}, '
                      f'loss is {loss}, '
                      f'match ratio is {exact_match_ratio}, '
                      f'micro f1 score is {micro_f1}, '
                      f'macro f1 score is {macro_f1}, '
                      f'each class f1 score is {each_class_f1}')
                self.model.train()

        print(f'best iteration is {best_info[0]}, '
              f'loss is {best_info[1]}, '
              f'match ratio is {best_info[2]}, '
              f'micro f1 score is {best_info[3]}, '
              f'macro f1 score is {best_info[4]}, '
              f'each class f1 score is {best_info[5]}')

    def validation(self):
        self.model.eval()
        dataset_path = '/home/songjian/project/datasets/CoralReef'
        data_info_path = '/home/songjian/project/datasets/CoralReef/test_set.json'
        dataset = CoralReefDataSet(dataset_path, data_info_path, max_iters=None, type='test')
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        val_data_loader = DataLoader(dataset, batch_size=16, num_workers=16, drop_last=False)

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for _itera, data in enumerate(val_data_loader):
                img, labels, _ = data
                labels = labels[:, 1:]

                img = img.cuda()
                outputs = self.model(img)
                prob = torch.sigmoid(outputs)

                # Apply threshold to get binary predictions
                predictions = (prob > 0.5).int().cpu().numpy()

                all_predictions.extend(predictions)
                all_labels.extend(labels.int().cpu().numpy())

        y_pred = np.array(all_predictions)
        y_true = np.array(all_labels)
        # F1 Score (average can be 'micro', 'macro', 'weighted', or 'samples')
        each_class_f1 = f1_score(y_true, y_pred, average=None)
        microf1 = f1_score(y_true, y_pred, average='micro')
        macrof1 = f1_score(y_true, y_pred, average='macro')
        # Exact Match Ratio (Subset Accuracy)
        exact_match_ratio = accuracy_score(y_true, y_pred)
        return each_class_f1, microf1, macrof1, exact_match_ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coral Reef DataLoader Test")
    parser.add_argument('--dataset', type=str, default='coral_reef')
    parser.add_argument('--max_iters', type=int, default=200000)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='/home/songjian/project/datasets/CoralReef')
    parser.add_argument('--data_info_path', type=str,
                        default='/home/songjian/project/datasets/CoralReef/train_set_fully_supervision.json')
    parser.add_argument('--model_save_path', type=str,
                        default='/home/songjian/project/CRC/saved_model/ResNet18')

    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
