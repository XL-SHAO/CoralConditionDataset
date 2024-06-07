import sys
sys.path.append('/home/songjian/project/CRC')
import argparse

import torch.nn as nn
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

from torchvision.models import swin_s, swin_b, Swin_S_Weights, Swin_B_Weights


# from utils_func.metrics import Evaluator
def load_checkpoint_for_evaluation(model, checkpoint):
    saved_state_dict = torch.load(checkpoint, map_location='cuda:0')
    model.load_state_dict(saved_state_dict)
    model.cuda()
    model.eval()


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # self.model_1 = EfficientNet.from_pretrained('efficientnet-b4')  # models.resnet34(pretrained=True)
        # num_ftrs = self.model_1._fc.in_features
        # self.model_1._fc = nn.Linear(num_ftrs, 8)
        # self.model_1.cuda()
        self.train_data_loader = make_data_loader(args)

        self.model_1 = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
        num_ftrs = self.model_1.head.in_features
        self.model_1.head = nn.Linear(num_ftrs, 8)
        self.model_1.cuda()

        self.model_2 = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        num_ftrs = self.model_2.head.in_features
        self.model_2.head = nn.Linear(num_ftrs, 8)
        self.model_2.cuda()

        self.model_3 =  EfficientNet.from_pretrained('efficientnet-b6')
        num_ftrs = self.model_3._fc.in_features
        self.model_3._fc = nn.Linear(num_ftrs, 8)
        self.model_3.cuda()

        self.weights = nn.Parameter(torch.ones(3) / 3)  # Start with equal weights, normalized
        self.weights.requires_grad = True
        self.weights.cuda()

        self.optimizer = AdamW([self.weights], lr=1e-4, weight_decay=5e-4)
        self.loss_fn = binary_cross_entropy_with_logits

        if os.path.exists(args.restore_from[0]):
            load_checkpoint_for_evaluation(self.model_1, args.restore_from[0])
            print('model 1 has been loaded')
        else:
            exit('no such model 1')

        if os.path.exists(args.restore_from[1]):
            load_checkpoint_for_evaluation(self.model_2, args.restore_from[1])
            print('model 2 has been loaded')
        else:
            exit('no such model 2')

        if os.path.exists(args.restore_from[2]):
            load_checkpoint_for_evaluation(self.model_3, args.restore_from[2])
            print('model 3 has been loaded')
        else:
            exit('no such model 3')

        for model in [self.model_1, self.model_2, self.model_3]:
            for param in model.parameters():
                param.requires_grad = False

        
    def validation(self):
        dataset_path = '/data/ggeoinfo/datasets/coral_reef_clf/winter_data_20240411'
        data_info_path = '/home/songjian/project/CRC/dataset/winter_data_test_set.json'
        dataset = CoralReefDataSet(dataset_path, data_info_path, max_iters=None, type='test')
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        val_data_loader = DataLoader(dataset, batch_size=8, num_workers=8, drop_last=False)

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for _itera, data in enumerate(val_data_loader):
                img, labels, _ = data
                labels = labels[:, 1:]

                img = img.cuda()
                outputs_1 = self.model_1(img)
                outputs_2 = self.model_2(img)
                outputs_3 = self.model_3(img)
                normalized_weights = torch.nn.functional.softmax(self.weights, dim=0)
                outputs = (normalized_weights[0] * outputs_1 +
                                normalized_weights[1] * outputs_2 +
                                normalized_weights[2] * outputs_3)

                prob = torch.sigmoid(outputs)

                # Apply threshold to get binary predictions
                predictions = (prob > 0.5).int().cpu().numpy()

                all_predictions.extend(predictions)
                all_labels.extend(labels.int().cpu().numpy())

        y_pred = np.array(all_predictions)
        y_true = np.array(all_labels)
        # F1 Score (average can be 'micro', 'macro', 'weighted', or 'samples')
        each_class_f1 = f1_score(y_true, y_pred, average=None)
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        # Exact Match Ratio (Subset Accuracy)
        exact_match_ratio = accuracy_score(y_true, y_pred)
        return each_class_f1, micro_f1, macro_f1, exact_match_ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coral Reef DataLoader Test")
    parser.add_argument('--dataset', type=str, default='coral_reef')
    parser.add_argument('--max_iters', type=int, default=50000)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='/home/songjian/project/datasets/CoralReef')
    parser.add_argument('--data_info_path', type=str,
                        default='/home/songjian/project/datasets/CoralReef/train_set_fully_supervision.json')
    parser.add_argument('--restore_from', type=str,
                        default=[
                            '/home/songjian/project/CRC/saved_model/SwinTransformer/SwinS/24000_model.pth',
                            '/home/songjian/project/CRC/saved_model/SwinTransformer/SwinB/23000_model.pth',
                            '/home/songjian/project/CRC/saved_model/EfficientNet/EfficientNetB6/20000_model.pth'])

    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    evaler = Trainer(args)
    evaler.validation()
