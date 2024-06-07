import sys
sys.path.append('/home/songjian/project/CRC')
import argparse

import torch.nn as nn
from torch.optim import AdamW
from datasets.make_dataloader import make_data_loader, CoralReefDataSet
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
from torchvision.models import swin_t, Swin_T_Weights
from deep_model.discriminator import Discriminator


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.head.in_features
        self.domain_discriminator = Discriminator(num_ftrs)

        # Replace the final fully connected layer
        # Here, 8 is the number of output classes
        self.model.head = nn.Linear(num_ftrs, 8)
        self.model.cuda()
        self.domain_discriminator.cuda()

        self.optimizer_G = AdamW(self.model.parameters(), lr=1e-4, weight_decay=5e-3)
        self.optimizer_D = AdamW(self.domain_discriminator.parameters(), lr=2e-4, weight_decay=1e-4)

        self.loss_fn = binary_cross_entropy_with_logits

        self.source_data_loader = make_data_loader(args)
        self.model_save_path = os.path.join(args.model_save_path, str(time.time()))
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)


    def train(self):
        best_info = []
        best_match_ratio = 0
        # Number of input features to the final fully connected layer
        source_dataset_path = '/data/ggeoinfo/datasets/coral_reef_clf/summer_data_20231004'
        source_data_info_path = '/home/songjian/project/CRC/datasets/train_set_fully_supervision.json'
        source_dataset = CoralReefDataSet(source_dataset_path, source_data_info_path, max_iters=self.args.max_iters, type='train')
        source_data_loader = DataLoader(source_dataset, batch_size=self.args.batch_size, num_workers=16, drop_last=False)

        target_dataset_path = '/data/ggeoinfo/datasets/coral_reef_clf/winter_data_20240411'
        target_data_info_path = '/home/songjian/project/CRC/datasets/winter_data_test_set.json'
        target_dataset = CoralReefDataSet(target_dataset_path, target_data_info_path, max_iters=self.args.max_iters, type='train')
        target_data_loader = DataLoader(target_dataset, batch_size=self.args.batch_size, num_workers=16, drop_last=False)

        self.model.train()
        self.domain_discriminator.train()
        for _iter, (source_data, target_data) in enumerate(zip(tqdm(source_data_loader), target_data_loader)):
            source_img, source_labels, _ = source_data
            target_img, _, _ = target_data  # Target labels might not be used depending on your UDA setup

            source_img, source_labels = source_img.cuda(), source_labels.cuda()[:, 1:]
            target_img = target_img.cuda()

           # Discriminator training
            self.optimizer_D.zero_grad()
            src_features = self.model.features(source_img).permute([0, 3, 1, 2])
            tgt_features = self.model.features(target_img).permute([0, 3, 1, 2])
           
            src_domain_preds = self.domain_discriminator(src_features.detach())
            tgt_domain_preds = self.domain_discriminator(tgt_features.detach())

            D_loss = self.loss_fn(src_domain_preds, torch.zeros_like(src_domain_preds)) + \
                     self.loss_fn(tgt_domain_preds, torch.ones_like(tgt_domain_preds))
            D_loss.backward()
            self.optimizer_D.step()

            # Classifier and adversarial training to fool discriminator
            self.optimizer_G.zero_grad()

            classifier_loss = self.loss_fn(self.model(source_img), source_labels)
            
            entropy_loss = self.batch_entropy_loss(torch.sigmoid(self.model(target_img)))
            
            fool_loss = self.loss_fn(self.domain_discriminator(src_features), torch.ones_like(src_domain_preds)) \
                        + self.loss_fn(self.domain_discriminator(tgt_features), torch.zeros_like(tgt_domain_preds))

            G_total_loss = classifier_loss + 0.3 * fool_loss + entropy_loss

            G_total_loss.backward()
            self.optimizer_G.step()

            if (_iter + 1) % 500 == 0:
                self.model.eval()

                S_each_class_f1, S_micro_f1, S_macro_f1, S_exact_match_ratio = self.source_validation()
                T_each_class_f1, T_micro_f1, T_macro_f1, T_exact_match_ratio = self.target_validation()

                if T_exact_match_ratio > best_match_ratio:
                    best_match_ratio = T_exact_match_ratio
                    best_info = [_iter + 1, G_total_loss, D_loss, S_exact_match_ratio, S_micro_f1, S_macro_f1, T_exact_match_ratio, T_micro_f1, T_macro_f1]
                    torch.save(self.model.state_dict(),
                               os.path.join(self.model_save_path, f'G_{_iter + 1}_model.pth'))
                    torch.save(self.domain_discriminator.state_dict(),
                               os.path.join(self.model_save_path, f'D_{_iter + 1}_model.pth'))
                    
                print(f'iteration is {_iter + 1}, '
                      f'G loss is {classifier_loss} + {fool_loss} + {entropy_loss}, '
                      f'd loss is {D_loss}, '
                      f'match ratio S/T is {S_exact_match_ratio} / {T_exact_match_ratio}, '
                      f'micro f1 score S/T is {S_micro_f1} / {T_micro_f1}, '
                      f'macro f1 score S/T is {S_macro_f1} / {T_macro_f1}, '
                      f'each class f1 score S/T is {S_each_class_f1} / {T_each_class_f1}')
                self.model.train()


        print(f'best iteration is {best_info[0]}, '
              f'loss is {best_info[1]} / {best_info[2]}, '
              f'match ratio is {best_info[3]} / {best_info[6]}, '
              f'micro f1 score is {best_info[4]} / {best_info[7]}, '
              f'macro f1 score is {best_info[5]} / {best_info[8]}, '
              f'each class f1 score is {S_each_class_f1} / {T_each_class_f1}')

    def batch_entropy_loss(self, probabilities):
        # Calculate entropy for probabilities
        epsilon = 1e-10
        entropy = -(probabilities * torch.log(probabilities + epsilon) + (1 - probabilities) * torch.log(1 - probabilities + epsilon))
        # Average entropy across the batch and sum over classes
        batch_entropy = entropy.mean(dim=0).mean()
        return batch_entropy
    
    def source_validation(self):
        dataset_path = '/data/ggeoinfo/datasets/coral_reef_clf/summer_data_20231004'
        data_info_path = '/home/songjian/project/CRC/datasets/test_set.json'
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


    def target_validation(self):
        dataset_path = '/data/ggeoinfo/datasets/coral_reef_clf/winter_data_20240411'
        data_info_path = '/home/songjian/project/CRC/datasets/winter_data_test_set.json'
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
    parser.add_argument('--max_iters', type=int, default=320000)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='/home/songjian/project/datasets/CoralReef')
    parser.add_argument('--data_info_path', type=str,
                        default='/home/songjian/project/datasets/CoralReef/train_set_fully_supervision.json')
    parser.add_argument('--model_save_path', type=str,
                        default='/home/songjian/project/CRC/saved_model/SwinTransformer')

    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
