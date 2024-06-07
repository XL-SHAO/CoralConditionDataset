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
import pandas as pd
from torchvision.models import swin_s, swin_b, Swin_S_Weights, Swin_B_Weights


# from utils_func.metrics import Evaluator
def load_checkpoint_for_evaluation(model, checkpoint):
    saved_state_dict = torch.load(checkpoint, map_location='cuda:0')
    model.load_state_dict(saved_state_dict)
    model.cuda()
    model.eval()


class Evaluator(object):
    def __init__(self, args):
        self.args = args

        # self.model_1 = EfficientNet.from_pretrained('efficientnet-b4')  # models.resnet34(pretrained=True)
        # num_ftrs = self.model_1._fc.in_features
        # self.model_1._fc = nn.Linear(num_ftrs, 8)
        # self.model_1.cuda()

        self.model_1 = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
        num_ftrs = self.model_1.head.in_features
        self.model_1.head = nn.Linear(num_ftrs, 8)
        self.model_1.cuda()

        self.model_2 = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        num_ftrs = self.model_2.head.in_features
        self.model_2.head = nn.Linear(num_ftrs, 8)
        self.model_2.cuda()

        self.model_3 = EfficientNet.from_pretrained('efficientnet-b7')
        num_ftrs = self.model_3._fc.in_features
        self.model_3._fc = nn.Linear(num_ftrs, 8)
        self.model_3.cuda()

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
    
    def per_class_entropy(self, probs):
        # Ensure we don't have any probabilities of exactly 0 or 1
        probs = torch.clamp(probs, min=1e-9, max=1-1e-9)
        # Calculate entropy for each class
        class_entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs))
        return class_entropy


    def validation(self):
        dataset_path = '/data/ggeoinfo/datasets/coral_reef_clf/winter_data_20240411'
        data_info_path = '/home/songjian/project/CRC/dataset/winter_data_test_set.json'
        dataset = CoralReefDataSet(dataset_path, data_info_path, max_iters=None, type='test')
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        val_data_loader = DataLoader(dataset, batch_size=8, num_workers=8, drop_last=False)

        all_predictions = []
        all_labels = []
        all_file_names = []
        with torch.no_grad():
            for _itera, data in enumerate(tqdm(val_data_loader)):
                img, labels, file_names = data
                labels = labels[:, 1:]

                img = img.cuda()
                outputs_1 = self.model_1(img)
                outputs_2 = self.model_2(img)
                outputs_3 = self.model_3(img)
                
                # Weighted sum of probabilities for each class
                prob = torch.sigmoid(outputs_1 + outputs_2 + outputs_3)
    

                # prob = torch.sigmoid(outputs)

                # Apply threshold to get binary predictions
                predictions = (prob > 0.5).int().cpu().numpy()

                all_predictions.extend(predictions)
                all_labels.extend(labels.int().cpu().numpy())

       
                all_file_names.extend(file_names)

        # Convert your results to a pandas DataFrame
        results_df = pd.DataFrame({
            'File Name': all_file_names,
            'Ground Truth': [' '.join(map(str, label)) for label in all_labels],
            'Prediction Result': [' '.join(map(str, pred)) for pred in all_predictions]
        })

        # Save to an Excel file
        results_df.to_excel('/home/songjian/project/CRC/winter_pred_results.xlsx', index=False)
        # F1 Score (average can be 'micro', 'macro', 'weighted', or 'samples')
        y_pred = np.array(all_predictions)
        y_true = np.array(all_labels)
        each_class_f1 = f1_score(y_true, y_pred, average=None)
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        # Exact Match Ratio (Subset Accuracy)
        exact_match_ratio = accuracy_score(y_true, y_pred)
        print(f'match ratio is {exact_match_ratio}, '
              f'micro f1 score is {micro_f1}, '
              f'macro f1 score is {macro_f1}, '
              f'each class f1 score is {each_class_f1}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coral Reef DataLoader Test")
    parser.add_argument('--dataset', type=str, default='coral_reef')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='/home/songjian/project/datasets/CoralReef')
    parser.add_argument('--restore_from', type=str,
                        default=[
                            '/home/songjian/project/CRC/saved_model/SwinTransformer/SwinS/24000_model.pth',
                            '/home/songjian/project/CRC/saved_model/SwinTransformer/SwinB/23000_model.pth',
                            '/home/songjian/project/CRC/saved_model/EfficientNet/EfficientNetB7/20000_model.pth'])

    parser.add_argument('--shuffle', type=bool, default=True)

    args = parser.parse_args()

    evaler = Evaluator(args)
    evaler.validation()
