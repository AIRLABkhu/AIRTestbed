import sys
from typing import Iterable
from tqdm.auto import tqdm
from datetime import datetime
from time import time

import torch
from torch import nn
from torch.utils.data import Subset, DataLoader

from torchvision import datasets, transforms

from data import *


class ClassificationEvaluator:
    def __init__(self, model, dataset, input_size=None, normalize_input=True, flatten_input=False, subsample_rate=1.0):
        self.model = model

        self.dataset_name = dataset
        self.input_size = input_size
        self.normalize_input = normalize_input
        self.flatten_input = flatten_input
        self.subsample_rate = subsample_rate

        if self.dataset_name == Names.imagenet:
            self.dataset = datasets.ImageNet(name_root_map[self.dataset_name], split='val')
            self.num_classes = 1000
        elif self.dataset_name == Names.cifar10:
            self.dataset = datasets.CIFAR10(name_root_map[self.dataset_name], train=False)
            self.num_classes = 10
        else:
            names = ', '.join(name_root_map.keys())
            raise NotImplementedError(f'{self.dataset_name} is not implemented. Please select one of {names}.')

        transform = [transforms.ToTensor()]
        if normalize_input:
            mean, std = name_norm_map[self.dataset_name]
            transform.append(transforms.Normalize(mean, std))
        if input_size:
            transform.append(transforms.Resize(size=input_size))
        if flatten_input:
            transform.append(transforms.Lambda(lambda x: x.flatten(start_dim=1)))

        self.dataset.transform = transforms.Compose(transform)
        if subsample_rate != 1:
            num_samples = len(self.dataset)
            indices = torch.linspace(0, num_samples - 1, int(num_samples * subsample_rate)).int()
            self.dataset = Subset(self.dataset, indices=indices)

    @torch.no_grad()
    def evaluate(
        self, batch_size=32, postfix=None, 
        metrics='all', 
        desc=None, device=None, log_streams=[sys.stdout], use_tqdm=True):

        postfix = f'-{postfix}' if postfix else ''
        if not device:
            device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        if metrics == 'all':
            metrics = 'accuracy,mean_cls_accuracy,mean_cross_entropy'
        metrics = metrics.split(',')

        def _print(*args, sep=' ', end='\n'):
            for stream in log_streams:
                print(*args, sep=sep, end=end, file=stream)

        def _timestamp():
            return datetime.now().strftime("%Y%m%d_%H%M%S")

        _print('Starting evaluation:', f'{_timestamp()}{postfix}')
        if desc:
            _print('    Description:', desc)
        _print('    Device:', device)
        _print('    Model:', type(self.model))
        _print('    Dataset:', self.dataset_name)
        _print('        # of samples:', len(self.dataset))
        _print('        Batch size:', batch_size)
        _print('        # of classes:', self.num_classes)
        _print('        Normalized:', self.normalize_input)
        _print('        Flattened:', self.flatten_input)
        _print('    Metrics:')
        for metric in metrics:
            _print('        ', metric, sep='')

        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        if use_tqdm:
            dataloader = tqdm(dataloader)

        total_num_correct = 0
        total_num_correct_per_class = torch.zeros(self.num_classes).to(device)
        total_num_samples_per_class = torch.zeros(self.num_classes).to(device)
        total_ce = 0
        
        started_time = time()
        self.model.eval().to(device)
        for img, gt_label in dataloader:
            img = img.to(device)
            gt_onehot = nn.functional.one_hot(gt_label, self.num_classes).to(device)
            gt_label = gt_label.to(device)

            out_score = self.model(img)
            out_label = out_score.argmax(dim=1)

            match_map = gt_label == out_label
            num_correct = torch.sum(match_map)
            total_num_correct += num_correct.item()

            class_match_map = gt_onehot * match_map.view(-1, 1)
            total_num_correct_per_class += torch.sum(class_match_map, dim=0)
            total_num_samples_per_class += torch.sum(gt_onehot, dim=0)

            ce = nn.functional.cross_entropy(out_score, gt_onehot.float(), reduction='sum')
            total_ce += ce.item()

        num_samples = len(self.dataset)
        result = {
            'accuracy': total_num_correct / num_samples,
            'mean_cls_accuracy': (total_num_correct_per_class / total_num_samples_per_class).mean().item(),
            'mean_cross_entropy': total_ce / num_samples,
        }
        done_time = time()
        filtered_result = {key: value for key, value in result.items() if key in metrics}

        _print('Evaluation complete:', _timestamp())
        _print('    Time elapsed: %.4fs' % (done_time - started_time))
        _print('    Result:')
        for key, value in filtered_result.items():
            if isinstance(value, Iterable):
                value = sum(value) / len(value)
            _print('        %s: %.6f' % (key, value))

        return filtered_result
            

if __name__ == '__main__':
    from torchvision.models import resnet18, ResNet18_Weights
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    evaluator = ClassificationEvaluator(net, Names.imagenet, input_size=(224, 224))
    result = evaluator.evaluate(postfix='first_trial', device='cuda:0')

    print()
    print(result)