import os.path as osp
import torch
from torchvision.models import resnet18, ResNet18_Weights, vgg11_bn, VGG11_BN_Weights

here = osp.abspath(f'{__file__}/..')

models = (resnet18, vgg11_bn)
params = (ResNet18_Weights.IMAGENET1K_V1, VGG11_BN_Weights.IMAGENET1K_V1)

for model, param in zip(models, params):
    model_name = model.__name__
    with open(osp.join(here, f'{model_name}.py'), 'w+') as file:
        file.writelines([
            f'from torchvision.models import {model_name}\n',
            '\n',
            'def get_model():\n',
            f'\treturn {model_name}()\n',
            '\n',
        ])

    model = model(weights=param)
    state_dict = model.state_dict()
    torch.save(state_dict, osp.join(here, f'{model_name}_weights.pt'))
    
    

