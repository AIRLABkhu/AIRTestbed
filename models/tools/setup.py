import os.path as osp
import torch
from torchvision import models as torchmodels

here = osp.abspath(f'{__file__}/../..')

models = (
    torchmodels.resnet18, 
    torchmodels.resnet152, 

    torchmodels.vgg11,
    torchmodels.vgg16,
    torchmodels.vgg19,

    torchmodels.vgg11_bn,
    torchmodels.vgg16_bn,
    torchmodels.vgg19_bn,

    torchmodels.inception_v3,
)
params = (
    torchmodels.ResNet18_Weights.IMAGENET1K_V1,
    torchmodels.ResNet152_Weights.IMAGENET1K_V2,

    torchmodels.VGG11_Weights.IMAGENET1K_V1,
    torchmodels.VGG16_Weights.IMAGENET1K_V1,
    torchmodels.VGG19_Weights.IMAGENET1K_V1,

    torchmodels.VGG11_BN_Weights.IMAGENET1K_V1,
    torchmodels.VGG16_BN_Weights.IMAGENET1K_V1,
    torchmodels.VGG19_BN_Weights.IMAGENET1K_V1,

    torchmodels.Inception_V3_Weights.IMAGENET1K_V1,
)

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
