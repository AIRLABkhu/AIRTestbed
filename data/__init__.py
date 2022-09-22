from os.path import abspath

path = abspath(f'{__file__}/../')


class Names:
    imagenet = 'imagenet'
    cifar10 = 'cifar10'

name_root_map = {
    Names.imagenet: f'{path}/archive/imagenet2012',
    Names.cifar10: f'{path}/archive',
}

name_norm_map = {
    Names.imagenet: ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    Names.cifar10: ((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
}
