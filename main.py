import os, sys
from traceback import print_exc
from typing import Callable, Iterable
from argparse import ArgumentParser
from importlib import machinery

import torch
from torch import nn

from classification_evaluator import ClassificationEvaluator

split = ','
parser = ArgumentParser(description='Classification Testbed')
parser.add_argument('--device', '-d', type=str, default=None, help='cpu or cuda or cuda:0. Default: cuda if supported, cpu otherwise.')
parser.add_argument('--models', '-m', type=str, help=f'The filename of the model or a list of names splitted with \'{split}\'. ex) resnet18.py or resnet18.py{split}vgg16.py')
parser.add_argument('--weights', '-w', type=str, help=f'The filename of the parameteres or a list of names splitted with \'{split}\'. ex) resnet18_weights.pt or resnet18_weights.pt{split}vgg16_weights.pt')
parser.add_argument('--datasets', '-ds', type=str, help=f'The name of a dataset or a list of names splitted with \'{split}\'. ex) imagenet or imagenet{split}cifar10')
parser.add_argument('--input-size', '-is', type=str, help='The size of the input image. ex) 224x224')
parser.add_argument('--normalize-input', '-ni', type=bool, default=True, help='Specifies whether to normalize the input or not.')
parser.add_argument('--flatten-input', '-fi', type=bool, default=False, help='Specifies whether to flatten the input or not.')
parser.add_argument('--universal-adversarial-perturbation', '-p', type=str, default=None, help='The file name of a perturbation image.')
parser.add_argument('--perturbation-magnitude', '-pm', type=float, default=10, help='The maximum inf-norm value of the perturbation image in [0. 255].')
parser.add_argument('--batch-size', '-b', type=int, default=32, help='The size of a batch on evaluation.')
parser.add_argument('--description', '-D', type=str, default=None, help='The description string for this experiment.')
parser.add_argument('--metrics', '-e', type=str, default='all', help=f'The name of a metric or a list of names splitted with \'{split}\'. ex) accuracy or accuracy{split}mean_cls_accuracy')
parser.add_argument('--outputfile', '-o', type=str, default=None, help='The CSV filename to write the result.')
parser.add_argument('--logfile', '-l', type=str, default=None, help='The filename to write the evaluation log.')
parser.add_argument('--verbose', '-v', type=bool, default=True, help='The log will be display on the terminal if true.')
parser.add_argument('--tqdm', '-t', type=bool, default=True, help='The progress bar will be displayed on the terminal if true.')
parser.add_argument('--subsample', '-ss', type=float, default=1, help='Sample subset of specified ratio from full dataset.')

args = parser.parse_args()
DEVICE = args.device
MODELS = args.models.split(split)
WEIGHTS = args.weights.split(split)
DATASETS = args.datasets.split(split)
INPUT_SIZE = tuple(int(x) for x in args.input_size.split('x'))
if not isinstance(INPUT_SIZE, Iterable) or len(INPUT_SIZE) != 2:
    raise TypeError('Input size must be a 2-tuple of integers.')
NORMALIZE_INPUT = args.normalize_input
FLATTEN_INPUT = args.flatten_input
if args.universal_adversarial_perturbation is not None:
    if os.path.exists(args.universal_adversarial_perturbation):
        PERTURBATION = torch.load(args.universal_adversarial_perturbation)
        if PERTURBATION.ndim == 4:
            PERTURBATION = PERTURBATION[0]
    else:
        raise FileNotFoundError('Cannot find the specified perturbation image file.')
    PERTURBATION_MAGNITUDE = args.perturbation_magnitude
else:
    PERTURBATION_MAGNITUDE = 10
    PERTURBATION = None
BATCH_SIZE = args.batch_size
DESCRIPTION = args.description
METRICS = args.metrics
if METRICS == 'all':
    METRICS = 'accuracy,mean_cls_accuracy,mean_cross_entropy'
LOG_FILENAME = args.logfile
OUT_FILENAME = args.outputfile
VERBOSE = args.verbose
SUBSAMPLE_RATE = args.subsample


def load_model(model_filename, weights_filename):
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f'No such file or directory: \'{model_filename}\'')
    if not os.path.exists(weights_filename):
        raise FileNotFoundError(f'No such file or directory: \'{weights_filename}\'')
    module_name = model_filename.split(os.sep)[-1].replace('.py', '')

    loader = machinery.SourceFileLoader(module_name, model_filename)
    module = loader.load_module()
    if 'get_model' in module.__dict__ and isinstance(module.__dict__['get_model'], Callable):
        model = module.get_model()
        if not isinstance(model, nn.Module):
            raise TypeError(f'The {module_name}.get_model() doesn\'t return nn.Module instance.')
    else:
        raise NotImplementedError('A module for model provide must implement global get_module() function.')

    state_dict = torch.load(weights_filename)
    try:
        model.load_state_dict(state_dict, strict=True)
    except:
        raise RuntimeError('Cannot apply provided parameters to the specific model.')
    
    return model


log_streams = []
log_file = None
out_file = None
if VERBOSE:
    log_streams.append(sys.stdout)
if LOG_FILENAME:
    log_file = open(LOG_FILENAME, 'w+', encoding='utf8')
    if DESCRIPTION:
        log_file.write(DESCRIPTION + '\n')
    log_streams.append(log_file)
if OUT_FILENAME:
    out_file = open(OUT_FILENAME, 'w+', encoding='utf8')
    out_file.write(',,' + ','.join(METRICS.split(',')) + '\n')
    print(METRICS)

try:
    for dataname in DATASETS:
        for i, (model_filename, weights_filename) in enumerate(zip(MODELS, WEIGHTS)):
            model_name = model_filename.split(os.sep)[-1].replace('.py', '')
            model = load_model(model_filename, weights_filename)

            evaluator = ClassificationEvaluator(model, dataname, INPUT_SIZE, NORMALIZE_INPUT, FLATTEN_INPUT, PERTURBATION, PERTURBATION_MAGNITUDE, SUBSAMPLE_RATE)
            result = evaluator.evaluate(batch_size=BATCH_SIZE, metrics=METRICS, device=DEVICE, log_streams=log_streams)

            first_column = dataname if i == 0 else ''
            scores = [str(v) for v in result.values()]
            line = ','.join([first_column, model_name] + scores) + '\n'
            out_file.writelines(line)

            for stream in log_streams:
                print(file=stream)
except Exception:
    print_exc(log_file)
finally:
    for file in (log_file, out_file):
        if file:
            file.flush()
            file.close()
