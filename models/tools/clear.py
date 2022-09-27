import os
from tqdm.auto import tqdm

here = os.path.abspath(f'{__file__}/../..')

files = set(file 
    for file in os.listdir(here) 
    if os.path.isfile(file)) - { 'setup.py', 'clear.py' }

for filename in tqdm(files):
    fullname = os.path.join(here, filename)
    os.remove(fullname)
