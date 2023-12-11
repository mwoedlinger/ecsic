from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from .utils import get_positional_encoding, get_positional_fourier_encoding
from .transforms import Compose, TWrapper
from time import time as t

__all__ = ['list_datasets', 'Dataset']

def list_datasets(root="./data"):
	return [n.stem for n in Path(root).iterdir() if n.is_dir()]
	
class Dataset(torch.utils.data.Dataset):

    def __init__(self, name, root, data_type, transform=None, debug=False, pos_encoding=False):
        super().__init__()

        list_of_data_names = [n.stem for n in Path(root).iterdir() if n.is_dir()]
        list_of_possible_types = [t.stem for t in (Path(root) / f'{name}').iterdir() if t.suffix == '.txt']

        assert name in list_of_data_names, f'{name} not in list of datasets: {list_of_data_names}.'
        assert data_type in list_of_possible_types, f'{data_type} not possible for {name}. Possible options: {list_of_possible_types}.'

        self.text_file = Path(root) / f'{name}/{data_type}.txt'
        self.pos_encoding = pos_encoding

        transform = transform or [TWrapper(lambda x: x)]
        self.transforms = Compose([*transform])

        self.files = self._filenames()

        if debug:
            self.files = self.files[:10]

    def __len__(self):
        return len(self.files)
        
    def _transform(self, images):
        images = [transforms.ToTensor()(img) for img in images]
        if self.pos_encoding: 
            if not hasattr(self, 'pos') or self.pos.shape[-2:] != images[0].shape[-2:]:
                self.pos = get_positional_fourier_encoding(*images[0].shape[-2:])
            images.append(self.pos)
        return self.transforms(*images)        

    def _filenames(self):
        with open(self.text_file, 'r') as text_file:
            files = [f.replace(' ', '').strip().split(',') for f in text_file.readlines()]
        return files
    
    def __getitem__(self, idx):
        images = [Image.open(f) for f in self.files[idx]]
        return self._transform(images)