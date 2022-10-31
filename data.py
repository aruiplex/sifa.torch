import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os
import logging
import itertools
from utils import cfg


class ProstateDataset(Dataset):

    def __init__(self, path) -> None:
        logging.info("Dataset setup")
        self.path = path
        self.data_list = os.listdir(self.path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = np.load(Path(self.path) / Path(self.data_list[index]))
        return {
            "image": torch.from_numpy(sample["arr_0"]),
            "label": torch.from_numpy(sample["arr_1"]),
        }


data_path = {}
# build data path for every procedures and domains
for (procedure, domain) in itertools.product(cfg['procedures'], cfg['domains']):
    data_path.update({f"{procedure}_{domain}": Path(
        cfg["dataset_path"]) / domain / procedure})


dataset = {}
# build dataset for every data path
for (procedure, domain) in itertools.product(cfg['procedures'], cfg['domains']):
    name = f"{procedure}_{domain}"
    dataset.update({name: ProstateDataset(data_path[name])})

dataLoader = {}
# build dataLoader for every dataset
for (procedure, domain) in itertools.product(cfg['procedures'], cfg['domains']):
    name = f"{procedure}_{domain}"
    dataLoader.update({name: DataLoader(dataset=dataset[name], batch_size=2, shuffle=True,
                                        drop_last=True, pin_memory=True)})
