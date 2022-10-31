import traceback
import torch
import os
from model import SIFA
import time
from data import dataLoader
import logging
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import datetime
import utils
writer = SummaryWriter()


def train(model=SIFA()):
    for epoch in range(utils.cfg["epochs"]):
        for sample_a, sample_b in zip(dataLoader["train_A"], dataLoader["train_B"]):
            image_a = sample_a["image"].cuda()
            label_a = sample_a["label"].cuda()
            image_b = sample_b["image"].cuda()
            label_b = sample_b["label"].cuda()
            print(image_a, label_a, image_b, label_b)


if __name__ == "__main__":
    train()
