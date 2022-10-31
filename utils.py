import torch
import os
import datetime
import logging
import json

cfg = json.load(open("config.json"))


def generate_filename():
    return datetime.datetime.now().strftime('%m-%d_%H-%S')


def check_dirs():
    os.makedirs(cfg["check_point_path"], exist_ok=True)
    os.makedirs(cfg["logging_file_path"], exist_ok=True)


def init_logging():
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(f"logs/{generate_filename()}.log"),
            logging.StreamHandler()
        ]
    )


def save(epoch, model, optimizer, loss):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join("checkpoint", f"{generate_filename}_{epoch}_epochs.pth")
    )
    logging.info("checkpoint has been saved.")


def load(checkpoint_path: str, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logging.info(
        f"Checkpoints has been loaded. Epoch number: {epoch}, loss: {loss}")
