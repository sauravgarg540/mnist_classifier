import os

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import BATCH_SIZE, DATA_ROOT_DIR, MODEL_NAME
from src.core.engine import Engine
from src.data.dataset import MNISTDataset
from src.data.transforms import Normalize, ToTorchFormatTensor
from src.model.mnist_model import MNISTModel
from src.utils.utils import logging


def eval():

    data_transform = T.Compose([ToTorchFormatTensor(), Normalize()])

    test_generator = MNISTDataset(
        os.path.join(DATA_ROOT_DIR, "test"), transform=data_transform
    )

    test_loader = DataLoader(test_generator, batch_size=BATCH_SIZE, shuffle=True)
    logging.info("Data loaders created...")

    model = MNISTModel()
    logging.info("Model created...")

    criterion = torch.nn.CrossEntropyLoss()
    trainer = Engine(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
    )
    trainer.load_model(name=MODEL_NAME)
    logging.info("Engine created...")
    logging.info("Starting Evaluation...")
    _, accuracy_test, per_class_accuracy = trainer.run(mode="eval")
    logging.info(f"Accuracy: {accuracy_test}")

    for i, acc in enumerate(per_class_accuracy):
        string = f"Class {i} accuracy"
        logging.info(f"{string:<8}: {acc}")


if __name__ == "__main__":
    eval()
