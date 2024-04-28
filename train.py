import logging
import os

import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import BATCH_SIZE, DATA_ROOT_DIR, EPOCHS, MODEL_NAME, CRITERION, OPTIMIZER, LEARNING_RATE
from src.core.engine import Engine
from src.data.dataset import MNISTDataset
from src.data.transforms import Normalize, ToTorchFormatTensor
from src.model.mnist_model import MNISTModel
from src.utils.utils import logging
from src.optimizer.optimizer import get_optimizer
from src.criterion.criterion import get_criterion


def main():

    data_transform = T.Compose([ToTorchFormatTensor(), Normalize()])
    train_generator = MNISTDataset(
        os.path.join(DATA_ROOT_DIR, "train"), transform=data_transform
    )
    test_generator = MNISTDataset(
        os.path.join(DATA_ROOT_DIR, "test"), transform=data_transform
    )

    train_loader = DataLoader(train_generator, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_generator, batch_size=BATCH_SIZE, shuffle=False)
    logging.info("Data loaders created...")

    model = MNISTModel()
    logging.info("Model created...")

    criterion = get_criterion(name=CRITERION)
    logging.info("Loss function created...")
    optimizer = get_optimizer(OPTIMIZER, model.parameters(), lr=LEARNING_RATE)

    epochs = EPOCHS

    trainer = Engine(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
    )
    logging.info("Engine created...")
    logging.info("Starting training...")
    _, accuracy_test, per_class_accuracy = trainer.run(mode="train")
    trainer.save_model(name=MODEL_NAME)
    logging.info(f"Accuracy: {accuracy_test}")

    for i, acc in enumerate(per_class_accuracy):
        string = f"Class {i} accuracy"
        logging.info(f"{string:<8}: {acc}")


if __name__ == "__main__":
    main()
