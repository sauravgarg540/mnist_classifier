import logging
import os.path
from typing import Literal, Union

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CHECKPOINT_DIR
from src.utils.plot_utils import plot_confusion_matrix, plot_results
from src.utils.utils import AverageMeter, ConfusionMatrix, get_device, logging


class Engine:
    """
    Class for training and evaluating PyTorch models.

    Args:
        model: The PyTorch model to be trained.
        train_loader: DataLoader for training data. Defaults to None.
        test_loader: DataLoader for testing data. Defaults to None.
        epochs: Number of epochs for training. Defaults to 100.
        criterion: Loss function. Defaults to None.
        optimizer: Optimizer for training. Defaults to None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader = None,
        test_loader: DataLoader = None,
        epochs: int = 100,
        criterion: torch.nn.Module = None,
        optimizer: torch.optim = None,
    ):
        """
        Initializes the Trainer class.

        Args:
            model: The PyTorch model to be trained.
            train_loader: DataLoader for training data. Defaults to None.
            test_loader: DataLoader for testing data. Defaults to None.
            epochs: Number of epochs for training. Defaults to 100.
            criterion: Loss function. Defaults to None.
            optimizer: Optimizer for training. Defaults to None.
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer

        self.device = get_device()
        self.model.to(self.device)
        self.loss_meter = AverageMeter()
        self.accuracy_meter = AverageMeter()
        self.confusion_matrix = ConfusionMatrix()
        self.plot_predictions = True

    def run_one_step(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        mode: Union[str, Literal["train", "eval"]] = 'train',
    ):
        """
        Trains the model for one step.

        Args:
            images: Input images.
            targets: Target labels.
            mode: Mode of operation ('train' or 'eval'). Defaults to 'train'.

        Returns:
            loss: Average loss for the step.
            accuracy: Average accuracy for the step.
        """
        images = images.to(self.device)
        targets = targets.to(self.device)
        if mode == 'train':
            self.optimizer.zero_grad()
        output = self.model(images)
        loss = self.criterion(output, targets)
        self.loss_meter.update(loss.item(), images.size(0))
        if mode == 'train':
            loss.backward()
            self.optimizer.step()

        y_score = (
            torch.topk(output, 1).indices.reshape(output.size(0)).detach().cpu().numpy()
        )
        self.confusion_matrix.y_pred.extend(
            y_score
        )  # Save Prediction for Confusion Matrix
        y_true = targets.detach().cpu().numpy()
        self.confusion_matrix.y_true.extend(y_true)  # Save True for Confusion Matrix
        acc = accuracy_score(y_true, y_score)
        self.accuracy_meter.update(acc, images.size(0))
        if mode == 'eval' and self.plot_predictions:
            plot_results(images.detach().cpu().numpy(), y_score, targets.detach().cpu())
            self.plot_predictions = False
        return self.loss_meter.avg, self.accuracy_meter.avg

    def run_one_epoch(
        self, epoch: int, mode: Union[str, Literal["train", "eval"]] = 'train'
    ) -> (float, float):
        """
        Trains the model for one epoch.

        Args:
            epoch: Current epoch number.
            mode: Mode of operation ('train' or 'eval'). Defaults to 'train'.

        Returns:
            loss: Average loss for one epoch.
            accuracy: Average accuracy for one epoch.
        """

        # reset the loss and accuracy meter for iteration
        self.accuracy_meter.reset()
        self.loss_meter.reset()

        # put model on train or eval mode
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        progress_bar = tqdm(
            self.train_loader if mode == 'train' else self.test_loader,
            desc=f'Epoch {epoch}: {mode}' if mode == 'train' else f'{mode}',
            unit="batch",
        )
        with torch.no_grad() if mode == 'eval' else torch.enable_grad():
            # start iteration loop
            for i, (images, targets) in enumerate(progress_bar):
                loss, accuracy = self.run_one_step(images, targets, mode)
                progress_bar.set_postfix(
                    {
                        f'{mode}_loss': loss,
                        f'{mode}_accuracy': accuracy,
                    },
                    refresh=False,
                )

        return loss, accuracy

    def run(
        self, mode: Union[str, Literal["train", "eval"]] = 'train'
    ) -> (float, float):
        """
        Run engine to train or evaluate models.

        mode: Mode of operation ('train' or 'eval'). Defaults to 'train'.

        Returns:
            loss: loss of the model over test dataset
            accuracy: accuracy of the model over test dataset
            per_class_accuracy: per class accuracy of the model over test dataset
        """
        assert mode in ['train', 'eval'], "Mode should be either 'train' or 'eval'"
        if mode == 'train':
            for epoch in range(self.epochs):
                self.run_one_epoch(epoch)
        mode = "eval"
        loss_test, accuracy_test = self.run_one_epoch(self.epochs, mode=mode)
        conf_mat, per_class_accuracy = self.__compute_eval_metrics()
        plot_confusion_matrix(conf_mat)
        return loss_test, accuracy_test, per_class_accuracy

    def __compute_eval_metrics(self):
        """
        Computes evaluation metrics for the model.

        Returns:
            conf_mat: confusion matrix of the model over test dataset
            per_class_accuracy: per class accuracy of
            the model over test dataset
        """
        conf_mat = self.confusion_matrix.compute()
        per_class_accuracy = self.confusion_matrix.get_per_class_accuracy()
        return conf_mat, per_class_accuracy

    def save_model(self, name: str):
        """
        Saves the model to the given path.

        Args:
            name: name of the model checkpoint.
        """
        checkpoint_file = os.path.join(CHECKPOINT_DIR, name)
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
            },
            checkpoint_file,
        )
        logging.info(f"Model saved at {checkpoint_file}")

    def load_model(self, name: str):
        """
        Loads the model from the given path.

        Args:
            name: name of the checkpoint file.
        """
        checkpoint_file = os.path.join(CHECKPOINT_DIR, name)
        assert os.path.exists(checkpoint_file), "Model path does not exist"
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model loaded from {checkpoint_file}")
