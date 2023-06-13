"""
LightningWrapper: A PyTorch Lightning general-purpose wrapper

This module defines a LightningModule subclass that wraps a specified model for use with PyTorch Lightning.
It provides a general-purpose framework that simplifies the training, validation, and optimizer configuration steps.

How to use:
Define your data, optimizer settings, model, loss function, and any other optional parameters,
then simply pass these to the LightningWrapper when initializing it. 

The input model and loss function (criterion) are necessary parameters. 
An optional metric function (metric_fc) can also be passed.

Image logging during the validation step is provided. This uses torchvision to convert tensor outputs to images,
and then logs these to TensorBoard.
"""

from ModelOptions import ModelOptions
from pytorch_lightning import LightningModule
from torch import nn
import torch
from torchvision.utils import make_grid
import numpy as np

class LightningWrapper(LightningModule):
    def __init__(self, opt, model, criterion, metric_fc=None):
        super().__init__()
        self.opt = opt  # optimizer options
        self.model = model  # the main model
        self.criterion = criterion  # the loss function
        self.metric_fc = metric_fc  # optional metric function

        # Using DataParallel for multi-GPU training
        try:
            self.model = torch.nn.DataParallel(self.model)
            if self.metric_fc is not None:
                self.metric_fc = torch.nn.DataParallel(self.metric_fc)
        except Exception as e:
            # Catch and print any exceptions that occurred during initialization
            print("Error while initializing model or metric_fc: ", e)

    def forward(self, x):
        # Forward pass of the model
        try:
            feature = self.model(x)
            if self.metric_fc is not None:
                output = self.metric_fc(feature)
            else:
                output = feature
        except Exception as e:
            # Catch and print any exceptions that occurred during forward pass
            print("Error during forward pass: ", e)
            output = None
        return output

    def training_step(self, batch, batch_idx):
        # Training step
        try:
            data_input, label = batch
            data_input = data_input.to(self.device)
            label = label.to(self.device).long()
            output = self(data_input)
            loss = self.criterion(output, label)
            # Logging the training loss to TensorBoard
            self.logger.experiment.add_scalar('Loss/Train', loss, self.current_epoch)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        except Exception as e:
            # Catch and print any exceptions that occurred during the training step
            print("Error during training step: ", e)
            loss = None
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        try:
            data_input, label = batch
            data_input = data_input.to(self.device)
            label = label.to(self.device).long()
            output = self(data_input)
            loss = self.criterion(output, label)
            # Logging the validation loss to TensorBoard
            self.logger.experiment.add_scalar('Loss/Validate', loss, self.current_epoch)
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            # Convert output tensor to images and log to TensorBoard
            grid = make_grid(output.detach().cpu(), nrow=8, normalize=True)
            self.logger.experiment.add_image('images', grid, self.current_epoch)

        except Exception as e:
            # Catch and print any exceptions that occurred during the validation step
            print("Error during validation step: ", e)
            loss = None
        return loss

    def configure_optimizers(self):
        # Configuring the optimizer and scheduler
        try:
            if self.opt.optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
            else:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lr_step, gamma=0.1)
        except Exception as e:
            # Catch and print any exceptions that occurred during optimizer configuration
            print("Error during optimizer configuration: ", e)
            optimizer = None
            scheduler = None
        return [optimizer], [scheduler]


if __name__ == "__main__":
    """
    To use this script, follow these steps:

    1. Import your data, model, loss function and optional metric function.
    2. Initialize your model, loss function and optional metric function.
    3. Create an instance of the LightningWrapper, passing your model, loss function and optional metric function to it.
    4. Initialize a PyTorch Lightning Trainer, passing any desired parameters to it.
    5. Call the Trainer's fit method, passing your LightningWrapper instance and your data.

    Note: The Logger is set to a TensorBoardLogger to enable TensorBoard support.
    """

    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from torch.utils.data import DataLoader
    from LightningWrapper import LightningWrapper
    from torchvision.models import resnet50
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms

    # Your data here. For this example, I will use CIFAR10
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    # Model and Loss function
    model = resnet50(pretrained=True)  # Pretrained ResNet50 model
    loss_func = CrossEntropyLoss()  # Cross-entropy loss function

    opt = ModelOptions()  # Your model options
    wrapper = LightningWrapper(opt, model, loss_func)  # Wrap your model with the PyTorch Lightning Wrapper

    # Define the logger for TensorBoard
    logger = TensorBoardLogger(save_dir=opt.logging_dir, name=opt.logging_name)

    # Initialize the PyTorch Lightning trainer and start training
    trainer = Trainer(max_epochs=3, logger=logger)
    trainer.fit(wrapper, trainloader)

