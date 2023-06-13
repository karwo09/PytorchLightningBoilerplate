from LightningWrapper import LightningWrapper
from torchvision.models import resnet50
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

# Define your hyperparameters and settings
class ModelOptions:
    def __init__(self):
        self.lr = 0.01  # learning rate
        self.weight_decay = 0.0005  # weight decay for optimizer
        self.lr_step = 10  # step size for learning rate scheduler
        self.optimizer = 'adam'  # type of optimizer to use

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
