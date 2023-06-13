import yaml

class ModelOptions:
    def __init__(self, path=None):
        if path is not None:
            with open(path, 'r') as file:
                options = yaml.safe_load(file)
            self.__dict__.update(options)
        else:
            # default settings
            self.lr = 0.01  # learning rate
            self.weight_decay = 0.0005  # weight decay for optimizer
            self.lr_step = 10  # step size for learning rate scheduler
            self.optimizer = 'adam'  # type of optimizer to use
            self.model_architecture = 'resnet50'  # model architecture
            self.pretrained = True  # use pre-trained model or not
            self.num_epochs = 100  # number of training epochs
            self.batch_size = 64  # batch size for training
            self.num_workers = 4  # number of workers for data loading
            self.shuffle = True  # shuffle dataset for each epoch or not
            self.pin_memory = True  # pin memory for data loading or not
            self.dropout = 0.5  # dropout rate
            self.activation = 'relu'  # activation function to use
            self.loss_function = 'CrossEntropyLoss'  # loss function to use
            self.metrics = ['accuracy']  # metrics to track
            self.logging_dir = 'tb_logs'  # directory for TensorBoard logs
            self.logging_name = 'my_model'  # name of your model for logging purposes
            self.save_model_path = './model.pth'  # path to save the model


if __name__ == "__main__":
    opt = ModelOptions(path='model_options.yml')
    print(opt.lr)