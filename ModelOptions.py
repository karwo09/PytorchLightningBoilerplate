class ModelOptions:
    def __init__(self):
        # Model options
        self.backbone = 'resnet50'  # model backbone
        self.num_classes = 1000  # number of classes in your dataset

        # Loss options
        self.loss = 'CrossEntropyLoss'  # type of loss function to use

        # Training options
        self.lr = 0.01  # learning rate
        self.weight_decay = 0.0005  # weight decay for optimizer
        self.lr_step = 10  # step size for learning rate scheduler

        # Optimizer options
        self.optimizer = 'sgd'  # type of optimizer to use

        # Logging options
        self.logging_dir = 'tb_logs'  # directory for TensorBoard logs
        self.logging_name = 'my_model'  # name of your model for logging purposes
