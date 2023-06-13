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

if __name__ == "__main__":
    opt = ModelOptions(path='model_options.yml')
    print(opt.lr)