def get_hyparams_config_class(dataset_name):
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class Boiler:
    def __init__(self):
        super(Boiler, self).__init__()
        self.test_per_step = 50
        self.training_steps = 1000
        self.drop_prob = 0
        self.learning_rate = 0.0015
        self.coeff = 10
        self.h_dim = 10
        self.dense_dim = 100
        self.lstm_layer = 1
        self.weight_decay = 4e-7


