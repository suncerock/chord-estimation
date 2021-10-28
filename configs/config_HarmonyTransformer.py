class OptimizerConfig(object):
    def __init__(self):
        self.type = 'Adam'
        self.lr = 1e-3
        self.betas = (0.9, 0.999)
        self.weight_decay = 0.04


class SchedulerConfig(object):
    def __init__(self):
        self.epochs = 100

        self.type = 'cosine'
        self.T_max = 100
        self.eta_min = 1e-6


class Config(object):
    def __init__(self):
        self.start_epoch = None
        self.log_interval = 1
        self.optimizer_cfg = OptimizerConfig()
        self.scheduler_cfg = SchedulerConfig()

