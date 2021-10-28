import torch


def build_optimizer(model, config):
    if config.type == 'SGD':
        return torch.optim.SGD(model.parameters(),
                               lr=config.lr,
                               weight_decay=config.weight_decay
                               )
    elif config.type == 'Adam':
        return torch.optim.Adam(model.parameters(),
                                lr=config.lr,
                                weight_decay=config.weight_decay,
                                betas=config.betas
                                )
    else:
        raise KeyError("Unknown optimizer type {}!".format(config.type))
