import torch


def build_scheduler(optimizer, config):
    num_epochs = config.epochs
    if config.type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.step_size,
                                                    gamma=config.gamma
                                                    )
    elif config.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=config.T_max,
                                                               eta_min=config.eta_min
                                                               )
    else:
        raise KeyError("Unknown scheduler type {}!".format(config.type))
    return scheduler, num_epochs

