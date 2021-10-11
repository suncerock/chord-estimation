import numpy as np
import torch.utils.data as Data


def load_data():
    """
    x_train:
    :return:
    """
    fileDir = 'preprocessed_data/Billboard_data_mirex_Mm_model_input_final.npz'
    with np.load(fileDir, allow_pickle=True) as input_data:
        x_train = input_data['x_train']
        TC_train = input_data['TC_train']
        y_train = input_data['y_train']
        y_cc_train = input_data['y_cc_train']
        y_len_train = input_data['y_len_train']
        x_valid = input_data['x_valid']
        TC_valid = input_data['TC_valid']
        y_valid = input_data['y_valid']
        y_cc_valid = input_data['y_cc_valid']
        y_len_valid = input_data['y_len_valid']
        split_sets = input_data['split_sets']
    split_sets = split_sets.item()

    return x_train, TC_train, y_train, y_cc_train, y_len_train, \
        x_valid, TC_valid, y_valid, y_cc_valid, y_len_valid, \
        split_sets


def build_dataloader():

    x_train, TC_train, y_train, y_cc_train, y_len_train, \
        x_valid, TC_valid, y_valid, y_cc_valid, y_len_valid, \
        split_sets = load_data()

    class Dataset(Data.Dataset):
        def __init__(self, x, y_cc, y, y_len):
            super(Dataset, self).__init__()
            self.x = x
            self.y_cc = y_cc
            self.y = y
            self.y_len = y_len

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, item):
            return self.x[0], self.y_cc[0], self.y[0], self.y_len[0]

    train_dataset = Dataset(x_train, y_cc_train, y_train, y_len_train)
    valid_dataset = Dataset(x_valid, y_cc_valid, y_valid, y_len_valid)

    train_dataloader = Data.DataLoader(train_dataset, batch_size=64)
    valid_dataloader = Data.DataLoader(valid_dataset, batch_size=64)

    return train_dataloader, valid_dataloader


if __name__ == '__main__':
    train_dataloader, valid_dataloader = build_dataloader()
    for x, y_cc, y, y_len in train_dataloader:
        print(x.shape, y_cc.shape, y.shape, y_len.shape)
