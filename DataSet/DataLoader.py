from torch.utils.data import DataLoader
from DataSet.MyDataset import CsvDataset, MatDataset, NpzDataset
import os
import glob

npz_files = glob.glob(os.path.join('./Data', '*.npz'))
npz_datanames = [os.path.splitext(os.path.basename(file))[0] for file in npz_files]

mat_files = glob.glob(os.path.join('./Data', '*.mat'))
mat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]

def get_dataloader(model_config: dict):
    dataset_name = model_config['dataset_name']

    if dataset_name in npz_datanames:
        train_set = NpzDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], model_config['preprocess'], model_config, mode='train')
        test_set = NpzDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], model_config['preprocess'], model_config, mode='eval')

    elif dataset_name in mat_datanames:
        train_set = MatDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], model_config['preprocess'], model_config, mode='train')
        test_set = MatDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], model_config['preprocess'], model_config, mode='eval')
        
    else:
        train_set = CsvDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], model_config['preprocess'], model_config, mode='train')
        test_set = CsvDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], model_config['preprocess'], model_config, mode='eval')

    train_loader = DataLoader(train_set,
                              batch_size=model_config['batch_size'],
                              num_workers=model_config['num_workers'],
                              shuffle=False,
                              )
    test_loader = DataLoader(test_set, batch_size=model_config['batch_size'], shuffle=False)
    return train_loader, test_loader