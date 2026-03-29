import torch
import numpy as np
import argparse
import os
from scipy import io
import importlib
from sklearn.cluster import KMeans
from DataSet.DataLoader import get_dataloader
from utils import aucPerformance, get_logger, F1Performance
from sklearn.preprocessing import MinMaxScaler
import glob
import ipdb
import time
import sys

def set_thread_env():
    # Avoid OpenMP/multiprocessing issues on macOS.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

npz_files = glob.glob(os.path.join('./Data', '*.npz'))
npz_datanames = [os.path.splitext(os.path.basename(file))[0] for file in npz_files]

mat_files = glob.glob(os.path.join('./Data', '*.mat'))
mat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]

def resolve_device(device_arg):
    if device_arg is None or device_arg == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if device_arg == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return device_arg

def init_model(model_cls, **kwargs):
    try:
        return model_cls(**kwargs)
    except TypeError:
        kwargs.pop("device", None)
        return model_cls(**kwargs)

def available_datasets(data_dir):
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
    return [os.path.splitext(os.path.basename(file))[0] for file in npz_files + mat_files]

if __name__ == "__main__":

    ### Loading hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='wbc')
    parser.add_argument('--model_type', type=str, default='PN')
    parser.add_argument('--preprocess', type=str, default='none')
    parser.add_argument('--diversity', type=str, default='True')
    parser.add_argument('--plearn', type=str, default='False')
    parser.add_argument('--input_info', type=str, default='True')
    parser.add_argument('--input_info_ratio', type=float, default=0.01)
    parser.add_argument('--cl', type=str, default='True')
    parser.add_argument('--cl_ratio', type=float, default=0.01)
    parser.add_argument('--prototype_num', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--eval_each_epoch', type=str, default='False')
    args = parser.parse_args()

    diversity = True if args.diversity == 'True' else False
    plearn = True if args.plearn == 'True' else False
    input_info = True if args.input_info == 'True' else False
    cl = True if args.cl == 'True' else False

    dict_to_import = 'model_config_'+args.model_type
    module_name = 'configs'
    module = importlib.import_module(module_name)
    if not hasattr(module, dict_to_import):
        raise ValueError(f"Unknown model_type '{args.model_type}'. Available: {[k.replace('model_config_', '') for k in dir(module) if k.startswith('model_config_')]}")
    model_config = getattr(module, dict_to_import)

    model_config['preprocess'] = args.preprocess
    model_config['diversity'] = diversity
    model_config['plearn'] = plearn
    model_config['input_info'] = input_info
    model_config['input_info_ratio'] = args.input_info_ratio
    model_config['cl'] = cl
    model_config['cl_ratio'] = args.cl_ratio
    model_config['random_seed'] = args.seed
    if args.epoch is not None:
        model_config['epochs'] = args.epoch
    model_config['device'] = resolve_device(args.device)
    model_config['eval_each_epoch'] = True if args.eval_each_epoch == 'True' else False

    if args.dataname in npz_datanames:
        path = os.path.join(model_config['data_dir'], args.dataname + '.npz')
        data = np.load(path)
    elif args.dataname in mat_datanames:
        path = os.path.join(model_config['data_dir'], args.dataname + '.mat')
        data = io.loadmat(path)
    else:
        available = available_datasets(model_config['data_dir'])
        raise FileNotFoundError(
            f"Dataset '{args.dataname}' not found in {model_config['data_dir']}. "
            f"Available: {sorted(available)}"
        )
    samples = data['X']
    model_config['dataset_name'] = args.dataname
    model_config['data_dim'] = samples.shape[-1]

    ### System configuration
    if sys.platform == "darwin":
        set_thread_env()

    if model_config['device'] == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if model_config['device'].startswith("cuda"):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(model_config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(model_config['random_seed'])
    np.random.seed(model_config['random_seed'])
    if model_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn')

    ### DRL training
    if args.model_type == 'PN':
        from PrototypeNet.Trainer import Trainer
        model_config['prototype_num'] = args.prototype_num
    else:
        raise ValueError(f"Unsupported model_type '{args.model_type}'.")

    start_time = time.time()
    trainer = Trainer(model_config=model_config)
    trainer.training(model_config['epochs'])
    end_time = time.time()
    train_time = end_time - start_time

    ### DRL Evaluating
    start_time = time.time()
    mse_rauc, mse_ap, mse_f1 = trainer.evaluate()
    end_time = time.time()
    test_time = end_time - start_time

    ### Saving results
    print('##########################################################################')
    print("AUC-ROC: %.4f  AUC-PR: %.4f"
          % (mse_rauc, mse_ap))
    print("f1: %.4f" % (mse_f1))

    results_dict = {'AUC-ROC':mse_rauc, 'AUC-PR':mse_ap, 'f1':mse_f1}
    np.save(open(f'./results/{args.dataname}_{args.model_type}_{args.preprocess}_{args.diversity}_{args.plearn}_{args.input_info}_{args.input_info_ratio}_{args.cl}_{args.cl_ratio}_{args.prototype_num}_{args.seed}.npy','wb'), results_dict)
