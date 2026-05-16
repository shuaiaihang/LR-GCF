import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
from colorama import Fore
from prettytable import PrettyTable
from codes.builder import _build_from_cfg, build_dataloader, build_model
from codes.utils.utils import Namespace, parse_yaml



def get_args(config_file):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default=config_file,
                        help='train config file path: xxx.yaml')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args_dict = parse_yaml(args.config)
    for key, value in Namespace(args_dict).__dict__.items():
        vars(args)[key] = value
    return args


def set_deterministic(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def test(config_file, weights):
    args = get_args(config_file)
    set_deterministic(args.seed)
    def test_worker_init_fn(worker_id):
        random.seed(worker_id + args.seed)
    _, test_loader = build_dataloader(args.dataset, test_worker_init_fn)



    model = build_model(args.model)
    state_dict = torch.load(weights)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)

    print(F'\nModel: {model.__class__.__name__}')
    print(F'Weights file: {weights}')

    test_loader = tqdm(test_loader, desc='Testing', unit='batch',
                       bar_format='%s{l_bar}{bar}{r_bar}%s' % (Fore.LIGHTCYAN_EX, Fore.RESET))

    metrics = []
    if args.train.kwargs.metrics is not None:
        for metric_cfg in args.train.kwargs.metrics:
            metrics.append(_build_from_cfg(metric_cfg))

    test_res = None
    model.to(args.device)
    model.eval()
    for batch_data in test_loader:
        data, label = batch_data['image'].to(args.device), batch_data['label'].to(args.device)
        preds = model.inference_tf(data,args.device)
        batch_metric_res = {}
        for metric in metrics:
            batch_metric_res[metric.name] = metric(preds, label)

        if test_res is None:
            test_res = batch_metric_res
        else:
            for metric_name in test_res.keys():
                for key in test_res[metric_name].keys():
                    test_res[metric_name][key] += batch_metric_res[metric_name][key]
    for metric_name in test_res.keys():
        for key in test_res[metric_name].keys():
            test_res[metric_name][key] = test_res[metric_name][key] / len(test_loader)

        test_res_list = [_.cpu() if isinstance(_, torch.Tensor) else _ for _ in test_res[metric_name].values()]
        test_res[metric_name]['Mean'] = np.mean(test_res_list[1:])

    test_table = PrettyTable()
    test_table.field_names = ['Metirc'] + list(list(test_res.values())[0].keys())
    for metric_name in test_res.keys():
        if metric_name in ['Dice', 'Jaccard', 'Acc', 'IoU', 'Recall', 'Precision']:
            temp = [float(format(_ * 100, '.2f')) for _ in test_res[metric_name].values()]
        else:
            temp = [float(format(_, '.2f')) for _ in test_res[metric_name].values()]
        test_table.add_row([metric_name] + temp)
    print(test_table.get_string())



if __name__ == '__main__':
    config_path = 'model/gcf_unet_r50/gcf_unet_r50.yaml'
    weights_path = 'model/gcf_unet_r50/best_model.pth'
    
    test(config_path, weights_path)
