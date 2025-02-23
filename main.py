import torch
import argparse

from dataset.utils import get_time_series_loader
from model.TimeMixer import TimeMixer
from trainer.trainer import Trainer
from utils import parse_config_file

if __name__ == '__main__':
    # 读取参数
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--exp_id', type=str)
    parser.add_argument('--config', type=str, default='tm')
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--train', type=bool, default=True)
    args = parser.parse_args()
    config = parse_config_file(args.config)
    config["exp_id"] = args.exp_id
    use_gpu = config.get('gpu', True)
    gpu_id = config.get('gpu_id', 0)
    if use_gpu:
        torch.cuda.set_device(gpu_id)
    config['device'] = torch.device(
        "cuda:%d" % gpu_id if torch.cuda.is_available() and use_gpu else "cpu")


    loader_train, dataset_train, feature = get_time_series_loader(
        'train', config, args.n_worker)
    loader_valid, dataset_valid, _ = get_time_series_loader(
        'valid', config, args.n_worker)
    loader_test, dataset_test, _ = get_time_series_loader(
        'test', config, args.n_worker)

    model = TimeMixer(config, feature)
    trainer = Trainer(config, model, feature)
    model_cache_file = './cache/{}/model/model.m'.format(
        args.exp_id)
    if args.train:
        trainer.train(loader_train, loader_valid)
        trainer.save_model(model_cache_file)
    else:
        trainer.load_model(model_cache_file)
    trainer.evaluate(loader_test)



