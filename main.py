import os
import argparse
import yaml

from torch.backends import cudnn
from anomaly_transformer.solver import Solver
from loguru import logger

def main(config):
    cudnn.benchmark = True
    mode = config['mode']
    os.makedirs(config[f'{mode}_settings']['model_save_path'], exist_ok=True)
    solver = Solver(config)

    if mode == 'train':
        solver.train()
        solver.test()
    elif mode == 'test':
        solver.test()

    return solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config.yaml', help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)['anomaly_detection']

    logger.info('------------ Options -------------')
    for k, v in sorted(config.items()):
        logger.info('%s: %s' % (str(k), str(v)))
    logger.info('-------------- End ----------------')
    
    main(config)