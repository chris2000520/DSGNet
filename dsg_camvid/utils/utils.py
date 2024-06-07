import os, random, torch, json
import numpy as np
from matplotlib import pyplot as plt





def matplot_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 8))
    plt.plot(train_process['epoch'], train_process.train_loss_all,  label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all,  label="Val loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('./loss.jpg')

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
def get_writer(config, main_rank):
    if config.use_tb and main_rank:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.tb_log_dir)
    else:
        writer = None
    return writer
    
    
def get_logger(config, main_rank):
    if main_rank:
        import sys
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, format="[{time:YYYY-MM-DD HH:mm}] {message}", level="INFO")

        log_path = f'{config.save_dir}/{config.logger_name}.log'
        logger.add(log_path, format="[{time:YYYY-MM-DD HH:mm}] {message}", level="INFO")
    else:
        logger = None
    return logger


def save_config(config):
    config_dict = vars(config)
    with open(f'{config.save_dir}/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)


def log_config(config, logger):
    keys = ['dataset', 'num_class', 'model', 'encoder', 'decoder', 'loss_type', 
            'optimizer_type', 'lr_policy', 'total_epoch', 'train_bs', 'val_bs',  
            'train_num', 'val_num', 'gpu_num', 'num_workers', 'amp_training', 
            'DDP', 'kd_training', 'synBN', 'use_ema', 'use_aux']
            
    config_dict = vars(config)
    infos = f"\n\n\n{'#'*25} Config Informations {'#'*25}\n" 
    infos += '\n'.join('%s: %s' % (k, config_dict[k]) for k in keys)
    infos += f"\n{'#'*71}\n\n"
    logger.info(infos)
    

def get_colormap(config):
    if config.colormap == 'camvid':
        colormap = {0:(128, 128,128), 1:(128, 0, 0), 2:( 192, 192, 128), 3:(128,64,128),
                    4:(  0,   0,192), 5:(128,128,0), 6:( 192, 128, 128), 7:( 64,64,128),
                    8:( 64,   0,128), 9:( 64, 64,0), 10:(  0, 128, 192), 11:( 0, 0,  0),
                    }

    elif config.colormap == 'custom':
        raise NotImplementedError()
        
    else:
        raise ValueError(f'Unsupport colormap type: {config.colormap}.')

    colormap = [color for color in colormap.values()]
    
    if len(colormap) < config.num_class:
        raise ValueError('Length of colormap is smaller than the number of class.')
    else:
        return colormap[:config.num_class]