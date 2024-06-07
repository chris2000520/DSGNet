from .base_config import BaseConfig


class MyConfig(BaseConfig):
    def __init__(self,):
        super(MyConfig, self).__init__()
        # Dataset
        self.dataset = 'cityscapes'
        self.data_root = '/mnt/data/cityscapes'
        self.num_class = 19
        
        # Model
        self.model = 'cfgnet'
        
        # Training
        self.total_epoch = 200
        self.train_bs = 8
        self.loss_type = 'ohem'
        self.optimizer_type = 'adam'
        self.logger_name = 'seg_trainer'
        # self.use_aux = True
        self.aux_coef = [0.4] # 辅助损失参数

        # Validating
        self.val_bs = 10
        
        # Testing
        # self.is_testing = True
        # self.load_ckpt = True
        self.test_bs = 8
        self.test_data_folder = '/mnt/data/cityscapes/leftImg8bit/val/frankfurt'
        self.load_ckpt_path = '/mnt/save/best.pth'
        self.save_mask = True
        
        # Training setting
        self.use_ema = False
        self.amp_training = False
        
        # Augmentation
        self.crop_size = 1024
        self.randscale = [-0.5, 1.0]
        self.scale = 1.0
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.h_flip = 0.5
        
