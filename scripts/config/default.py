from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.EXP = 'default'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.name = 'Adobe_Image_Matting'
_C.DATASET.data_dir = '/media/yutong/DATA/data/datasets/Combined_Dataset'
_C.DATASET.train_fg_list = 'train_fg.txt'
_C.DATASET.train_alpha_list = 'train_alpha.txt'
_C.DATASET.train_bg_list = 'train_bg.txt'
_C.DATASET.val_list = 'test.txt'

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.stride = 16
_C.MODEL.aspp = False

# backbone
_C.MODEL.backbone = 'resnet34'  # ['resnet34', 'mobilenetv2']
_C.MODEL.arch = 'resnet34'
_C.MODEL.activation = 'leaky_relu'  # supported: relu, leaky_relu, elu, identity
_C.MODEL.activation_param = 0.01   # slope for leaky_relu, alpha for elu
_C.MODEL.input_3x3 = False
_C.MODEL.bn_mode = 'sync'  # supported: standard, inplace, sync
_C.MODEL.weight_gain_multiplier = 1  # note: this is ignored if weight_init == kaiming_*
_C.MODEL.weight_init = 'xavier_normal'  # supported: xavier_[normal,uniform], kaiming_[normal,uniform], orthogonal

# decoder
_C.MODEL.decoder_conv_operator = 'std_conv'  # ['std_conv', 'residual_conv']
_C.MODEL.decoder_kernel_size = 5
_C.MODEL.decoder_block_num = 1


# bilinear upsample
_C.MODEL.up_kernel_size = 3
_C.MODEL.downupsample_group = 1
_C.MODEL.encode_kernel_size = 5
_C.MODEL.share = False  # whether to share among different channels


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.restore = 'model_ckpt.pth'
_C.TRAIN.model_save_dir = './savemodel'
_C.TRAIN.result_dir = './results'
_C.TRAIN.random_seed = 6
_C.TRAIN.evaluate_only = False
_C.TRAIN.crop_size = 320
_C.TRAIN.batch_size = 2
_C.TRAIN.initial_lr = 0.01
_C.TRAIN.momentum = 0.9
_C.TRAIN.mult = 100
_C.TRAIN.num_epochs = 50
_C.TRAIN.epoch_iterations = 6000
_C.TRAIN.num_workers = 0
_C.TRAIN.print_every = 1
_C.TRAIN.record_every = 20
_C.TRAIN.apex = False  # whether to use fp16
_C.TRAIN.freeze_bn = False
_C.TRAIN.load_data = False  # whether to load fg, alpha to memory
_C.TRAIN.load_bg = False  # whether to load bg to memory
_C.TRAIN.lmdb = False  # whether to load data from lmdb file. It will obviously increase virtual memory usage.
_C.TRAIN.random_bgidx = False   # whether to use random bg for training samples
_C.TRAIN.all_bg = False  # whether to generate training data using all bg images


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
_C.VAL.batch_size = 1
_C.VAL.test_all_metrics = False  # set to True if you want to measure all 4 metrics during validation, other wise only sad and grad would be measured

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.batch_size = 1
_C.TEST.checkpoint = 'model_best.pth'
_C.TEST.result_dir = './testresults'
