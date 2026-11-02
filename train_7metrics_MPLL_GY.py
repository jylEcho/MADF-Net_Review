import argparse
import logging
import os
import random
import warnings
import time
import  sys
from pydoc import locate
# from test_multiphase import  dynamic_import
import importlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.nn as nn

from  trainer.trainer_lits import trainer_lits
from trainer.trainer_synapse import trainer_synapse
from trainer.trainer_phase_7metrics_MPLL_JSC import trainer_multiphase
from trainer.trainer_single_7metrics_MPLL_GY import trainer_single

from networks.unet_family import NestedUNet

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="/home/image006/multi-phase/dataset_2d/dataset_v/train/",
    help="root dir for train data",
)
parser.add_argument(
    "--test_path",
    type=str,
    default="/home/image006/multi-phase/dataset_2d/dataset_v/test/",
    help="root dir for test data",
)
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--list_dir", type=str, default="./lists/lists_liver", help="list dir")
parser.add_argument("--num_classes", type=int, default=3, help="output channel of network")

# parser.add_argument("--output_dir", type=str, default="./model_out", help="output dir")
parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/model_out", help="output dir")

parser.add_argument("--max_iterations", type=int, default=90000, help="maximum epoch number to train")
parser.add_argument("--max_epochs", type=int, default=400, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument("--eval_interval", type=int, default=5, help="eval_interval")
parser.add_argument("--model_name", type=str, default="synapse", help="model_name")
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument("--deterministic", type=int, default=0, help="whether to use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network base learning rate")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--z_spacing", type=int, default=1, help="z_spacing")
parser.add_argument("--seed", type=int, default=49, help="random seed")
parser.add_argument("--single_phase", type=str, default="PV", choices=["ART", "PV", "DL", "A", "V", "D"],
                    help="single CECT phase used by single-input models")
parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="no: no cache, "
    "full: cache all data, "
    "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
)
parser.add_argument("--resume", help="resume from checkpoint")
parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
)
parser.add_argument("--crop_inference",default="random_crop_inference",\
                    help="style of inference: resize_inference, standard_crop_inference, random_crop_inference")
parser.add_argument("--crop_train",default='max_crop',\
                    help="style of train: random_crop, max_crop, no")

# parser.add_argument("--is_savenii", default= 1, help="whether to save results during inference")
# parser.add_argument("--is_savenii", default= 0, help="whether to save results during inference")
parser.add_argument("--is_savenii", type=int, default=0, help="whether to save results during inference")

parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--throughput", action="store_true", help="Test throughput only")
parser.add_argument(
    "--module", default="networks.unet.UNet",help="The module that you want to load as the network, e.g. networks.DAEFormer.DAEFormer"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="The path of the checkpoint to be loaded, e.g../saved_models/model_epoch_100.pth",
)
parser.add_argument("--medsam_checkpoint", type=str, default=None,
                    help="Path to a SAM/MedSAM checkpoint for networks.single_phase_ours_models.SinglePhaseMedSAM")
parser.add_argument("--medsam_model_type", type=str, default="vit_b",
                    help="SAM/MedSAM backbone type, e.g. vit_b, vit_l, vit_h")
parser.add_argument("--freeze_medsam_encoder", type=int, default=1,
                    help="Freeze MedSAM/SAM image encoder when using SinglePhaseMedSAM")

args = parser.parse_args()

def dynamic_import(module_name, function_name):
    # 导入指定的模块
    module = importlib.import_module(module_name)
    # 获取模块中的函数
    function = getattr(module, function_name)
    return function

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias, 0)
if __name__ == "__main__":
    # setting device on GPU if available, else CPU
    transformer = locate(args.module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("model:",args.module)



    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        "Synapse": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": args.num_classes,
        },
        "Multiphase": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": args.num_classes,
        },
        "Singlephase": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": args.num_classes,
        },
    }

    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.root_path = dataset_config[dataset_name]["root_path"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.model_name == 'Pareseg':
         net = transformer(args.num_classes,1,64).cuda()
    elif args.model_name == 'Acftransunet':
         from networks.acftransunet import CONFIGS
         vit_name = 'R50'
         img_size = 224
         vit_patches_size = 16
         config_vit = CONFIGS[vit_name]
         # config_vit.hidden_size = 128 * 8
         config_vit.n_classes = 3
         # config_vit.ff
         config_vit.n_skip = 3
         if vit_name.find('R50') != -1:
             config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
         net = transformer(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
##################################################################################################################         
    # elif args.model_name == 'Acftransunet':
    #     from  networks.acftransunet_version2 import CONFIGS
    #
    #     vit_name = 'R50'
    #     img_size = 224
    #     vit_patches_size = 16
    #     config_vit = CONFIGS[vit_name]
    #     config_vit.hidden_size = 128 * 8
    #     config_vit.n_classes = 3
    #     # config_vit.ff
    #     config_vit.n_skip = 3
    #     if vit_name.find('R50') != -1:
    #         config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    #     net = transformer(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
##################################################################################################################  
    # for swin-uent:
    # else:
    #     from config.config import get_config
    #     args.cfg = "/root/autodl-tmp/multi_phase/config/swin_tiny_patch4_window7_224_lite.yaml"
    #     args.opts = []   # ⭐ 手动补上空列表
    #     config = get_config(args)
    #     # config = get_config("/root/autodl-tmp/multi_phase/config/swin_tiny_patch4_window7_224_lite.yaml")
    #     net = transformer(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
##################################################################################################################    
        # net = transformer(args)
    elif args.model_name == 'ASSNet':    
        net = transformer(args).cuda()
    elif args.model_name == 'Transunet':
        from networks.transunet import CONFIGS
        config_vit = CONFIGS['R50']
        config_vit.n_classes = args.num_classes  # 或你的类别数
        config_vit.n_skip = 3
        config_vit.patches.grid = (args.img_size // 16, args.img_size // 16)
        net = transformer(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    elif args.model_name == 'SwinUnet':    
        from config.config import get_config
        args.cfg = "/root/autodl-tmp/multi_phase/config/swin_tiny_patch4_window7_224_lite.yaml"
        args.opts = []   # ⭐ 手动补上空列表
        config = get_config(args)
        # config = get_config("/root/autodl-tmp/multi_phase/config/swin_tiny_patch4_window7_224_lite.yaml")
        net = transformer(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    elif args.model_name == 'KiU_Net':    
        net = transformer(args).cuda()
    elif args.model_name == 'NestedUNet':
        # 传入输入通道数等参数，不要传args整体给transformer
        in_channels = 1  # 你的输入通道，比如3、1
        num_classes = 3  # 分割类别数
        net = NestedUNet(in_channels, num_classes).cuda()
    elif args.model_name == 'PA-ResSeg':    
        net = transformer(args.num_classes,1,64).cuda()
    elif args.model_name == 'MCDAnet':    
        net = transformer(num_classes=args.num_classes).cuda()
    elif args.model_name == 'MW-UNet':    
        net = transformer().cuda()
    elif args.model_name == 'SAnet':    
        net = transformer().cuda()
###########################################        
    elif args.model_name == 'VMamba':    
        net = transformer(num_classes=args.num_classes, img_size=args.img_size).cuda()
    elif args.model_name == 'Swin-UMamba':    
        net = transformer(num_classes=args.num_classes, img_size=args.img_size).cuda()
########################################
    elif args.model_name == 'Mamba_UNet':
        net = transformer(num_classes=args.num_classes, img_size=args.img_size).cuda()
    elif args.model_name == 'VMunet':
        net = transformer(num_classes=args.num_classes, img_size=args.img_size).cuda()
###########################################
    elif args.model_name == 'MedSAM':
        net = transformer(
            num_classes=args.num_classes,
            img_size=args.img_size,
            checkpoint=args.medsam_checkpoint,
            model_type=args.medsam_model_type,
            freeze_encoder=args.freeze_medsam_encoder,
        ).cuda()
    elif args.model_name == 'SvANet':
        net = transformer(
            num_classes=args.num_classes,
            img_size=args.img_size,
        ).cuda()
###########################################
    else:
        net = transformer(args).cuda()
        # for swin-unet : 
        # net = transformer(config)
##################################################################################################################  
        



        
    if args.checkpoint:
        lr_ = torch.load(os.path.join(args.output_dir, args.checkpoint))['lr']
        args.base_lr = lr_ if args.base_lr > lr_  else args.base_lr
        print("use checkpoint:",args.checkpoint,'lr:',args.base_lr)

        state_dict = torch.load(os.path.join(args.output_dir, args.checkpoint))['state_dict']
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)

    assert args.crop_inference is not None
    # print(f'inference_single_volume_style: {args.crop_inference}\ntrain_single_volume_style: {args.crop_train}')

    # 获取当前的本地时间
    current_time = time.localtime()

    # 提取年、月、日和小时
    year = str(current_time.tm_year)
    month = str(current_time.tm_mon)
    day = str(current_time.tm_mday)
    hour = str(current_time.tm_hour)
#####################################################################################
    # logging.basicConfig(filename=args.output_dir + "/log_"+args.model_name+"_"+str(args.max_epochs)+"_epoch_"+str(args.base_lr)+"_lr_"+year+month+\
    #                              day+hour+".txt", level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # # logging.info("use checkpoint:", args.checkpoint, 'lr:', args.base_lr)
    # logging.info(f"use checkpoint: {args.checkpoint}, lr: {args.base_lr}")
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#####################################################################################
    log_file = os.path.join(
        args.output_dir,
        f'log_{args.model_name}_{time.strftime("%Y%m%d_%H%M%S")}.txt'
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    print("log file:", log_file)
#####################################################################################
    logging.info('train_single_volume_style:%s, inference_single_volume_style:%s', args.crop_train, args.crop_inference)
    logging.info(str(args))

    # Choose the matching 7-metric inference utility.
    # Singlephase dataset / single-input models should use utils_singlephase_7metrics.
    if dataset_name == 'Singlephase' or args.model_name in ['Acftransunet', 'Daeformer', 'ASSNet']:
        args.crop_inference = dynamic_import("utils.utils_singlephase_7metrics_GY", args.crop_inference)
    else:
        args.crop_inference = dynamic_import("utils.utils_multiphase_7metrics_JSC", args.crop_inference)

    trainer = {
        "Synapse": trainer_synapse,
        "Multiphase":trainer_multiphase,
        'Lits': trainer_lits,
        'Singlephase':trainer_single
    }

    # net.apply(weights_init)


    trainer[dataset_name](args, net, args.output_dir)
