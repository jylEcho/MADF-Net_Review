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
import tqdm

from torch.utils.data import DataLoader
from datasets.dataset_multiphase import Multiphase_dataset
from trainer.trainer_single import inference  as inference_single
from  trainer.trainer_phase import inference as inference_multiphase

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="/data/qklee/liver/process_data/train/",
    help="root dir for train data",
)
parser.add_argument(
    "--test_path",
    type=str,
    default="/data/qklee/liver/process_data/test/",
    help="root dir for test data",
)
parser.add_argument("--dataset", type=str, default="Singlephase", help="experiment_name")
parser.add_argument("--list_dir", type=str, default="./lists/lists_liver", help="list dir")
parser.add_argument("--num_classes", type=int, default=3, help="output channel of network")
parser.add_argument("--output_dir", type=str, default="./model_out", help="output dir")
parser.add_argument("--batch_size", type=int, default=1, help="batch_size per gpu")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument("--eval_interval", type=int, default=5, help="eval_interval")
parser.add_argument("--model_name", type=str, default="Acftransunet", help="model_name")
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument("--deterministic", type=int, default=0, help="whether to use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network base learning rate")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--z_spacing", type=int, default=1, help="z_spacing")
parser.add_argument("--seed", type=int, default=49, help="random seed")
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
parser.add_argument("--crop_inference",default="standard_crop_inference",\
                    help="style of inference: resize_inference, standard_crop_inference, random_crop_inference")
parser.add_argument("--crop_train",default='max_crop',\
                    help="style of train: random_crop, max_crop, no")

parser.add_argument("--is_savenii", default= 1, help="whether to save results during inference")

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
    "--module", default="networks.acftransunet.VisionTransformer",help="The module that you want to load as the network, e.g. networks.DAEFormer.DAEFormer"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="Acftransunet_Singlephase_epoch_499.pth",
    help="The path of the checkpoint to be loaded, e.g../saved_models/model_epoch_100.pth",
)

args = parser.parse_args()


if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
# config = get_config(args)


def dynamic_import(module_name, function_name):
    # 导入指定的模块
    module = importlib.import_module(module_name)
    # 获取模块中的函数
    function = getattr(module, function_name)
    return function

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    transformer = locate(args.module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("model:", args.module)

    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), "GB")
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

        net = transformer(args.num_classes, 1, 64).cuda()
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
    else:
        net = transformer()
    if args.checkpoint:
        lr_ = torch.load(os.path.join(args.output_dir, args.checkpoint))['lr']
        args.base_lr = lr_ if args.base_lr > lr_ else args.base_lr
        print("use checkpoint:", args.checkpoint, 'lr:', args.base_lr)

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

    if args.model_name == 'Acftransunet':
        args.crop_inference = dynamic_import("utils.utils_singlephase", args.crop_inference)
    else:
        args.crop_inference = dynamic_import("utils.utils_multiphase", args.crop_inference)

    logging.basicConfig(
        filename=args.output_dir + "/log_test3.txt", level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    db_test = Multiphase_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir,
                                 img_size=args.img_size)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    if args.model_name == 'Acftransunet':
        inference_single(net, testloader, args, test_save_path=None)
    else:
        inference_multiphase(net, testloader, args, test_save_path=None)
