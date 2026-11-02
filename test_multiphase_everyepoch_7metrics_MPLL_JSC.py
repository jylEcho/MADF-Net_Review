import argparse
import logging
import os
import random
from pydoc import locate
import sys
from datasets.dataset_multiphase import Multiphase_dataset_MPLL
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import importlib
from torch.utils.data import DataLoader
from datasets.dataset_synapse import Synapse_dataset
from trainer.trainer_phase_7metrics_JSC import inference


parser = argparse.ArgumentParser()

parser.add_argument("--volume_path", type=str,
                    default="/images/PublicDataset/Transunet_synaps/project_TransUNet/data/Synapse/",
                    help="root dir for validation volume data")
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--num_classes", type=int, default=3, help="output channel of network")
parser.add_argument("--output_dir", type=str, default="./model_out", help="output dir")
parser.add_argument("--max_iterations", type=int, default=30000)
parser.add_argument("--max_epochs", type=int, default=400)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--is_savenii", action="store_true")
parser.add_argument("--test_save_dir", type=str, default="./model_out/test")
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--base_lr", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--opts", default=None, nargs="+")
parser.add_argument("--list_dir", type=str, default="./lists/lists_liver")
parser.add_argument("--inference", default="inference")
parser.add_argument("--gpu", default="1")
parser.add_argument("--crop_inference", default="random_crop_inference")
parser.add_argument("--test_path", type=str,
                    default="/home/image006/multi-phase/dataset_2d/dataset_v/test/")

# 新增：checkpoint 文件夹
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    required=True,
    help="folder containing .pth checkpoints"
)

# 新增：统一结果 txt
parser.add_argument(
    "--result_txt",
    type=str,
    default="/root/autodl-tmp/test/all_checkpoints_result.txt",
    help="single txt file to save all checkpoint test results"
)

parser.add_argument("--module", help="network module, e.g. networks.DAEFormer.DAEFormer")
parser.add_argument("--cache-mode", type=str, default="part", choices=["no", "full", "part"])
parser.add_argument("--use-checkpoint", action="store_true")
parser.add_argument("--amp-opt-level", type=str, default="O1", choices=["O0", "O1", "O2"])
parser.add_argument("--tag")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--throughput", action="store_true")
parser.add_argument("--model_name", type=str, default="HAformerSpatialFrequency")


def dynamic_import(module_name, function_name):
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function


args = parser.parse_args()

if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

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

    dataset_config = {
        "Synapse": {
            "Dataset": Synapse_dataset,
            "z_spacing": 1,
        },
        "Multiphase": {
            "Dataset": Multiphase_dataset_MPLL,
            "z_spacing": 1,
        },
    }

    dataset_name = args.dataset
    args.Dataset = dataset_config[dataset_name]["Dataset"]
    args.z_spacing = dataset_config[dataset_name]["z_spacing"]
    args.is_pretrain = True

    assert args.crop_inference is not None
    args.crop_inference = dynamic_import("utils.utils_multiphase", args.crop_inference)

    # 建立测试集，只需要建立一次
    db_test = Multiphase_dataset_MPLL(
        base_dir=args.test_path,
        split="test_vol",
        list_dir=args.list_dir,
        img_size=args.img_size
    )

    testloader = DataLoader(
        db_test,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    # 找到 checkpoint_dir 下所有 .pth 文件
    ckpt_files = sorted([
        os.path.join(args.checkpoint_dir, f)
        for f in os.listdir(args.checkpoint_dir)
        if f.endswith(".pth")
    ])

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No .pth files found in {args.checkpoint_dir}")

    # 只建立一个 txt
    os.makedirs(os.path.dirname(args.result_txt), exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=args.result_txt,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        filemode="w"
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("Start testing all checkpoints")
    logging.info(str(args))
    logging.info(f"Found {len(ckpt_files)} checkpoints")
    logging.info(f"Result txt: {args.result_txt}")

    for ckpt_path in ckpt_files:
        snapshot_name = os.path.basename(ckpt_path)

        logging.info("\n" + "=" * 100)
        logging.info(f"Testing checkpoint: {snapshot_name}")
        logging.info(f"Checkpoint path: {ckpt_path}")
        logging.info("=" * 100)

        # 每个 checkpoint 重新初始化模型
        net = locate(args.module)
        net = net(args).cuda()

        checkpoint = torch.load(ckpt_path, map_location="cuda")

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v

        missing_keys, unexpected_keys = net.load_state_dict(new_state_dict, strict=False)

        logging.info("=== 模型加载详细信息 ===")
        logging.info(f"缺失的键数量: {len(missing_keys)}")
        if missing_keys:
            logging.info(f"缺失的键前10个: {missing_keys[:10]}")

        logging.info(f"意外的键数量: {len(unexpected_keys)}")
        if unexpected_keys:
            logging.info(f"意外的键前10个: {unexpected_keys[:10]}")

        if len(missing_keys) > 10:
            logging.info("⚠️ 警告：缺失大量参数，模型可能无法正常工作！")
####################################################################################
        # 是否保存 nii
        # if args.is_savenii:
        #     test_save_path = os.path.join(
        #         os.path.dirname(args.result_txt),
        #         snapshot_name.replace(".pth", "_nii")
        #     )
        #     os.makedirs(test_save_path, exist_ok=True)
        # else:
        #     test_save_path = None

        # logging.info(f"inference_single_volume_style: {args.crop_inference}")
        test_save_path = None
####################################################################################
        # 推理，指标会由 inference 内部 logging.info 写入同一个 txt
        # inference(net, testloader, args, test_save_path)

        # del net
        # torch.cuda.empty_cache()
####################################################################################
        # 推理，接收 6 个指标
        mean_dice, mean_jaccard, mean_hd95, mean_assd, mean_rmsd, mean_voe, mean_rvd = inference(
            net,
            testloader,
            args,
            test_save_path
        )
        
        logging.info(
            "Checkpoint summary: "
            "DSC %.6f "
            "JSC %.6f "
            "HD95 %.6f "
            "ASSD %.6f "
            "RMSD %.6f "
            "VOE %.6f "
            "RVD %.6f"
            % (
                mean_dice,
                mean_jaccard,
                mean_hd95,
                mean_assd,
                mean_rmsd,
                mean_voe,
                mean_rvd,
            )
        )
        
        del net
        torch.cuda.empty_cache()
####################################################################################
    logging.info("\n" + "=" * 100)
    logging.info("All checkpoints testing finished.")
    logging.info("=" * 100)