import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import DiceLoss
from torchvision import transforms
from utils.utils_singlephase_7metrics_GY import  merge_test_single_volume
from datasets.dataset_multiphase import Multiphase_dataset_MPLL as Multiphase_dataset
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import shutil


METRIC_NAMES = ['dice', 'jaccard', 'hd95', 'assd', 'rmsd', 'voe', 'rvd']


def _single_phase_key(args):
    phase = getattr(args, "single_phase", "PV").upper()
    aliases = {"A": "ART", "V": "PV", "D": "DL"}
    return aliases.get(phase, phase)


def _select_single_phase(image_v, image_a, image_d, args):
    phase = _single_phase_key(args)
    if phase == "ART":
        return image_a
    if phase == "DL":
        return image_d
    return image_v

def inference(model, testloader, args, test_save_path=None):
    model.eval()

    all_metrics = []
    case_rows = []

    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image_v = sampled_batch["image_V"]
            image_a = sampled_batch["image_A"]
            image_d = sampled_batch["image_D"]
            label = sampled_batch["label"]
            case_name = sampled_batch["case_name"][0]
            box = sampled_batch["box"]
            case_box = sampled_batch["case_box"]

            metric_i = merge_test_single_volume(
                image_v,
                image_a,
                image_d,
                label,
                box,
                case_box,
                model,
                classes=args.num_classes,
                crop_inference=args.crop_inference,
                patch_size=[args.img_size, args.img_size],
                test_save_path=test_save_path,
                case=case_name,
                z_spacing=args.z_spacing,
                args=args
            )

            metric_i = np.array(metric_i, dtype=np.float64)
            all_metrics.append(metric_i)

            case_mean = np.nanmean(metric_i, axis=0)
            lesion_metric = metric_i[min(1, len(metric_i) - 1)]

            row = {
                'case': case_name,
                'model': args.model_name,
                'phase': _single_phase_key(args),
                'seed': args.seed,
            }
            for idx, metric_name in enumerate(METRIC_NAMES):
                row[f'mean_{metric_name}'] = case_mean[idx]
                row[f'lesion_{metric_name}'] = lesion_metric[idx]
            case_rows.append(row)

            logging.info(
                ' idx %d case %s '
                'mean_dice %f '
                'mean_jaccard %f '
                'mean_hd95 %f '
                'mean_assd %f '
                'mean_rmsd %f '
                'mean_voe %f '
                'mean_rvd %f'
                % (
                    i_batch,
                    case_name,
                    case_mean[0],
                    case_mean[1],
                    case_mean[2],
                    case_mean[3],
                    case_mean[4],
                    case_mean[5],
                    case_mean[6],
                )
            )

    all_metrics = np.array(all_metrics, dtype=np.float64)

    metric_list = np.nanmean(all_metrics, axis=0)

    for i in range(1, args.num_classes):
        logging.info(
            'Mean class %d '
            'mean_dice %f '
            'mean_jaccard %f '
            'mean_hd95 %f '
            'mean_assd %f '
            'mean_rmsd %f '
            'mean_voe %f '
            'mean_rvd %f'
            % (
                i,
                metric_list[i - 1][0],
                metric_list[i - 1][1],
                metric_list[i - 1][2],
                metric_list[i - 1][3],
                metric_list[i - 1][4],
                metric_list[i - 1][5],
                metric_list[i - 1][6],
            )
        )

    performance = np.nanmean(metric_list, axis=0)[0]
    mean_jaccard = np.nanmean(metric_list, axis=0)[1]
    mean_hd95 = np.nanmean(metric_list, axis=0)[2]
    mean_assd = np.nanmean(metric_list, axis=0)[3]
    mean_rmsd = np.nanmean(metric_list, axis=0)[4]
    mean_voe = np.nanmean(metric_list, axis=0)[5]
    mean_rvd = np.nanmean(metric_list, axis=0)[6]

    summary = {
        'model': args.model_name,
        'phase': _single_phase_key(args),
        'seed': args.seed,
        'n_cases': len(case_rows),
        'mean_dice': performance,
        'mean_jaccard': mean_jaccard,
        'mean_hd95': mean_hd95,
        'mean_assd': mean_assd,
        'mean_rmsd': mean_rmsd,
        'mean_voe': mean_voe,
        'mean_rvd': mean_rvd,
    }

    if case_rows:
        case_df = pd.DataFrame(case_rows)
        case_csv = os.path.join(
            args.output_dir,
            f"case_metrics_{args.model_name}_{_single_phase_key(args)}_seed{args.seed}.csv",
        )
        case_df.to_csv(case_csv, index=False)
        worst_k = max(1, int(np.ceil(len(case_df) * 0.1)))
        summary['worst10_mean_dice'] = case_df.nsmallest(worst_k, 'mean_dice')['mean_dice'].mean()
        summary['worst10_lesion_dice'] = case_df.nsmallest(worst_k, 'lesion_dice')['lesion_dice'].mean()
        summary['case_metrics_csv'] = case_csv

        for scope in ('mean', 'lesion'):
            for metric_name in METRIC_NAMES:
                col = f'{scope}_{metric_name}'
                summary[f'case_{col}_mean'] = case_df[col].mean()
                summary[f'case_{col}_std'] = case_df[col].std(ddof=1) if len(case_df) > 1 else 0.0
    else:
        summary['worst10_mean_dice'] = 0.0
        summary['worst10_lesion_dice'] = 0.0
        summary['case_metrics_csv'] = ''

    summary_csv = os.path.join(
        args.output_dir,
        f"eval_summary_{args.model_name}_{_single_phase_key(args)}_seed{args.seed}.csv",
    )
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    summary['summary_csv'] = summary_csv

    logging.info(
        'Testing performance in best val model: '
        'mean_dice : %f '
        'mean_jaccard : %f '
        'mean_hd95 : %f '
        'mean_assd : %f '
        'mean_rmsd : %f '
        'mean_voe : %f '
        'mean_rvd : %f'
        % (
            performance,
            mean_jaccard,
            mean_hd95,
            mean_assd,
            mean_rmsd,
            mean_voe,
            mean_rvd
        )
    )
    logging.info(
        'Case std / worst10: '
        'DSC_std %f HD95_std %f ASSD_std %f Worst10MeanDice %f Worst10LesionDice %f'
        % (
            summary.get('case_mean_dice_std', 0.0),
            summary.get('case_mean_hd95_std', 0.0),
            summary.get('case_mean_assd_std', 0.0),
            summary.get('worst10_mean_dice', 0.0),
            summary.get('worst10_lesion_dice', 0.0),
        )
    )
    logging.info('Saved case metrics to %s', summary.get('case_metrics_csv', ''))
    logging.info('Saved eval summary to %s', summary_csv)

    torch.cuda.empty_cache()

    return summary


def plot_result(metric_history, snapshot_path, args):
    """Save metric history to CSV and plot each metric curve."""
    metric_names = [
        'mean_dice',
        'mean_jaccard',
        'mean_hd95',
        'mean_assd',
        'mean_rmsd',
        'mean_voe',
        'mean_rvd',
    ]
    df = pd.DataFrame(metric_history, columns=metric_names)
    date_and_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    resolution_value = 1200

    for metric_name in metric_names:
        plt.figure()
        df[metric_name].plot()
        plt.title(metric_name)
        filename = f'{args.model_name}_{date_and_time}_{metric_name}.png'
        save_mode_path = os.path.join(snapshot_path, filename)
        plt.savefig(save_mode_path, format='png', dpi=resolution_value)
        plt.close()

    filename = f'{args.model_name}_{date_and_time}_7metrics_results.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep='\t', index=False)


def trainer_single(args, model, snapshot_path):
    os.makedirs(os.path.join(snapshot_path, 'test_' + args.model_name), exist_ok=True)

    test_save_path = os.path.join(snapshot_path, 'test') if args.is_savenii else None

    base_lr = args.base_lr
    num_classes = args.num_classes
    if args.model_name == 'Fusion':
        batch_size = 9 #15
    else:
        batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  #预处理还需要这个吗，应该不需要了---需要
    ])
    y_transforms = transforms.ToTensor()

    db_train = Multiphase_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                  img_size=args.img_size,
                                  norm_x_transform=x_transforms, norm_y_transform=y_transforms,crop_train=args.crop_train)

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    print('batch_size:............',batch_size)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    db_test = Multiphase_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir,
                                 img_size=args.img_size)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    model = model.cuda()
    if args.n_gpu > 1:

        model = nn.DataParallel(model)
    model.train()

    weights = torch.tensor([0.2, 0.5, 4.5])  # 加权交叉熵
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = weights.to(device)
    ce_loss = CrossEntropyLoss(weight=weights)
    dice_loss = DiceLoss(num_classes)
    ##########  optim ########################################
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    logging.info("__________________________________loss ce weights: {}  ".format(weights))
    logging.info("__________________________________tag: {}  ".format(args.tag))
    best_performance = 0.0
    best_path = None
    iterator = tqdm(range(max_epoch), ncols=70)
    metric_history = []

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch_v, image_batch_a, image_batch_d, label_batch = sampled_batch['image_V'], \
                                                 sampled_batch['image_A'], sampled_batch['image_D'], sampled_batch['label']
            # print("data shape---------", image_batch.shape, label_batch.shape)
            image_batch_v, image_batch_a, image_batch_d, label_batch = image_batch_v.cuda(),\
                                                image_batch_a.cuda(), image_batch_d.cuda(),label_batch.squeeze(1).cuda()
            image_batch_single = _select_single_phase(image_batch_v, image_batch_a, image_batch_d, args)
            # pareseg
            if args.model_name == 'Pareseg':
                outputs = model(image_batch_a, image_batch_v, image_batch_d)
            elif args.model_name == 'Acftransunet' or args.model_name == 'Daeformer':
                outputs = model(image_batch_single)
            elif args.model_name == 'ASSNet':
                outputs = model(image_batch_single)
            elif args.model_name == 'Transunet':
                outputs = model(image_batch_single)
            elif args.model_name == 'SwinUnet':
                outputs = model(image_batch_single)
            elif args.model_name == 'KiU_Net':
                outputs = model(image_batch_single)
            elif args.model_name == 'NestedUNet':
                outputs = model(image_batch_single)
            else:
                outputs = model(image_batch_single)
            # outputs = F.interpolate(outputs, size=label_batch.shape[1:], mode='bilinear', align_corners=False)
            loss_ce = ce_loss(outputs.float(), label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch,softmax=True)
            loss = loss_ce +  loss_dice
            # loss = loss_dice*1
            # print("loss-----------", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            # lr_ = scheduler.get_lr()[0]
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, lr_: %f' % (
            iter_num, loss.item(), loss_ce.item(), loss_dice.item(), lr_))

            # if iter_num % 20 == 0:
            #     image = image_batch_v[1, 0:1, :, :]
            #     image = (image - image.min()) / (image.max() - image.min())
            #     writer.add_image('home/image006/multi-phase/multiphase-liverlesion/model_out/train/Image', image,
            #                      iter_num)
            #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('home/image006/multi-phase/multiphase-liverlesion/model_out/train/Prediction',
            #                      outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('home/image006/multi-phase/multiphase-liverlesion/model_out/train/GroundTruth', labs,
            #                      iter_num)

        # scheduler.step()
        # Test
        eval_interval = args.eval_interval
        # if epoch_num >= int(max_epoch / 2) and (epoch_num + 1) % eval_interval == 0: epoch_num>=40 and
        if  epoch_num % eval_interval == 0 and epoch_num != 0:
            filename = f'{args.model_name}_{args.dataset}_epoch_{epoch_num}_lr_{args.base_lr}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            # torch.save({
            #  'iteration': iter_num,
            #  'state_dict': model.state_dict(),
            #  'optimizer' : optimizer.state_dict(),
            # 'lr': lr_},
            #            save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            logging.info("*" * 20)
            logging.info(f"Running Inference after epoch {epoch_num}")
            print(f"Epoch {epoch_num}")
            # mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
            eval_summary = inference(
                model,
                testloader,
                args,
                test_save_path=test_save_path
            )
            mean_dice = eval_summary['mean_dice']
            mean_jaccard = eval_summary['mean_jaccard']
            mean_hd95 = eval_summary['mean_hd95']
            mean_assd = eval_summary['mean_assd']
            mean_rmsd = eval_summary['mean_rmsd']
            mean_voe = eval_summary['mean_voe']
            mean_rvd = eval_summary['mean_rvd']
            metric_history.append([mean_dice, mean_jaccard, mean_hd95, mean_assd, mean_rmsd, mean_voe, mean_rvd])
            if mean_dice > best_performance:
                if best_path and os.path.exists(best_path):
                    os.remove(best_path)
                best_performance = mean_dice
                best_path = os.path.join(
                    snapshot_path,
                    f"best_{args.model_name}_{_single_phase_key(args)}_seed{args.seed}_epoch_{epoch_num}_dice_{mean_dice:.6f}.pth",
                )
                torch.save({
                    'iteration': iter_num,
                    'epoch': epoch_num,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr': lr_,
                    **eval_summary,
                }, best_path)
                best_summary_csv = os.path.join(
                    snapshot_path,
                    f"best_summary_{args.model_name}_{_single_phase_key(args)}_seed{args.seed}.csv",
                )
                pd.DataFrame([eval_summary]).to_csv(best_summary_csv, index=False)
                if eval_summary.get('case_metrics_csv'):
                    best_case_csv = os.path.join(
                        snapshot_path,
                        f"best_case_metrics_{args.model_name}_{_single_phase_key(args)}_seed{args.seed}.csv",
                    )
                    shutil.copyfile(eval_summary['case_metrics_csv'], best_case_csv)
                logging.info("save best model to %s", best_path)
            model.train()

        if epoch_num >= max_epoch - 1:
            filename = f'{args.model_name}_{args.dataset}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save({
                'iteration': iter_num,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr': lr_},
                save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            # if not (epoch_num + 1) % args.eval_interval == 0:
            logging.info("*" * 20)
            logging.info(f"Running Inference after epoch {epoch_num} (Last Epoch)")
            print(f"Epoch {epoch_num}, Last Epcoh")
            # mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
            eval_summary = inference(
                model,
                testloader,
                args,
                test_save_path=test_save_path
            )
            mean_dice = eval_summary['mean_dice']
            mean_jaccard = eval_summary['mean_jaccard']
            mean_hd95 = eval_summary['mean_hd95']
            mean_assd = eval_summary['mean_assd']
            mean_rmsd = eval_summary['mean_rmsd']
            mean_voe = eval_summary['mean_voe']
            mean_rvd = eval_summary['mean_rvd']
            metric_history.append([mean_dice, mean_jaccard, mean_hd95, mean_assd, mean_rmsd, mean_voe, mean_rvd])
            if mean_dice > best_performance:
                if best_path and os.path.exists(best_path):
                    os.remove(best_path)
                best_performance = mean_dice
                best_path = os.path.join(
                    snapshot_path,
                    f"best_{args.model_name}_{_single_phase_key(args)}_seed{args.seed}_epoch_{epoch_num}_dice_{mean_dice:.6f}.pth",
                )
                torch.save({
                    'iteration': iter_num,
                    'epoch': epoch_num,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr': lr_,
                    **eval_summary,
                }, best_path)
                best_summary_csv = os.path.join(
                    snapshot_path,
                    f"best_summary_{args.model_name}_{_single_phase_key(args)}_seed{args.seed}.csv",
                )
                pd.DataFrame([eval_summary]).to_csv(best_summary_csv, index=False)
                if eval_summary.get('case_metrics_csv'):
                    best_case_csv = os.path.join(
                        snapshot_path,
                        f"best_case_metrics_{args.model_name}_{_single_phase_key(args)}_seed{args.seed}.csv",
                    )
                    shutil.copyfile(eval_summary['case_metrics_csv'], best_case_csv)
                logging.info("save best model to %s", best_path)
            model.train()

            iterator.close()
            break

    plot_result(metric_history, snapshot_path, args)
    writer.close()
    return "Training Finished!"
