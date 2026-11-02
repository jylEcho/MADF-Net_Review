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

# from utils.utils_multiphase import  merge_test_single_volume, increment_test_round
from utils.utils_multiphaseV1 import  merge_test_single_volume

from datasets.dataset_multiphase import Multiphase_dataset
import matplotlib.pyplot as plt
import pandas as pd
import datetime

def inference(model, testloader, args, test_save_path=None):
    model.eval()
    metric_list = 0.0  # all cases

    with torch.no_grad():
    # loop all testdata
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image_V"].size()[2:]
            image_v,image_a,image_d, label, case_name, box, case_box = sampled_batch["image_V"],sampled_batch["image_A"],sampled_batch["image_D"], sampled_batch["label"], sampled_batch['case_name'][0], \
                sampled_batch['box'], sampled_batch['case_box']

            metric_i = merge_test_single_volume(image_v,image_a,image_d, label, box, case_box, model, classes=args.num_classes,
                                                crop_inference=args.crop_inference, patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing,args=args)
            # increment_test_round()
#####################################################################################################            
            metric_list += np.array(metric_i)
            # logging.info(' idx %d case %s mean_dice %f mean_hd95 %f' %
            # (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            logging.info(
                ' idx %d case %s mean_dice %f mean_jaccard %f mean_hd95 %f mean_assd %f' %
                (
                    i_batch,
                    case_name,
                    np.mean(metric_i, axis=0)[0],
                    np.mean(metric_i, axis=0)[1],
                    np.mean(metric_i, axis=0)[2],
                    np.mean(metric_i, axis=0)[3],
                )
            )
#####################################################################################################            
    metric_list = metric_list / len(testloader.dataset)
#####################################################################################################            
    # for i in range(1, args.num_classes):
    #     logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))

    # performance = np.mean(metric_list, axis=0)[0]
    # mean_hd95 = np.mean(metric_list, axis=0)[1]

    # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))

    # torch.cuda.empty_cache()
    # return performance, mean_hd95
    for i in range(1, args.num_classes):
        logging.info(
            'Mean class %d mean_dice %f mean_jaccard %f mean_hd95 %f mean_assd %f' %
            (
                i,
                metric_list[i - 1][0],
                metric_list[i - 1][1],
                metric_list[i - 1][2],
                metric_list[i - 1][3],
            )
        )
    
    performance = np.mean(metric_list, axis=0)[0]
    mean_jaccard = np.mean(metric_list, axis=0)[1]
    mean_hd95 = np.mean(metric_list, axis=0)[2]
    mean_assd = np.mean(metric_list, axis=0)[3]
    
    logging.info(
        'Testing performance in best val model: mean_dice : %f mean_jaccard : %f mean_hd95 : %f mean_assd : %f' %
        (performance, mean_jaccard, mean_hd95, mean_assd)
    )
    
    torch.cuda.empty_cache()
    return performance, mean_jaccard, mean_hd95, mean_assd
    
#####################################################################################################            

def plot_result(dice, h, snapshot_path, args):
    dict = {'mean_dice': dice, 'mean_hd95': h}
    df = pd.DataFrame(dict)
    plt.figure(0)
    df['mean_dice'].plot()
    resolution_value = 1200
    plt.title('Mean Dice')
    date_and_time = datetime.datetime.now()
    filename = f'{args.model_name}_' + str(date_and_time) + 'dice' + '.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(1)
    df['mean_hd95'].plot()
    plt.title('Mean hd95')
    filename = f'{args.model_name}_' + str(date_and_time) + 'hd95' + '.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    # save csv
    filename = f'{args.model_name}_' + str(date_and_time) + 'results' + '.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep='\t')


def trainer_multiphase(args, model, snapshot_path):
    os.makedirs(os.path.join(snapshot_path, 'test_' + args.model_name), exist_ok=True)

    test_save_path = os.path.join(snapshot_path, 'test') if args.is_savenii else None

    base_lr = args.base_lr
    num_classes = args.num_classes
    if args.model_name == 'Fusion':
        batch_size = args.batch_size #15
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

    print("batch_size:................",batch_size)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    db_test = Multiphase_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir,
                                 img_size=args.img_size)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    model.cuda()
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    else:
        model = model.cuda()

    model.train()

    weights = torch.tensor([0.3, 0.8, 4.5]) # 加权交叉熵
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = weights.to(device)
    ce_loss = CrossEntropyLoss(weight=weights)
    dice_loss = DiceLoss(num_classes)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs // 2, eta_min=1e-6)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    logging.info("_________________________loss ce weights: {}  ".format(weights))
    logging.info("__________________________________tag: {}  ".format(args.tag))
    best_performance = 0.0
    best_epoch = 0
    iterator = tqdm(range(max_epoch), ncols=70)
    dice_ = []
    hd95_ = []

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch_v, image_batch_a, image_batch_d, label_batch = sampled_batch['image_V'], \
                                                 sampled_batch['image_A'], sampled_batch['image_D'], sampled_batch['label']
            # print("data shape---------", image_batch.shape, label_batch.shape)
            image_batch_v, image_batch_a, image_batch_d, label_batch = image_batch_v.cuda(),\
                                                image_batch_a.cuda(), image_batch_d.cuda(),label_batch.squeeze(1).cuda()
            # pareseg
            if args.model_name == 'Pareseg':
                outputs = model(image_batch_v, image_batch_a)
            else:
                outputs = model(image_batch_a, image_batch_v, image_batch_d)
            # outputs = F.interpolate(outputs, size=label_batch.shape[1:], mode='bilinear', align_corners=False)
            loss_ce = ce_loss(outputs.float(), label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch,softmax=True)
            loss = loss_ce*0.7 +  loss_dice*0.3
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
        if (epoch_num+1) % eval_interval == 0:
            filename = f'{args.model_name}_{args.dataset}_epoch_{epoch_num}_lr_{args.base_lr}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save({
             'iteration': iter_num,
             'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict(),
            'lr': lr_},
                       save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
            # logging.info("skip save model")
            logging.info("*" * 20)
            logging.info(f"Running Inference after epoch {epoch_num}")
            print(f"Epoch {epoch_num}")
            
            # increment_test_round()
            
            mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
            dice_.append(mean_dice)
            hd95_.append(mean_hd95)
            model.train()

        if epoch_num >= max_epoch - 1:
            # filename = f'{args.model_name}_{args.dataset}_epoch_{epoch_num}.pth'
            # save_mode_path = os.path.join(snapshot_path, filename)
            # torch.save({
            #     'iteration': iter_num,
            #     'state_dict': model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     'lr': lr_},
            #     save_mode_path)
            logging.info("skip save model")

            # if not (epoch_num + 1) % args.eval_interval == 0:
            logging.info("*" * 20)
            logging.info(f"Running Inference after epoch {epoch_num} (Last Epoch)")
            print(f"Epoch {epoch_num}, Last Epcoh")
            mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
            dice_.append(mean_dice)
            hd95_.append(mean_hd95)
            model.train()

            iterator.close()
            break

    plot_result(dice_, hd95_, snapshot_path, args)
    writer.close()
    return "Training Finished!"