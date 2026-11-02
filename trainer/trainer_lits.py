import torch
from utils.utils_lits import *
from functools import partial
from monai.losses import DiceLoss, DiceCELoss
from torch.cuda.amp import GradScaler, autocast #native AMP
from optimizers.lr_scheduler import WarmupCosineSchedule,LinearWarmupCosineAnnealingLR
from monai.inferers import sliding_window_inference
import nibabel as nb
import time

start_epoch = 0
weight_decay= 1e-5
scaler = GradScaler()
roi_size = [96,96,96]
inf_size = [96,96,96]# [args.roi_x, args.roi_y, args.roi_x]
val_overlap = 0.5
val_files= ''
amp = 1

def get_dataloader(args):

    train_loader = []
    test_loader = []

    return  train_loader, test_loader

def train_epoch(model, train_loader, optimizer, scaler, \
                                 epoch, loss_func, args):
    train_loss = 0

    return train_loss
def get_scheduler(args, optimizer):
    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )


    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)

    else:
        scheduler = None
    return  scheduler


def val_epoch(model, loader, val_shape_dict, epoch, loss_func,
                                          model_inferer= None, args= None):
    model.eval()
    start_time = time.time()
    run_loss = AverageMeter()
    run_acc = AverageMeter()
    val_loss = 0
    with torch.no_grad():

        for idx, batch_data in enumerate(loader):

            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data['image'], batch_data['label']

            data, target = data.cuda(), target.cuda()

            with autocast(enabled= amp):
                if model_inferer is not None:
                    logits = model_inferer(data)  # another inferer (e.g. sliding window)
                else:
                    logits = model(data)

            loss = loss_func(logits, target)

            logits = torch.softmax(logits, 1).cpu().numpy()
            logits = np.argmax(logits, axis=1).astype(np.uint8)
            target = target.cpu().numpy()[:, 0, :, :, :]

            name = batch_data["image_meta_dict"]['filename_or_obj'][0].split('/')[-1]
            val_shape = val_shape_dict[name]

            pred = resample(logits[0], val_shape)
            y = resample(target[0], val_shape)

            dice_list_sub = []
            for i in range(1, args.num_classes):
                organ_Dice = dice(pred == i, y == i)
                dice_list_sub.append(organ_Dice)
            avg_classes = np.array(dice_list_sub)
            run_acc.update(avg_classes, n=args.batch_size)
            run_loss.update(loss.item(), n=args.batch_size)

            print(
                'Batch mean: Liver: {}, Tumor: {}, all:{}'.format(avg_classes[0], avg_classes[1], np.mean(avg_classes)))
            print('Val {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                    'loss: {:.4f}'.format(run_loss.avg),
                    'acc', run_acc.avg,
                    'acc_avg: {:.4f}'.format(np.mean(run_acc.avg)),
                    'time {:.2f}s'.format(time.time() - start_time))
            start_time = time.time()



    return run_loss.avg, run_acc.avg
def trainer_lits(args, model, output_dir):
    train_loader,test_loader = get_dataloader(args)

    # 这里可以调用超参优化 optuna_run
    # 准备优化器、损失函数、学习率调度器、模型推断器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay= weight_decay)
    loss_func = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-6)
    scheduler = get_scheduler(args, optimizer)
    val_shape_dict = get_valshape(val_files)
    model_inferer = partial(sliding_window_inference, roi_size=inf_size, sw_batch_size=1, predictor= model,
                            overlap= val_overlap, mode='gaussian')

    for epoch in range(start_epoch, args.max_epochs):
        epoch_time = time.time()
        print('Epoch:', epoch)
        train_loss = train_epoch(model, train_loader, optimizer, scaler=scaler, \
                                 epoch=epoch, loss_func=loss_func, args=args)


        print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
      'time {:.2f}s'.format(time.time() - epoch_time))

        val_acc_max = 0
        # val epoch
        if epoch % 10 == 0:
            val_acc = 0

            val_loss, val_acc = val_epoch(model, test_loader, val_shape_dict, epoch=epoch, loss_func=loss_func,
                                          model_inferer=model_inferer, args=args)

            if np.mean(val_acc) > val_acc_max:
                val_acc_max = np.mean(val_acc)
                print('new best ({:.6f} --> {:.6f}). '.format(val_acc_max, np.mean(val_acc)))
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler)

    print('monai_trainer DONE, best  acc', val_acc_max)

    return val_acc_max
