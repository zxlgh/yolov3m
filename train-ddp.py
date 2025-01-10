import yaml
from tqdm import tqdm

from models import Darknet, YOLOLayer
from datasets import LoadImagesAndLabels
from utils import *
from eval import evaluate

import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
 
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    

def train(rank, world_size, opt):

    
    init_seeds()
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # ================================================
    # dataset
    # ================================================
    
    train_set = LoadImagesAndLabels(opt["train_path"],
                                    img_size=opt["img_size"],
                                    batch_size=opt["batch_size"],
                                    augment=True,
                                    hyp=opt)
    train_sampler = DistributedSampler(train_set) if rank >= 0 else None
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=opt["batch_size"],
                                               num_workers=opt["num_workers"],
                                               shuffle=(train_sampler is None),
                                               pin_memory=True,
                                               collate_fn=train_set.collate_fn,
                                               sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(LoadImagesAndLabels(opt["val_path"],
                                                                 img_size=opt['img_size'],
                                                                 batch_size=2*opt["batch_size"],
                                                                 hyp=opt,
                                                                 rect=True),
                                             batch_size=opt["batch_size"],
                                             num_workers=opt["num_workers"],
                                             pin_memory=True,
                                             collate_fn=train_set.collate_fn)

    # ================================================
    # model
    # ================================================
    model = Darknet(opt["cfg"])
    
    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    optimizer = optim.SGD(pg0, lr=opt['lr0'], momentum=opt['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': opt['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    if opt["weights"].endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(opt["weights"], weights_only=False))
    elif len(opt["weights"]) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        load_darknet_weights(model, opt["weights"])

    if opt["freeze_layers"]:
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if isinstance(module, YOLOLayer)]
        freeze_layer_indices = [x for x in range(len(model.module_list)) if
                                (x not in output_layer_indices) and
                                (x - 1 not in output_layer_indices)]
        for idx in freeze_layer_indices:
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / opt["epochs"])) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Model parameters
    model.nc = 80  # attach number of classes to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.hyp = opt
    model.cuda()

    if rank >= 0:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    ema = ModelEMA(model)

    nb = len(train_loader)
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(80)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    best_fitness = 0.0
    for epoch in range(opt["epochs"]):
        train_sampler.set_epoch(epoch)
        model.train()
        mloss = torch.zeros(4).cuda()
        if rank == 0:
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(enumerate(train_loader), total=nb)
        else:
            pbar = enumerate(train_loader)
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.cuda().float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.cuda()

            # Burn-in
            if ni <= n_burn:
                xi = [0, n_burn]  # x interp
                model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    x['weight_decay'] = np.interp(ni, xi, [0.0, opt['weight_decay'] if j == 1 else 0.0])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, opt['momentum']])

            pred = model(imgs)

            loss, loss_items = compute_loss(pred, targets, model)

            # Backward
            loss *= opt["batch_size"] * world_size / 64  # scale loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            ema.update(model)

            # Print
            if rank == 0:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch+1, opt["epochs"]), mem, *mloss, len(targets), opt["img_size"])
                pbar.set_description(s)
        
        scheduler.step()
        ema.update_attr(model)
        if rank == 0:
            results, maps = evaluate(opt, ema.ema, val_loader)
            with open('results.txt', 'a') as f:
                f.write(s + '%10.3g' * 7 % results + '\n') # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
            if fi > best_fitness:
                best_fitness = fi
                torch.save(ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(), opt["save_model"])

    dist.destroy_process_group()



if __name__ == '__main__':
    with open('settings.yaml', 'r') as f:
        opt = yaml.safe_load(f)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(train, args=(WORLD_SIZE, opt), nprocs=WORLD_SIZE, join=True)
    # train(opt)
    