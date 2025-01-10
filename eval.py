import sys
from tqdm import tqdm

import torch
import yaml

from models import Darknet
from datasets import LoadImagesAndLabels
from utils import *


def evaluate(opt, model=None, dataloader=None, augment=False):
    if model is None:
        is_training = False

        model = Darknet(opt["cfg"])
        weights = opt["weights"]
        if weights.endswith("pt"):
            model.load_state_dict(torch.load(weights, weights_only=False))
        else:
            load_darknet_weights(model, weights)
        model.fuse()
        model.cuda()

    else:
        is_training = True
    
    nc = opt["classes"]
    # names = opt["names"]
    iouv = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    if dataloader is None:
        dataset = LoadImagesAndLabels(opt["val_path"], batch_size=opt["batch_size"], rect=True, pad=0.5)
        dataloader = torch.utils.data.DataLoade(dataset,
                                                batch_size=2 * opt["batch_size"],
                                                num_workers=8,
                                                pin_memory=True,
                                                collate_fn=dataset.collate_fn)
    seen = 0
    model.eval()
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3).cuda()
    jdict, stats, ap, ap_class = [], [], [], []

    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s, total=len(dataloader))):
        imgs = imgs.cuda().float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.cuda()
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).cuda()

        with torch.no_grad():
            inf_out, train_out = model(imgs, augment=augment)  # inference and training outputs

            # Compute loss
            if is_training:  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3]  # GIoU, obj, cls
            output = non_max_suppression(inf_out, conf_thres=opt["conf_thres"], iou_thres=opt["iou_thres"], multi_label=opt["multi_label"])

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool).cuda()
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        # print(type(maps[c]), type(ap[i]))
        maps[c] = ap[i][0]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    with open('settings.yaml', 'r') as f:
        opt = yaml.safe_load(f)

    torch.cuda.set_device(1)

    val_set = LoadImagesAndLabels(opt["val_path"],
                                img_size=opt['img_size'],
                                batch_size=2*opt["batch_size"],
                                hyp=opt,
                                rect=True)

    val_loader = torch.utils.data.DataLoader(val_set,
                                            batch_size=opt["batch_size"],
                                            num_workers=opt["num_workers"],
                                            pin_memory=True,
                                            collate_fn=val_set.collate_fn)
    
    evaluate(opt, model=None, dataloader=val_loader)