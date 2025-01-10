from pathlib import Path
import random
import cv2
import shutil, yaml

import torch

from datasets import LoadImages
from models import Darknet
from utils import non_max_suppression, plot_one_box, scale_coords


def detect(opt):

    # ================================================
    # Initialize
    # ================================================
    # setting the device, then directly using .cuda() to use gpu calculate.
    torch.cuda.set_device(opt["gpu_index"])

    out_dir = Path(opt["out_dir"])
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Darknet(opt["cfg"])

    model.load_state_dict(torch.load(opt["weights"], weights_only=False))
    model.cuda().eval()

    dataset = LoadImages(opt["source"])

    with open("settings/names.yaml", 'r') as f:
        names = yaml.safe_load(f.read())["names"]

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).cuda().float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        pred = model(img)[0]

        pred = non_max_suppression(pred, opt["conf_thres"], opt["iou_thres"], multi_label=False)

        for i, det in enumerate(pred):
            p, s, im0 = path, "", im0s
            save_path = str(out_dir / Path(p).name)
            s += "%gx%g " % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += "%g %ss, " % (n, names[int(c)])
                
                for *xyxy, conf, cls in reversed(det):
                    label = "%s %.2f" % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            print("%s Done." % s)

            cv2.imwrite(save_path, im0)


if __name__ == '__main__':
    with open("settings/detect.yaml") as f:
        opt = yaml.safe_load(f.read())

    detect(opt)
    