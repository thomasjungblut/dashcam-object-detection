#!/usr/bin/python3
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import (
    check_img_size,
    non_max_suppression,
)
from yolov5.utils.torch_utils import select_device

print(torch.__version__, torch.cuda.is_available())
# yolov5 is terrible at configuring logging correctly, so we disable it altogether
logging.disable(sys.maxsize)

parser = argparse.ArgumentParser(description='Filter videos with objects in them.')
parser.add_argument('-i', '--input', type=str, required=True, help='the input folder containing *.mp4 files')
parser.add_argument('-o', '--output', type=str, required=True,
                    help='the output folder that will contain mp4 files that have objects in them.')
parser.add_argument('-b', '--batch-size', type=int, default=512, required=False,
                    help='how many frames to batch in one prediction, reduce if they do not fit into RAM or VRAM')
parser.add_argument('-w', '--weights', type=str, default='yolov5s.pt', required=False,
                    help='the weights path for a YOLOv5 model')
parser.add_argument('-g', '--ignore', type=str, default='', required=False,
                    help='class names to ignore, comma separated. E.g: car,airplane')
parser.add_argument('-t', '--include', type=str, default='', required=False,
                    help='class names to include, comma separated. E.g: car,airplane.')
parser.add_argument('-c', '--confidence', type=float, default=0.7, required=False,
                    help='a fraction between 0 and 1, where 1.0 is really sure this is the object')
parser.add_argument('-dd', '--device', type=str, default="0", required=False,
                    help="which device to choose, by default the first GPU (0). Can be any number or 'cpu' for CPU")
parser.add_argument('-s', '--img-size', nargs='+', type=int, default=[640], help='inference size w,h')

args = parser.parse_args()
input_path = args.input
output_path = Path(args.output)
weights = args.weights
batch_size = args.batch_size
device = args.device
confidence = args.confidence
imgsz = args.img_size
imgsz *= 2 if len(imgsz) == 1 else 1
ignore_set = set(args.ignore.split(','))
include_set = set(args.include.split(',')) if args.include else None
video_codec = "MP4V"

device = select_device(device)
model = DetectMultiBackend(weights, device=device, data="coco128_classes.yml")
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)


def rename_on_class_match(source_path: str, dtc: set):
    p = Path(source_path)
    dtc = dtc.difference(ignore_set)
    if len(dtc) > 0:
        print("moving file [%s], detected [%s]" % (p.name, ", ".join(dtc)))
        p.rename(output_path.joinpath(p.name))
    else:
        print("file [%s] has no detections, ignoring" % p.name)


def predict_batch(batch) -> set:
    start = time.time()
    stacked = np.stack(batch)
    ix = torch.from_numpy(stacked).to(device).float()

    # we can skip NMS since we don't care about the bounding boxes at all, thus the transfer back of the result
    # from the GPU is only an aggregated set of classes
    pred = model(ix)
    # score can be found at 4, classes start after index 5 (before that are the bb coords)
    predicted_classes = pred[pred[..., 4] > confidence][:, 5:].max(dim=1)[1].unique().cpu()
    s = set(map(lambda x: names[x], predicted_classes.numpy()))
    print("batch with %d images found [%s], took %s" % (len(batch), ", ".join(s), time.time() - start))
    batch.clear()
    return s


dataset = LoadImages(input_path, img_size=imgsz, stride=stride, auto=pt)
last_vid_path = None
batch = []
detected_classes = set()
for path, im, im0s, vid_cap, s in dataset:
    # TODO include set early skip
    # if len(detected_classes.intersection(include_set)) > 0:

    if path != last_vid_path:
        if len(batch) > 0:
            # new video detected, clean the batch from the last video
            detected_classes = detected_classes.union(predict_batch(batch))
            rename_on_class_match(last_vid_path, detected_classes)

        detected_classes.clear()
        last_vid_path = path
        print("processing [%s]..." % path)

    im = im / 255  # normalize between 0-1
    batch.append(im)
    if len(batch) >= batch_size:
        detected_classes = detected_classes.union(predict_batch(batch))

# overflow when len(batch) > 0
if len(batch) > 0:
    detected_classes = detected_classes.union(predict_batch(batch))
    rename_on_class_match(last_vid_path, detected_classes)

print("done")
