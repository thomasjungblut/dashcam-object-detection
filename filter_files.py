#!/usr/bin/python3
import argparse
import glob
import time
from pathlib import Path

import cv2
import torch

print(torch.__version__, torch.cuda.is_available())

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose=False)

parser = argparse.ArgumentParser(description='Filter videos with objects in them.')
parser.add_argument(
    '-i', '--input', type=str, required=True, help='the input folder containing *.mp4 files'
)
parser.add_argument(
    '-o',
    '--output',
    type=str,
    required=True,
    help='the output folder that will contain mp4 files that have objects in them.',
)

parser.add_argument(
    '-b',
    '--batch-size',
    type=int,
    default=512,
    required=False,
    help='how many frames to batch in one prediction, reduce if they do not fit into RAM or VRAM',
)

parser.add_argument(
    '-g',
    '--ignore',
    type=str,
    default='',
    required=False,
    help='class names to ignore, comma separated. E.g: car,airplane',
)

parser.add_argument(
    '-t',
    '--include',
    type=str,
    default='',
    required=False,
    help='class names to include, comma separated. E.g: car,airplane.',
)

parser.add_argument(
    '-c',
    '--confidence',
    type=float,
    default=0.7,
    required=False,
    help='a fraction between 0 and 1, where 1.0 is really sure this is the object',
)


args = parser.parse_args()
input_path = args.input
output_path = Path(args.output)
batch_size = args.batch_size
confidence = args.confidence
ignore_set = set(args.ignore.split(','))
include_set = set(args.ignore.split(','))
video_codec = "MP4V"


def predict_batch(batch) -> set:
    s = set()
    start = time.time()
    result_df = model(batch).pandas()
    for i in range(len(result_df.xyxy)):
        iloc = result_df.xyxy[i]
        iloc = iloc[iloc.confidence > confidence]
        if len(iloc) > 0:
            s = s.union(iloc.name.values)
    batch.clear()
    print("found [%s], took %s" % (", ".join(s), time.time() - start))
    return s


files = glob.glob(input_path + "/*.mp4")
for file in files:
    vid_cap = cv2.VideoCapture(file)
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    batch = []
    detected_classes = set()
    while True:
        more_frames, img = vid_cap.read()
        if not more_frames:
            break

        batch.append(img)
        if len(batch) >= batch_size:
            detected_classes = detected_classes.union(predict_batch(batch))
            # short circuit if we're looking for a specific class that we've found already
            if len(detected_classes.intersection(include_set)) > 0:
                break

    vid_cap.release()

    if len(batch) > 0:
        detected_classes = detected_classes.union(predict_batch(batch))

    detected_classes = detected_classes.difference(ignore_set)
    if len(detected_classes) > 0:
        p = Path(file)
        p.rename(output_path.joinpath(p.name))
        print("moving file %s, detected [%s]" % (p.name, ", ".join(detected_classes)))
