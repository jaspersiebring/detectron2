# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
from pathlib import Path
import time
import cv2
import json
import tqdm
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo

# Constants
WINDOW_NAME = "Ambulance Detection"


def setup_meta(cfg):
    # Temporary fix, need to retrain with 1 class
    #from detectron2.data import MetadataCatalog
    #dataset_name = 'toy_ambulance'
    #MetadataCatalog.get(dataset_name).set(thing_classes=['ambulance'])
    #cfg.DATASETS.TEST = (dataset_name,)
    cfg.DATASETS.TEST = ('coco_2017_train',)
    cfg.freeze() #added from setup_cfg
    return cfg

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    #cfg.freeze() moved to setup_meta
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default=r"/home/jsieb/Downloads/repos/drone2go/scripts/retinanet_ambulance.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()

    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    cfg = setup_meta(cfg)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        json_lines = []
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation

            img = read_image(path, format="BGR")
            start_time = time.time()

            predictions, visualized_output = demo.run_on_image(img)

            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                json_name = 'predictions.json'

                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                    json_path = os.path.join(args.output, json_name)
                else:
                    json_path = os.path.join(os.path.dirname(args.output), json_name)
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output

                visualized_output.save(out_filename)

                ####
                cpu_instances = predictions['instances'].to("cpu")
                class_filter = cpu_instances.pred_classes.numpy() == 0

                if np.any(class_filter):
                    cpu_instances = cpu_instances[class_filter]
                    box_centers = cpu_instances.pred_boxes.get_centers().numpy().tolist()
                    box_scores = cpu_instances.scores.numpy().tolist()
                    box_classes = cpu_instances.pred_classes.numpy().tolist()

                else:
                    box_centers = []
                    box_scores = []
                    box_classes = []

                pred_json = {'input' : path, 'box_centers' : box_centers, 'box_scores' : box_scores, 'box_classes' : box_classes}
                json_lines.append(pred_json)

        json.dump(json_lines,open(json_path, 'a'))




