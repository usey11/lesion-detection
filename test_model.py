import os
import sys
import random
import math
import re
import time
import numpy as np

import pandas as pd

from tqdm import tqdm
from moles import MolesConfig
from moles import MolesDataset

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from moles import MolesConfig
from moles import MolesDataset
from moles import BalancedDataset
from moles import MolesDatasetFast

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
metrics_file_path = os.path.join(MODEL_DIR, "metrics")
#from mrcnn.config import Config
#from mrcnn import utils
#import mrcnn.model as modellib
#from mrcnn import visualize
#from mrcnn.model import log

# Inference Configuration
class InferenceConfig(MolesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

def eval_model(model, dataset):
    image_ids = dataset.image_ids

    APs = []
    Ps = []
    Rs = []
    preds = []
    gts = []

    for image_id in tqdm(image_ids):
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, inference_config,
                                image_id, use_mini_mask=False)

        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
    
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, [gt_class_id], gt_mask,
                            r["rois"], r["class_ids"], r["scores"], r['masks'])
        
        APs.append(AP)
        Ps.append(precisions)
        Rs.append(recalls)
        gts.append(gt_class_id[0])
        if len(r["scores"]) != 0:
            preds.append(r["class_ids"][r["scores"].argmax()])
        else:
            preds.append(-1)
    
    return APs, preds, gts

def main(argv):
    if len(argv) == 0:
        print("Usage: python test_model.py model_paths")
        quit()

    # Load in Datasets
    #dataset_train = MolesDataset()
    #dataset_train.load_moles("../ISIC-Archive-Downloader/Data/train")
    #dataset_train.prepare()

    dataset_val = MolesDatasetFast()
    dataset_val.load_moles("../ISIC-Archive-Downloader/Data/val")
    dataset_val.prepare()

    print(len(dataset_val.image_ids))


    model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

    metrics_file = None
    if os.path.isfile(metrics_file_path):
        metrics_file = open(metrics_file_path, "a+")
    else:
        metrics_file = open(metrics_file_path, "a+")
        metrics_file.write("file, missing_predictions, mAP, accuracy")

    
    model_files = []
    for arg in argv:
        # If arg is a directory get all files in directory with .h5 extension
        if os.path.isdir(arg):
            path_files = os.listdir(arg)

            for pf in path_files:
                if pf[-3:] == ".h5":
                    model_files.append(os.path.join(arg, pf))
        elif os.path.isfile(arg) and arg[-3:] == ".h5":
            model_files.append(arg)
        else:
            print("File: " + arg, " doesn't exist.")
            continue
    for f in model_files:
        if not os.path.isfile(f):
            print("File: " + f, " doesn't exist.")
            continue

        model_path  = f
        print("Evaluating model: " + model_path)

        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)

        print("Evaluating on val set")

        (APs, preds, gts) = eval_model(model, dataset_val)

        df = pd.DataFrame({"gts": gts, "preds": preds})
        
        print(df.head())

        total_preds = len(df)

        missing_preds = len(df[df["preds"] == -1])
        missing_percentage = (missing_preds/total_preds) * 100

        correct_preds = len(df[df["gts"] == df["preds"]])
        accuracy = (correct_preds/total_preds) * 100

        mAP = np.mean(APs)
        print("Missing predictions: {}, ({:.2f}%)".format(missing_preds, missing_percentage))
        print("mAP: ", mAP)
        print("Accuracy: {}%".format(accuracy))
        print()

        # Write to metrics file
        metrics_file.write("{},{},{},{}\n".format(model_path, missing_percentage, mAP, accuracy))



if __name__ == "__main__":
   main(sys.argv[1:])