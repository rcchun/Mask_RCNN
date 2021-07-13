import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import find_contours
from matplotlib.patches import Polygon
import csv
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import save_result_box, save_result_mask
import mrcnn.model as modellib
from mrcnn.model import log

from samples.balloon import train

############################################################
#  Using GPU 
############################################################

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "samples")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases

# TODO: update this path
log_dir = "/balloon/logs/classification20200806T1744"
model_file = "/mask_rcnn_classification_0049.h5"
BALLON_WEIGHTS_PATH = MODEL_DIR + log_dir + model_file
IMAGE_DIR = os.path.join(ROOT_DIR, "datasets/sample_800_all/test")
config = train.BalloonConfig()
DATA_DIR = os.path.join(ROOT_DIR, "datasets/sample_800_all")

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
    
def extract_image_id(image_ids):
    image_id_list = []
    for i in image_ids:
        image_name = dataset.image_info[i]['id']

        result = {i:image_name}
        image_id_list.append(result)
    #initail image_id
    results_list = [0]    
    
    for i in image_ids:
        if i < image_ids[-1]:
            if list(image_id_list[i].values()) != list(image_id_list[i+1].values()):
                #extract_image_id
                results_list.append(i+1)
    return results_list

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        fig, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    fig.savefig(os.path.join(IMAGE_DIR, save_name), bbox_inches='tight')
    if auto_show:
        plt.show()

def display_differences(image,
                        gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        class_names, title="", ax=None,
                        show_mask=True, show_box=True,
                        iou_threshold=0.5, score_threshold=0.5):
    """Display ground truth and prediction instances on the same image."""
    # Match predictions to ground truth
    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_box, gt_class_id, gt_mask,
        pred_box, pred_class_id, pred_score, pred_mask,
        iou_threshold=iou_threshold, score_threshold=score_threshold)
    # Ground truth = green. Predictions = red
    colors = [(0, 1, 0, .8)] * len(gt_match)\
           + [(1, 0, 0, 1)] * len(pred_match)
    # Concatenate GT and predictions
    class_ids = np.concatenate([gt_class_id, pred_class_id])
    scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
    boxes = np.concatenate([gt_box, pred_box])
    masks = np.concatenate([gt_mask, pred_mask], axis=-1)
    class_names_list = []
    for i in range(len(class_ids)):
        class_names_list.append(dataset.class_names[class_ids[i]])

    # Captions per instance show score/IoU
    captions = [class_names_list[i] for i in range(len(gt_match))] + [class_names_list[len(gt_match)+i] + "{:.2f} / {:.2f}".format(
        pred_score[i],
        (overlaps[i, int(pred_match[i])]
            if pred_match[i] > -1 else overlaps[i].max()))
            for i in range(len(pred_match))]

    # Captions per instance show score
    captions_score = ["0" for m in gt_match] + ["{:.2f}".format(pred_score[i])
            for i in range(len(pred_match))]
    # Captions per instance show IoU
    captions_IoU = ["" for m in gt_match] + ["{:.2f}".format(
        (overlaps[i, int(pred_match[i])]
            if pred_match[i] > -1 else overlaps[i].max()))
            for i in range(len(pred_match))]
    # Set title if not provided
    title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
    
    # save score
    
    f = open(os.path.join(IMAGE_DIR, 'output.csv'), 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(['FileName', 'class_ids', 'class_name', 'boxes', 'score', 'IoU'])
    for i in range(len(class_ids)):
        f = open(os.path.join(IMAGE_DIR, 'output.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow([info["id"], class_ids[i], class_names_list[i], boxes[i], scores[i], captions_IoU[i]])
        f.close()

    # Display
    display_instances(
        image,
        boxes, masks, class_ids,
        class_names, scores, ax=ax,
        show_bbox=show_box, show_mask=show_mask,
        colors=colors, captions=captions,
        title=title)

    
if __name__ == '__main__':
    
    # Load validation dataset
    dataset = train.BalloonDataset()
    dataset.load_balloon(DATA_DIR, "test")

    # Must call before using the dataset
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
    
    

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR + log_dir,
                                  config=config)


    # Set path to balloon weights file

    # Download file from the Releases page and set its path
    # https://github.com/matterport/Mask_RCNN/releases
    # weights_path = "/path/to/mask_rcnn_balloon.h5"

    # Or, load the last model you trained
    # weights_path = model.find_last()
    weights_path = BALLON_WEIGHTS_PATH

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    for image_id in extract_image_id(dataset.image_ids):
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                               dataset.image_reference(image_id)))

        # Run object detection
        results = model.detect([image], verbose=1)

        # Display results
        ax = get_ax(1)
        r = results[0]
#        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
#                                    dataset.class_names, r['scores'], ax=ax,
#                                    title="Predictions")       
                                 
        count = r.get("rois").shape[0]
#        basename = os.path.basename(path)
#        save_name = "{}_detect_count{}.{}".format(basename.split('.')[0], count, basename.split('.')[1])
        save_name = info["id"]
        
        display_differences(image, 
                gt_bbox, gt_class_id, gt_mask,
                r['rois'], r['class_ids'], r['scores'], r['masks'],
                dataset.class_names, ax=None, 
                show_mask=True, show_box=True, 
                iou_threshold=0.5, score_threshold=0.5) 
        
#        visualize.save_result_box(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, 
#                                IMAGE_DIR, save_name, r['scores'], auto_show=False)
#        visualize.save_result_mask(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, 
#                                IMAGE_DIR, save_name, r['scores'], auto_show=False)                        
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)
