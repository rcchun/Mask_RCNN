import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import skimage
import cv2
############################################################
#  Configurations
############################################################


class BalloonConfig():
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "class"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("class", 1, "1")
        self.add_class("class", 2, "2")
        self.add_class("class", 3, "3")
        self.add_class("class", 4, "4")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_export_json0413.json")))

        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                objects = [s['region_attributes'] for s in a['regions'].values()]
                num_ids = [n['class'] for n in objects]

            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
                objects = [s['region_attributes'] for s in a['regions']]
                num_ids = [n['class'] for n in objects]

            #if type(a['regions']) is dict:
            #    polygons_x , polygons_y = [r['all_points_x'] for r in a['regions'].values()],[r['all_points_y'] for r in a['regions'].values()]
            #    polygons = [polygons_x , polygons_y]
            #else:
            #    polygons_x , polygons_y = [r['all_points_x'] for r in a['regions']],[r['all_points_y'] for r in a['regions']]
            #    polygons = [polygons_x , polygons_y]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            filename = a['filename']
            

            self.add_image(
                "class",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids, filename=filename)     
        
    def get_class_id_from_class_name(self, class_name):
        class_ids=[]
        for i in range(len(class_name)):
            for j in range(len(self.class_info)):
                if self.class_info[j]['name'] == class_name[i]:
                    id_name = self.class_info[j]['id']
                    class_ids.append(id_name)
        return class_ids
        
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        
        if image_info["source"] != "class":
            return super(self.__class__, self).load_mask(image_id)
            
        num_ids = image_info['num_ids']
        
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        filename = info['filename']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 255
        
        # change num_dis to class_ids (class -> id)
        class_ids=self.get_class_id_from_class_name(num_ids)
        
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        #return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        return mask, np.array(class_ids, dtype=np.int32), filename

############################################################
#  Generate load mask file
############################################################
if __name__ == '__main__':
    # Load dataset
    config = BalloonConfig()
    BALLOON_DIR = os.path.join(ROOT_DIR, "datasets/test")

    dataset = BalloonDataset()
    dataset.load_balloon(BALLOON_DIR, "val")

    if not(os.path.isdir(BALLOON_DIR + '/result')):
        os.makedirs(os.path.join(BALLOON_DIR + '/result'))
        

    # Must call before using the dataset
    dataset.prepare()

    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    
        
    # Load and save mask image

    image_ids = dataset.image_ids
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids, filename = dataset.load_mask(image_id)
        filename = filename.split('.')[0] + '.png'
        print(class_ids)
        print(image_id)
        print(filename)
        

      
        mask_ = mask[:,:,0]
        if len(class_ids) >= 1:
            for i in range(1,len(class_ids)):
                mask_ += mask[:,:,i]
        skimage.io.imsave(BALLOON_DIR + "/result/%s" %filename,mask_)
        #cv2.imwrite(BALLOON_DIR + "/result/%s" %filename,mask_)