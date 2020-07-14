import json
import os
import cv2
import json
import skimage
import numpy as np

from mrcnn.config import Config
from mrcnn import utils

class MolesConfig(Config):
    """Derives from the base Config class and overrides values specific
    to the Moles dataset.
    """
    # Give the configuration a recognizable name
    NAME = "moles"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + malignant + benign

    # Uses small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.6

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Validation steps at end of epoch
    VALIDATION_STEPS = 5

class MolesDataset(utils.Dataset):
    def load_moles(self, data_path):
        # Add classes
        self.add_class("moles", 1, "malignant")
        self.add_class("moles", 2, "benign")
        
        if not os.path.exists(data_path):
            raise Exception(data_path + " Does not exists")
            
        
        for filename in os.listdir(os.path.join(data_path, "Descriptions")):
            #TODO Use real image ids
            data = json.load(open(os.path.join(data_path,"Descriptions",filename)))
            
            image_id = int(filename[-7:])
            
            image_filename = os.path.join(data_path, "Images", filename + ".jpeg")
            if not os.path.isfile(image_filename):
                continue
                
            mask_filename = os.path.join(data_path, "Segmentation", filename + "_expert.png")
            if not os.path.isfile(mask_filename):
                continue
            
            metadata = data["meta"]
            mole_type = metadata["clinical"]["benign_malignant"]
            if (mole_type != "benign") and (mole_type != "malignant"):
                continue
            
            self.add_image("moles", image_id, path=image_filename, mask_path=mask_filename,mole_type=mole_type)

        
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        
        image = skimage.io.imread(self.image_info[image_id]['path'])
        image = cv2.resize(image, (256,256))
        return image
    
    
    def load_mask(self, image_id):
        info  = self.image_info[image_id]
        mask_path = info["mask_path"]
        
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (256, 256))
        mask = mask[:,:,0] > 11
        mask = mask[...,np.newaxis]
        
        class_id = self.class_names.index(info["mole_type"])
        class_ids = np.array([class_id])
        return mask.astype(np.bool), class_ids.astype(np.int32)
    
    def image_reference(self, image_id):
        if self.image_info[image_id]["source"] == "moles":
            return self.image_info[image_id]["mask_path"]
        else:
            super(self.__class__).image_reference(self, image_id)

class MolesDatasetFast(MolesDataset):
    def __init__(self):
        self.images = {}
        self.masks = {}
        super().__init__()

    def load_moles(self, data_path):
        # Add classes
        self.add_class("moles", 1, "malignant")
        self.add_class("moles", 2, "benign")
        
        if not os.path.exists(data_path):
            raise Exception(data_path + " Does not exists")
            
        
        for filename in os.listdir(os.path.join(data_path, "Descriptions")):
            data = json.load(open(os.path.join(data_path,"Descriptions",filename)))
            
            image_id = int(filename[-7:])
            
            image_filename = os.path.join(data_path, "Images", filename + ".jpeg")
            if not os.path.isfile(image_filename):
                continue
                
            mask_filename = os.path.join(data_path, "Segmentation", filename + "_expert.png")
            if not os.path.isfile(mask_filename):
                continue
            
            metadata = data["meta"]
            mole_type = metadata["clinical"]["benign_malignant"]
            if (mole_type != "benign") and (mole_type != "malignant"):
                continue
            
            self.add_image("moles", image_id, path=image_filename, mask_path=mask_filename,mole_type=mole_type)
    
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        if image_id in self.images:
            return self.images[image_id]
        
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        image = cv2.resize(image, (256,256))
        
        # Add to dict
        self.images[image_id] = image

        return image

    def load_mask(self, image_id):
        
        # Check if it's in the dictionary
        if image_id in self.masks:
            return self.masks[image_id]
        
        # If not then load mask
        mask = super().load_mask(image_id)

        # Add to dict
        self.masks[image_id] = mask

        return mask

class BalancedDataset(MolesDataset):
    def load_moles(self, data_path):
        # Add classes
        self.add_class("moles", 1, "malignant")
        self.add_class("moles", 2, "benign")
        
        if not os.path.exists(data_path):
            raise Exception(data_path + " Does not exists")
        
        cap = 3000
        b_count = 0
        
        for filename in os.listdir(os.path.join(data_path, "Descriptions")):
            #TODO Use real image ids
            data = json.load(open(os.path.join(data_path,"Descriptions",filename)))
            
            image_id = int(filename[-7:])
            
            image_filename = os.path.join(data_path, "Images", filename + ".jpeg")
            if not os.path.isfile(image_filename):
                continue
                
            mask_filename = os.path.join(data_path, "Segmentation", filename + "_expert.png")
            if not os.path.isfile(mask_filename):
                continue
            
            metadata = data["meta"]
            mole_type = metadata["clinical"]["benign_malignant"]
            if (mole_type == "benign") and b_count < cap:
                b_count += 1
            elif (mole_type == "malignant"):
                pass
            else:
                continue
            
            self.add_image("moles", image_id, path=image_filename, mask_path=mask_filename,mole_type=mole_type)

class BalancedDatasetFast(BalancedDataset):
    def __init__(self):
        self.images = {}
        self.masks = {}
        super().__init__()
    
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        if image_id in self.images:
            return self.images[image_id]
        
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        image = cv2.resize(image, (256,256))
        
        # Add to dict
        self.images[image_id] = image

        return image
    
    def load_mask(self, image_id):
        # Check if it's in the dictionary
        if image_id in self.masks:
            return self.masks[image_id]
        
        # If not then load mask
        mask = super().load_mask(image_id)

        # Add to dict
        self.masks[image_id] = mask

        return mask