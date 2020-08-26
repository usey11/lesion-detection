import json
import os
import cv2
import json
import skimage
import numpy as np
import pandas as pd

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
    IMAGES_PER_GPU = 16

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + nevus + melanoma + sk

    # Uses small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.3

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 400

    # Validation steps at end of epoch
    VALIDATION_STEPS = 5

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 10.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

config = MolesConfig()

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
        mask = cv2.resize(mask, (config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM))
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
        image = cv2.resize(image, (config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM))
        
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

class ISIC17Dataset(MolesDatasetFast):
     def load_moles(self, data_path):
        # Add classes
        self.add_class("moles", 1, "nevus")
        self.add_class("moles", 2, "melanoma")
        self.add_class("moles", 3, "seborrheic_keratosis")
        
        classes = {1:"nevus", 2:"melanoma", 3:"seborrheic_keratosis"}
        if not os.path.exists(data_path):
            raise Exception(data_path + " Does not exists")
            
        ground_truth_file = os.path.join(data_path, "../", "ISIC-2017_Training_Part3_GroundTruth.csv")
        ground_truth_data = pd.read_csv(ground_truth_file)

        images_path = os.path.join(data_path, "images")
        masks_path = os.path.join(data_path, "masks")

        for filename in os.listdir(images_path):
            #TODO Use real image ids
            #data = json.load(open(os.path.join(data_path,"Descriptions",filename)))
            
            image_id = int(filename[-11:-4])
            image_name = filename[:12]
            
            image_filename = os.path.join(images_path, image_name + ".jpg")
            if not os.path.isfile(image_filename):
                continue
                
            mask_filename = os.path.join(masks_path, image_name + "_segmentation.png")
            if not os.path.isfile(mask_filename):
                continue

                
            image_info = ground_truth_data[ground_truth_data["image_id"] == image_name]
            label = int((image_info["melanoma"] * 1 + image_info["seborrheic_keratosis"] * 2) + 1)

            self.add_image("moles", image_id, path=image_filename, mask_path=mask_filename,mole_type=classes[label])

class ISIC17AugDataset(MolesDatasetFast):
    def load_moles(self, data_path):
        # Add classes
        self.add_class("moles", 1, "nevus")
        self.add_class("moles", 2, "melanoma")
        self.add_class("moles", 3, "seborrheic_keratosis")
        
        classes = {1:"nevus", 2:"melanoma", 3:"seborrheic_keratosis"}
        if not os.path.exists(data_path):
            raise Exception(data_path + " Does not exists")
            
        ground_truth_file = os.path.join(data_path, "labels.csv")
        ground_truth_data = pd.read_csv(ground_truth_file)

        images_path = os.path.join(data_path, "images")
        masks_path = os.path.join(data_path, "masks")

        for filename in os.listdir(images_path):
            #TODO Use real image ids
            
            image_id = int(filename[-11:-4])
            image_name = filename[:12]
            
            image_filename = os.path.join(images_path, image_name + ".jpg")
            if not os.path.isfile(image_filename):
                continue
                
            mask_filename = os.path.join(masks_path, image_name + "_segmentation.png")
            if not os.path.isfile(mask_filename):
                continue

                
            image_info = ground_truth_data[ground_truth_data["image_id"] == image_name]
            label = int(image_info["label"])

            self.add_image("moles", image_id, path=image_filename, mask_path=mask_filename,mole_type=classes[label])
