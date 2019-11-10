import os
import sys
import json
import cv2
import pandas as pd
import numpy as np
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

############################################################
#  Support Functions
############################################################

def cv2plt(img, isColor=True):
    original_img = img
    original_img = original_img.transpose(1, 2, 0)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    return original_img


def make_mask_img(dataset, segment_df):
    category_num = dataset.get_category_count()
    seg_width = segment_df.Width.values[0]
    seg_height = segment_df.Height.values[0]
    seg_img = np.full(seg_width * seg_height, category_num - 1, dtype=np.uint8)
    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):
        pixel_list = list(map(int, encoded_pixels.split(" ")))
        for i in range(0, len(pixel_list), 2):
            start_index = pixel_list[i] - 1
            index_len = pixel_list[i + 1] - 1
            seg_img[start_index:start_index + index_len] = \
                int(int(class_id.split("_")[0]) / (category_num - 1) * 255)
    seg_img = seg_img.reshape((seg_height, seg_width), order='F')
    return seg_img


def train_generator(dataset, batch_size):
    df = dataset.get_train_info()
    img_ind_num = df.groupby("ImageId")["ClassId"].count()
    index = df.index.values[0]
    trn_images = []
    seg_images = []
    for i, (img_name, ind_num) in enumerate(img_ind_num.items()):
        img = cv2.imread(os.path.join(dataset.get_train_data_dir(), img_name))
        segment_df = (df.loc[index:index + ind_num - 1, :]).reset_index(drop=True)
        index += ind_num
        if segment_df["ImageId"].nunique() != 1:
            raise Exception("Index Range Error")
        seg_img = make_mask_img(dataset, segment_df)

        # HWC -> CHW
        img = img.transpose((2, 0, 1))

        trn_images.append(img)
        seg_images.append(seg_img)
        if (i + 1) % batch_size == 0:
            return trn_images, seg_images

############################################################
#  Dataset
############################################################


class IMaterialistDataset(utils.Dataset):

    def __init__(self, dataset_dir=None):
        if dataset_dir == "" or dataset_dir is None:
            dataset_dir = os.path.join(ROOT_DIR, "data")
        
        assert os.path.exists(dataset_dir), "Dataset directory not found!"
        self.dataset_dir = dataset_dir
        self.train_data_dir = os.path.join(dataset_dir, 'train')
        self.val_data_dir = os.path.join(dataset_dir, 'test')
        self.__read_from_disk()

    def __read_from_disk(self):
        labels_path = os.path.join(self.dataset_dir, 'label_descriptions.json')
        # Read labels file
        with open(labels_path, 'r') as f:
            self.__labels = json.load(f)
        self.__categories_info = pd.DataFrame(self.__labels.get('categories'))
        self.__attributes_info =  pd.DataFrame(self.__labels.get('attributes'))

        train_info_path = os.path.join(self.dataset_dir, 'train.csv')
        # Read Train Info
        self.__traininfo = pd.read_csv(train_info_path)

        # Read some information about training data folder
        self.__num_training_data = len(os.listdir(self.get_train_data_dir()))

        # Read some information about test data folder
        self.__num_val_data = len(os.listdir(self.get_val_data_dir()))

        # Validate data integrity
        self.__validate_data_integrity()

    def __validate_data_integrity(self):
        assert len(self.__traininfo.groupby("ImageId")) == self.__num_training_data, \
            "Num of unique ImageIDs in Train Info ~= Num images in training data folder!"

    @staticmethod
    def classid2label(class_id):
        """Converts the Class ID string for iMaterialist fashion dataset and
        converts it to the respective cateogry number and corresponding
        list attributes values."""
        category, *attribute = class_id.split("_")
        return category, attribute

    def get_dataset_dir(self):
        return self.dataset_dir

    def get_train_data_dir(self):
        return self.train_data_dir

    def get_num_training_images(self):
        return self.__num_training_data

    def get_num_val_images(self):
        return self.__num_val_data

    def get_val_data_dir(self):
        return self.val_data_dir

    def get_labels_dict(self):
        return self.__labels

    def get_labels_info(self):
        return self.__labels.get('info')

    def get_categories_info(self):
        return self.__categories_info

    def get_category_count(self):
        return len(self.__categories_info.id)

    def get_attributes_info(self):
        return self.__attributes_info

    def get_train_info(self):
        return self.__traininfo

    def classid2str(self, classid):
        assert classid in self.__categories_info.id, "Invalid Class ID"
        return self.__categories_info[self.__categories_info.id == int(classid)].name.values[0]

    def attrid2str(self, attrid):
        assert attrid in self.__attributes_info.id, "Invalid Class ID"
        return self.__attributes_info[self.__attributes_info.id == int(attrid)].name.values[0]

    def find_segment_with_imageid(self, img_name):
        search_filter = self.__traininfo.ImageId == img_name
        return self.__traininfo.loc[search_filter]

    def print_img(self, img_name, ax):
        """
        Called to display an image with its categories and attributes
        """
        search_filter = self.__traininfo.ImageId == img_name
        img_df = self.__traininfo.loc[search_filter]
        img_labels = list(set(img_df.ClassId.values))

        self.print_img_with_labels(img_name, img_labels, ax)

    def print_img_with_labels(self, img_name, img_labels, ax):
        """
        Support function for print_image().
        Displays a single image with its respective categories and attributes
        for each identified element of clothing in the image.
        """
        img_path = os.path.join(self.get_train_data_dir(), img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label_interval = (img.shape[0] * 0.9) / len(img_labels)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(img_name)

        for label in img_labels:
            cat, attrs = IMaterialistDataset.classid2label(label)
            print("Category: {}".format(self.classid2str(int(cat))))
            if attrs is not None:
                for a in attrs:
                    print("\tAttribute: ", end="")
                    print("  {}  ".format(self.attrid2str(int(a)), end=""))
                    print()

    def get_image_with_mask(self, image_name):
        # Display a random image with its mask
        df_img = self.find_segment_with_imageid(image_name)
        if df_img["ImageId"].nunique() != 1:
            raise Exception("Index Range Error")
        img = cv2.imread(os.path.join(self.get_train_data_dir(), image_name))
        seg = make_mask_img(self, df_img)
        return img, seg