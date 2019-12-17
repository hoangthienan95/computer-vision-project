from mrcnn import utils
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import os
import skimage

def load_dataset(filepath:str):
    with open(filepath, 'rb') as f:
        print(filepath)
        data = pickle.load(f)
    return data

def save_dataset(dataset, filepath:str):
    with open(filepath, 'wb') as f:
        print(filepath)
        pickle.dump(dataset, f)

class FashionDataset(utils.Dataset):
    """
    Implements mrcnn.utils.Dataset.
    FashionDataset holds data relevant to the imaterialist challenge data.
    """

    def __init__(self):
        super(FashionDataset, self).__init__()
        self.class_names= []

    def __len__(self):
        return len(self.image_info)

    def create_classes(self, cat_file:str) -> [dict]:
        """
        Added to FashionDataset.
        Initialize the classes.
        param:cat_file - filepath to fashion dataset's label_descriptions.json file
        """
        # read labels file
        with open(cat_file, 'r') as data_file:
            data=data_file.read()

        # parse file
        labels = json.loads(data)

        categories = labels.get('categories')
        df_categories = pd.DataFrame(categories)
        df_categories['source'] = "imaterialist"

        dict_categories = [dict(x[1]) for x in df_categories.iterrows()]

        for c in dict_categories:
            self.add_class(c['source'], c['id']+1, c['name']) # add 1 to make room for background

        print ("{} classes added.".format(len(dict_categories)))

        return dict_categories


    def create_anns(self, sub_df_images:pd.DataFrame) -> dict:
        """
        Creates an 'annotations' entry in an image's image_info entry.
        dict_keys(['id', 'image_id', 'segmentation', 'category_id', 'area', 'iscrowd', 'bbox']
        """
        annotations = []

        for mask in sub_df_images.iterrows():
            h      = int(mask[1].Height)
            w      = int(mask[1].Width)
            counts = np.fromstring(mask[1].EncodedPixels, dtype=int, sep=" ")
            ann_dict = {'id'            : mask[0],
                        'image_id'      : mask[1].ImageId,
                        'segmentation'  : {'counts' : counts, 'size': [h, w] },
                        'category_id'   : int(mask[1].ClassId.split('_')[0])+1, # add 1 to make room for background
                        'iscrowd'       : True, # True indicates the use of uncompressed RLE
                        'bbox'          : [] }

            annotations.append(ann_dict)

        return annotations


    def create_images(self, images_file:str, train_dir:str, imgids:list=None, limit:int=None) -> (dict, pd.DataFrame):
        """
        Build the image_info['images'] dictionary element with all images.
        If imgids list is None, all images in the images_file will be included, otherwise,
        only the imgids in the list will be included.
        """

        df_images = pd.read_csv(images_file, nrows=limit)

        # restrict the dataframe to items in imgids list, if list is provided
        if imgids is not None:
            df_images = df_images[df_images.ImageId.isin(imgids)]

        df_images_unique = df_images.drop_duplicates('ImageId')

        for image in tqdm(df_images_unique.iterrows(), desc="Add images to object"):
            self.add_image(source       = 'imaterialist',
                           image_id     = image[0],
                           path         = os.path.join(train_dir,image[1].ImageId),
                           height       = image[1].Height,
                           width        = image[1].Width,
                           file_name    = image[1].ImageId,
                           annotations  = self.create_anns(df_images[df_images.ImageId==image[1].ImageId]))

        print("Added {} images.".format(len(df_images_unique)))
        print("Added {} annotations.".format(len(df_images)))

        return self.image_info


    def load_image(self, image_id):
        """
        Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []

        # returns list of masks/annotations for the image
        annotations = self.image_info[image_id]["annotations"]

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = annotation['category_id'] # one of 46 categories

            if class_id:
                # updated to reflect problems with original maskutils implementtaion of decode
                m = self.kaggle_rle_decode(annotation, image_info["height"], image_info["width"])

                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids


    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.
        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        # assume user provided the integer id of the image
        for img in self.image_info:
            if img['id'] == image_id:
                return img['path']

        # check if the user entered the file name
        for img in self.image_info:
            if img['file_name'] == image_id:
                return img['path']

        print ("Image '{}' not found.".format(image_id))
        return None


    def kaggle_rle_decode(self, ann, h, w):
        """
        https://github.com/amirassov/kaggle-imaterialist/blob/master/src/rle.py
        Takes uncompressed RLE for a single mask.  Returns binary mask.
        param: ann - annotation including uncompressed rle in ['segmentation']['counts']
        -- where counts is a list of integers.  Also includes 'size' which is a list [int(h), int(w)]
        """
        rle = ann['segmentation']['counts']

        starts, lengths = map(np.asarray, (rle[::2], rle[1::2]))
        starts -= 1
        ends = starts + lengths
        img = np.zeros(h * w, dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape((w, h)).T


def create_datasets(images_file:str,
                              cat_file:str,
                              images_dir:str,
                              split:float=0.8,
                              limit:int=None) -> (FashionDataset,FashionDataset):
    """
    Returns a train and a val dataset object.
    If limit is None, all entries in file will be used as the population.
    """
    # split the train.csv file into train and val dataframes
    df_images = pd.read_csv(images_file, nrows=limit)

    image_filenames = np.unique(df_images.ImageId)
    train_imgids, val_imgids = train_test_split(image_filenames , train_size=.8)

    # Create empty objects
    fash_train = FashionDataset()
    fash_val = FashionDataset()

    # build classes in dataset objects
    train_classes = fash_train.create_classes(cat_file) # takes seconds
    val_classes   = fash_val.create_classes(cat_file)   # takes seconds

    # load image references and masks into dataset objects
    print("Building trainig dataset...")
    train_image_info = fash_train.create_images( images_file, images_dir, train_imgids, limit=limit)
    print("Building validation dataset...")
    val_image_info   = fash_val.create_images(   images_file, images_dir, val_imgids,   limit=limit)

    fash_train.prepare()
    fash_val.prepare()

    return fash_train, fash_val
