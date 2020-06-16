import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import copy

__all__ = ['ImageClassificationDataset', 'MaskDataset']

class ImageClassificationDataset(Dataset):

    def __init__(self, builder, split='train', transform=None, target_dict=None):
        """Image Classification Dataset

        Args:
            builder (xt_cvdata.apis.Builder): The dataset builder
            split (str, optional): The split (train/val/test). Defaults to 'train'.
            transform (callable, optional): Callable transforms. Defaults to None.
            target_dict (dict, optional): Any label mapping can be implemented here. Defaults to None.
        """

        self.builder = copy.deepcopy(builder)

        # Set index to image_id for faster mapping
        if self.builder.annotations.index.name != 'image_id':
            self.builder.annotations = self.builder.annotations.reset_index().set_index('image_id')

        # Subset to train/val/test set
        self.builder.images = self.builder.images[self.builder.images['set'] == split]
        self.builder.annotations = self.builder.annotations[self.builder.annotations['set'] == split]

        assert (
            not self.builder.images[self.builder.images['set'] == split].empty
            and not self.builder.annotations[self.builder.annotations['set'] == split].empty
        )

        self.split = split
        self.transform = transform
        self.target_dict = target_dict

    def __getitem__(self, index):
        image_data = self.builder.images.iloc[index]
        impath = os.path.join(image_data.source, image_data.file_name)
        ann_data = self.builder.annotations.xs(image_data.name)
        img = Image.open(impath).convert('RGB')
        label = ann_data.category_id


        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_dict:
            label = self.target_dict[label]

        return img, label


    def __len__(self):
        return len(self.builder.images)
    
    def get_class_prop(self):
        """Calculate the class proportion. Useful for weighted sampling/loss

        Returns:
            list: List of class proportions
        """
        print("Calculating Class Distribution...")
        cat_sizes = self.builder.annotations.groupby('category_id').size()
        cat_prop = cat_sizes / len(self)

        weight = []
        for i in range(len(self.builder.categories)):
            try:
                weight.append(cat_prop[i])
            except KeyError:
                print(f'Warning: Class {i} does not exist in the training set. \
                        Not using weighted loss')
            return None

        return weight


class MaskDataset(Dataset):

    def __init__(self, builder, name_to_pix, split='train', transform=None):
        """Mask dataset
        
        Arguments:
            builder {xt_cvdata.apis.Builder} -- Full xt-cv dataset builder
            name_to_pix {dict} -- Dictionary mapping class names to pixel values
            split (str, optional): The split (train/val/test). Defaults to 'train'.
            transform (callable, optional): Callable transforms. Defaults to None.        
        """

        self.builder = copy.deepcopy(builder)

        # Set index to image_id for faster mapping
        if self.builder.annotations.index.name != 'image_id':
            self.builder.annotations = self.builder.annotations.set_index('image_id')

        # Subset to train/val/test set
        self.builder.images = self.builder.images[self.builder.images['set'] == split]
        self.builder.annotations = self.builder.annotations[self.builder.annotations['set'] == split]

        assert (
            not self.builder.images[self.builder.images['set'] == split].empty
            and not self.builder.annotations[self.builder.annotations['set'] == split].empty
        )

        self.split = split
        self.transform = transform
        self.name_to_pix = name_to_pix
        assert 'background' in name_to_pix.keys(), "MaskDataset Expects A Background Class"

    def __getitem__(self, index):
        image_data = self.builder.images.iloc[index]
        impath = os.path.join(image_data.source, image_data.file_name)
        ann_data = self.builder.annotations.xs(image_data.name)
        ann_path = os.path.join(image_data.source, ann_data.segmentation)
        img = Image.open(impath).convert('RGB')
        mask = Image.open(ann_path).convert('L')

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        ignore_inds = np.isin(mask, list(self.name_to_pix.values()))
        mask[~ignore_inds] = self.name_to_pix['background']
        for i, v in enumerate(self.name_to_pix.values()):
            mask[mask == v] = i

        return img, mask.squeeze()


    def __len__(self):
        return len(self.builder.images)
    
    def get_class_prop(self):
        """Get pixel class proportinos

        Returns:
            list: list of pixel class proportions
        """
        class_to_count = {}
        for _, mask in self:
            values, counts = np.unique(mask, return_counts=True)
            for i in range(len(values)):
                if values[i] in class_to_count:
                    class_to_count[values[i]] += counts[i]
                else:
                    class_to_count[values[i]] = counts[i]
        
        total_pixels = sum(list(class_to_count.values()))
        weight = []
        for c in range(len(self.name_to_pix)):
            weight.append(class_to_count[c]/total_pixels)
        
        return weight