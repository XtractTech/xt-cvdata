import os
import json
import numpy as np
import pandas as pd

from .builder import Builder


class COCO(Builder):

    def __init__(
        self, source,
        inst_val_path='annotations/instances_val2017.json',
        inst_train_path='annotations/instances_train2017.json',
        image_paths={'train': 'train', 'val': 'val'}
    ):
        """COCO 2017 dataset api and builder.
        
        Arguments:
            source {str} -- Local location of full dataset. Folder structure should
                be as shown below.
        
        Keyword Arguments:
            inst_val_path {str} -- Relative path to validation annotations. 
            inst_train_path {str} -- Relative path to train annotations. 
            image_paths {dict} -- Dict of relative path to train and val images.
        """

        self.inst_val_path = inst_val_path
        self.inst_train_path = inst_train_path
        self.image_paths = [image_paths]

        # Load annotations into memory
        with open(os.path.join(source, self.inst_val_path)) as jf:
            instances_val = json.load(jf)
        with open(os.path.join(source, self.inst_train_path)) as jf:
            instances_train = json.load(jf)

        self.info = [instances_train['info']]
        self.licenses = pd.DataFrame(instances_train['licenses'])
        self.licenses['source'] = source

        self.categories = pd.DataFrame(instances_train['categories'])
        self.categories.set_index('id', inplace=True)

        # Combine training and validation sets into single data frame
        images_train = pd.DataFrame(instances_train['images'])
        images_val = pd.DataFrame(instances_val['images'])
        images_train['set'] = 'train'
        images_val['set'] = 'val'
        self.images = pd.concat((images_train, images_val))
        self.images.set_index('id', inplace=True)
        self.images['source'] = source

        annotations_train = pd.DataFrame(instances_train['annotations'])
        annotations_val = pd.DataFrame(instances_val['annotations'])
        annotations_train['set'] = 'train'
        annotations_val['set'] = 'val'
        self.annotations = pd.concat((annotations_train, annotations_val))
        self.annotations.set_index('category_id', inplace=True)
        self.annotations = self.annotations.join(self.categories[['name']], how='inner')
        self.annotations.index.name = 'category_id'
        self.annotations['ignore'] = False

        # Save source in list so that we can append later if datasets are merged
        self.source = [source]

        # Generate unique numeric ID for this data source
        self.source_id = [str(abs(hash(source)) % (10 ** 8))]
        self.transformations = {}

        # Make image and annotation IDs unique to this data source using unique ID
        self.images.index = self.source_id[0] + '_' + self.images.index.astype(str)
        self.images.index.name = 'id'
        self.annotations.image_id = self.source_id[0] + '_' + self.annotations.image_id.astype(str)
        self.annotations.id = self.source_id[0] + self.annotations.id.astype(str)

        self.verify_schema()
        self.analyze()
