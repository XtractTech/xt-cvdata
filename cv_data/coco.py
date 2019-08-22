import os
import shutil
from tempfile import TemporaryDirectory
import json
from zipfile import ZipFile
import wget
import numpy as np
import pandas as pd

from .builder import Builder


class COCO(Builder):

    base_url = 'http://images.cocodataset.org'
    ann_url = 'annotations/annotations_trainval2017.zip'
    inst_val_path = 'annotations/instances_val2017.json'
    inst_train_path = 'annotations/instances_train2017.json'

    def __init__(self, source: str=None):
        """COCO 2017 dataset api and builder.
        
        Keyword Arguments:
            source {str} -- Local location of full dataset. If not specified, data will be
                downloaded directly from 'http://images.cocodataset.org'. Folder structure should
                be as shown below. (default: {None})
        
        Notes:
            The COCO dataset directory should have the following structure:
                ./annotations
                    instances_val2017.json
                    instances_train2017.json
                ./train
                    <image1>.jpg
                    <image2>.jpg
                    ...
                ./val
                    <image1>.jpg
                    <image2>.jpg
                    ...
        """
        self.source = source
        self.transformations = {}

        # Check if annotations already downloaded
        downloaded = False
        if self.source is not None:
            downloaded = all(
                os.path.exists(os.path.join(self.source, p)) 
                    for p in [self.inst_val_path, self.inst_train_path]
            )

        # Load annotations into object
        with TemporaryDirectory() as ann_dir:
            if not downloaded:
                print(f'Downloading annotations from {self.base_url}')
                zip_path = os.path.join(ann_dir, self.ann_url)
                os.makedirs(os.path.dirname(zip_path), exist_ok=True)
                wget.download(os.path.join(self.base_url, self.ann_url), zip_path)
                with ZipFile(zip_path) as zf:
                    zf.extractall(ann_dir)
            else:
                ann_dir = self.source

            with open(os.path.join(ann_dir, self.inst_val_path)) as jf:
                instances_val = json.load(jf)
            with open(os.path.join(ann_dir, self.inst_train_path)) as jf:
                instances_train = json.load(jf)

            self.info = instances_train['info']

            self.licenses = pd.DataFrame(instances_train['licenses'])
            self.licenses.set_index('id', inplace=True)

            self.categories = pd.DataFrame(instances_train['categories'])
            self.categories.set_index('id', inplace=True)

            images_train = pd.DataFrame(instances_train['images'])
            images_train['set'] = 'train'
            images_val = pd.DataFrame(instances_val['images'])
            images_val['set'] = 'val'
            self.images = pd.concat((images_train, images_val))
            self.images.set_index('id', inplace=True)

            annotations_train = pd.DataFrame(instances_train['annotations'])
            annotations_train['set'] = 'train'
            annotations_val = pd.DataFrame(instances_val['annotations'])
            annotations_val['set'] = 'val'
            self.annotations = pd.concat((annotations_train, annotations_val))
            self.annotations.set_index('category_id', inplace=True)
            self.annotations = self.annotations.join(self.categories[['name']], how='inner')
        
        self.analyze()
