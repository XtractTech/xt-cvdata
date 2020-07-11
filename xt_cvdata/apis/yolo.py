import os
import json
import numpy as np
import pandas as pd
from PIL import Image

from .builder import Builder


class YOLO(Builder):

    image_sizes_train='train_yolo.shapes'
    image_sizes_val='val_yolo.shapes'

    def __init__(
        self, source,
        val_path='val_yolo.txt',
        train_path='train_yolo.txt',
        classes_path='classes.txt'
    ):
        """YOLO dataset api and builder.
        
        Arguments:
            source {str} -- Local location of full dataset. Folder structure should
                be YOLO structure.
        
        Keyword Arguments:
            val_path {str} -- Relative path to validation annotations. 
            train_path {str} -- Relative path to train annotations. 
            classes_path {str} -- Relative path to the file containing classes.
        """

        self.val_path = val_path
        self.train_path = train_path
    
        # Load classes into memory
        with open(os.path.join(source, classes_path)) as f:
            classes = [line.strip() for line in f.readlines()]

        self.categories = pd.DataFrame(classes, columns=['name'])
        self.categories['supercategory'] = None
        self.categories.index.name = 'index'
        self.categories.reset_index(inplace=True)

        # Combine training and validation sets into single data frame
        # Image paths
        with open(os.path.join(source, train_path)) as f:
            train_image_paths = [l.strip() for l in f.readlines() if l]
        with open(os.path.join(source, val_path)) as f:
            val_image_paths = [l.strip() for l in f.readlines() if l]
        # Image sizes
        # Get image dimensions
        if not os.path.exists(os.path.join(source, self.image_sizes_val)):
            with open(os.path.join(source, self.image_sizes_val), 'w') as f:
                for im_path in val_image_paths:
                    w, h = Image.open(im_path).size
                    f.write(f'{w} {h}\n')
            with open(os.path.join(source, self.image_sizes_train), 'w') as f:
                for im_path in train_image_paths:
                    w, h = Image.open(im_path).size
                    f.write(f'{w} {h}\n')

        with open(os.path.join(source, self.image_sizes_train)) as f:
            lines_wh = f.readlines()
            train_w = [int(l.strip().split(' ')[0]) for l in lines_wh]
            train_h = [int(l.strip().split(' ')[1]) for l in lines_wh]

        with open(os.path.join(source, self.image_sizes_val)) as f:
            lines_wh = f.readlines()
            val_w = [int(l.strip().split(' ')[0]) for l in lines_wh]
            val_h = [int(l.strip().split(' ')[1]) for l in lines_wh]

        images_train = pd.DataFrame({
            'file_name': train_image_paths,
            'width': train_w, 
            'height': train_h
        })

        images_val = pd.DataFrame({
            'file_name': val_image_paths,
            'width': val_w, 
            'height': val_h
        })        
        images_train['set'] = 'train'
        images_val['set'] = 'val'
        
        self.images = pd.concat((images_train, images_val))
        self.images['id'] = [os.path.splitext(fn)[0] for fn in self.images.file_name]
        self.images['source'] = source
        self.images['license'] = None        

        # Annotations
        bboxes = []
        category_ids = []
        image_ids = []
        for image_id in self.images.id:
            txt_ann_file = image_id + '.txt'
            assert os.path.isfile(txt_ann_file)
            with open(txt_ann_file) as f:
                for ann in f.readlines():
                    split_ann = ann.split()
                    category_id = int(split_ann[0])
                    bbox = [float(b) for b in split_ann[1:]]
                    bboxes.append(bbox)
                    category_ids.append(category_id)
                    image_ids.append(image_id)

        self.annotations = pd.DataFrame({
            'bbox': bboxes,
            'index': category_ids,
            'id': image_ids
        })
        self.annotations['segmentation'] = None
        self.annotations['area'] = None #TODO is area useful?
        self.annotations['iscrowd'] = False
        self.annotations['ignore'] = False

        self.annotations = self.annotations.merge(self.categories[['name', 'index']], how='inner', on='index')
        self.annotations = self.annotations.merge(self.images[['set', 'id']], how='inner', on='id')

        self.annotations.rename(columns={'index':'category_id', 'id': 'image_id'}, inplace=True)
        self.annotations.set_index('category_id', inplace=True)
        self.annotations['id'] = range(1, len(self.annotations) + 1)

        self.images.set_index('id', inplace=True)

        self.categories.set_index('index', inplace=True)
        self.categories.index.name = 'id'

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
