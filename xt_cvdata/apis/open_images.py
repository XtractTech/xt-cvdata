import os
import numpy as np
import pandas as pd

from .builder import Builder


class OpenImages(Builder):

    class_desc_path = 'annotations/class-descriptions-boxable.csv'
    inst_val_path = 'annotations/validation-annotations-bbox.csv'
    inst_train_path = 'annotations/train-annotations-bbox.csv'
    img_sizes_train = 'annotations/train-img-sizes.csv'
    img_sizes_val = 'annotations/val-img-sizes.csv'
    image_paths = [{'train': 'train', 'val': 'val'}]

    def __init__(self, source):
        """Open Image V5 dataset api and builder.
        
        Arguments:
            source {str} -- Local location of full dataset. Folder structure should
                be as shown below.
        
        Notes:
            The Open Images V5 dataset directory should have the following structure:
                ./annotations
                    train-annotations-bbox.csv
                    validation-annotations-bbox.csv
                    class-descriptions-boxable.csv
                ./train
                    <image1>.jpg
                    <image2>.jpg
                    ...
                ./val
                    <image1>.jpg
                    <image2>.jpg
                    ...
        """

        # If not already there, get a list of the image sizes
        # This is needed to convert fractional bboxes to pixels
        if not os.path.exists(os.path.join(source, self.img_sizes_val)):
            find_cmd = ( 
                "find {} -type f -printf '\"%P\"' -exec identify -format ',%w,%h\n' {{}} \; | " 
                "pv -lrbt -N 'Collecting image dimensions' > {}" 
            )
            os.system(find_cmd.format(
                os.path.join(source, self.image_paths[0]['val']),
                os.path.join(source, self.img_sizes_val)
            ))
            os.system(find_cmd.format(
                os.path.join(source, self.image_paths[0]['train']),
                os.path.join(source, self.img_sizes_train)
            ))

        self.info = [
            {
                'description': 'Open Images Dataset v5',
                'url': 'https://storage.googleapis.com/openimages/web/index.html',
                'version': '5.0',
                'year': 2019
            }
        ]

        self.licenses = pd.DataFrame(
            [{
                'id': 1,
                'name': 'Apache License Version 2.0',
                'url': 'http://www.apache.org/licenses/',
                'source': source
            }]
        )
        self.licenses.set_index('id', inplace=True)

        # Read in categories and define numeric ID
        self.categories = pd.read_csv(os.path.join(source, self.class_desc_path), header=None)
        self.categories.columns = ['id', 'name']
        self.categories['supercategory'] = None
        self.categories = self.categories.reset_index().set_index('id')

        # Read in annotations files (for full Open Images, these are pretty big)
        annotations_val = pd.read_csv(os.path.join(source, self.inst_val_path))
        annotations_train = pd.read_csv(os.path.join(source, self.inst_train_path))

        # Combine training and validation sets into single data frame
        images_train = pd.DataFrame(annotations_train.ImageID.unique(), columns=['id'])
        images_val = pd.DataFrame(annotations_val.ImageID.unique(), columns=['id'])
        images_train['set'] = 'train'
        images_val['set'] = 'val'
        self.images = pd.concat((images_train, images_val))
        self.images['file_name'] = self.images.id + '.jpg'
        self.images['license'] = 1
        self.images['source'] = source

        # Load and join image dimensions
        img_dims = pd.concat((
            pd.read_csv(os.path.join(source, self.img_sizes_train), header=None),
            pd.read_csv(os.path.join(source, self.img_sizes_val), header=None)
        ))
        img_dims.columns = ['file_name', 'width', 'height']
        self.images = self.images.merge(img_dims, how='inner', on='file_name')
        self.images.set_index('id', inplace=True)

        # Build annotations, converting bboxes to pixel values (x, y, w, h)
        # Here, we also set some required columns to default values
        annotations_train.drop(
            columns=[
                'Source', 'Confidence', 'IsOccluded', 'IsTruncated',
                'IsGroupOf', 'IsDepiction', 'IsInside'
            ],
            inplace=True
        )
        annotations_val.drop(
            columns=[
                'Source', 'Confidence', 'IsOccluded', 'IsTruncated',
                'IsGroupOf', 'IsDepiction', 'IsInside'
            ],
            inplace=True
        )
        annotations_train['set'] = 'train'
        annotations_val['set'] = 'val'
        self.annotations = pd.concat((annotations_train, annotations_val))
        del annotations_train, annotations_val
        self.annotations.set_index('ImageID', inplace=True)
        self.annotations = self.annotations.join(self.images[['width', 'height']])
        self.annotations.XMin = self.annotations.XMin * self.annotations.width
        self.annotations.YMin = self.annotations.YMin * self.annotations.height
        # Hijacking XMax and YMax to save memory (below, they are really width and height of bbox)
        self.annotations.XMax = self.annotations.XMax * self.annotations.width - self.annotations.XMin
        self.annotations.YMax = self.annotations.YMax * self.annotations.height - self.annotations.YMin
        self.annotations['area'] = self.annotations.XMax * self.annotations.YMax
        self.annotations.drop(columns=['width', 'height'], inplace=True)
        self.annotations['bbox'] = np.stack(
            (
                self.annotations.XMin,
                self.annotations.YMin,
                self.annotations.XMax,
                self.annotations.YMax
            ), axis=1
        ).round(2).tolist()
        self.annotations['segmentation'] = [[]] * len(self.annotations)
        self.annotations['iscrowd'] = False
        self.annotations['ignore'] = False
        self.annotations.drop(columns=['XMin', 'XMax', 'YMin', 'YMax'], inplace=True)
        self.annotations = self.annotations.reset_index().set_index('LabelName')
        self.annotations.rename(columns={'index': 'image_id'}, inplace=True)
        self.annotations['id'] = range(1, len(self.annotations) + 1)
        self.annotations = self.annotations.join(self.categories[['name', 'index']], how='inner')
        self.annotations.set_index('index', inplace=True)
        self.annotations.index.name = 'category_id'

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
