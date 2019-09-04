import os
import numpy as np
import pandas as pd

from .builder import Builder


class VoTTCSV(Builder):

    inst_path = 'annotations.csv'
    image_paths = [{'train': 'images', 'val': 'images'}]
    img_sizes = 'image-sizes.csv'

    def __init__(self, source):
        """VoTT CSV dataset api and builder.
        
        Arguments:
            source {str} -- Local location of full dataset. Folder structure should
                be as shown below.
        
        Notes:
            The VoTT CSV dataset directory should have the following structure:
                ./annotations.csv
                ./images
                    <image1>.jpg
                    <image2>.jpg
                    ...
        """

        # If not already there, get a list of the image sizes
        # This is needed to convert fractional bboxes to pixels
        if not os.path.exists(os.path.join(source, self.img_sizes)):
            find_cmd = ( 
                "find {} -type f -printf '\"%P\"' -exec identify -format ',%w,%h\n' {{}} \; | " 
                "pv -lrbt -N 'Collecting image dimensions' > {}" 
            )
            os.system(find_cmd.format(
                os.path.join(source, self.image_paths[0]['train']),
                os.path.join(source, self.img_sizes)
            ))

        self.info = [
            {
                'description': 'VoTT CSV',
                'url': '',
                'version': '2.0',
                'year': 2019
            }
        ]

        self.licenses = pd.DataFrame(
            [{
                'id': 1,
                'name': 'undefined',
                'url': '',
                'source': source
            }]
        )
        self.licenses.set_index('id', inplace=True)

        # Read in annotation CSV and prepare
        annotations = pd.read_csv(os.path.join(source, self.inst_path))
        annotations['image_id'] = annotations.groupby('image').ngroup() + 1
        annotations['category_id'] = annotations.groupby('label').ngroup() + 1
        annotations['width'] = annotations.xmax - annotations.xmin
        annotations['height'] = annotations.ymax - annotations.ymin
        annotations['area'] = annotations.width * annotations.height
        annotations['bbox'] = np.stack(
            (
                annotations.xmin,
                annotations.ymin,
                annotations.width,
                annotations.height
            ), axis=1
        ).round(2).tolist()
        annotations.drop(columns=['xmin', 'xmax', 'ymin', 'ymax', 'width', 'height'], inplace=True)
        annotations.reset_index(inplace=True)
        annotations.rename(columns={'index': 'id', 'image': 'file_name', 'label': 'name'}, inplace=True)
        annotations['segmentation'] = [[]] * len(annotations)
        annotations['iscrowd'] = False
        annotations['ignore'] = False
        annotations['set'] = 'train'
        annotations['source'] = source

                # Load and join image dimensions
        img_dims = pd.read_csv(os.path.join(source, self.img_sizes), header=None)
        img_dims.columns = ['file_name', 'width', 'height']
        annotations = annotations.merge(img_dims, on='file_name')

        # Define categories
        self.categories = annotations[['category_id', 'name']].groupby('category_id').first()
        self.categories.index.name = 'id'
        self.categories['supercategory'] = None

        # Define images
        self.images = (
            annotations[['image_id', 'file_name', 'set', 'source', 'width', 'height']]
                .groupby('image_id').first()
        )
        self.images.index.name = 'id'
        self.images['license'] = 1

        # Remove unnecessary columns from annotations
        self.annotations = annotations.drop(columns=['file_name', 'width', 'height'])
        self.annotations.set_index('category_id', inplace=True)

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
