import os
from os.path import splitext
import numpy as np
import pandas as pd
from pathlib import Path

from .builder import Builder

def _default_category_transform(categories):
    return categories, None

class ImageFolderBuilder(Builder):

    image_sizes = 'image_sizes.csv'

    def __init__(self, source, category_transform=None):
        """Image Folder builder for classification.
        
        Arguments:
            source {str} -- Local location of full dataset. Folder structure should
                be as shown below.
            category_transform {func} -- Function to map each category dir to the category name
        
        Notes:
            The Image Folder dataset directory should have the following structure:
                ./category1
                  img.jpg
                  ...

                ./category2
                  img.jpg
                  ...

                ...
        """
        # Get image dimensions
        if not os.path.exists(os.path.join(source, self.image_sizes)):
            find_cmd = ( 
                "find {} -mindepth 2 -type f -printf '\"%P\"' -exec identify -format ',%w,%h\n' {{}} \; | " 
                "pv -lrbt -N 'Collecting image dimensions' > {}" 
            )
            os.system(find_cmd.format(
                source,
                os.path.join(source, self.image_sizes)
            ))

        # Read in categories and define numeric ID
        categories = [d.name for d in Path(source).iterdir() if d.is_dir()]
        if category_transform:
            categories, supercategories = category_transform(categories)
        else:
            categories, supercategories = _default_category_transform(categories)
        categories = np.unique(categories)

        self.categories = pd.DataFrame(categories, columns=['name'])
        self.categories['supercategory'] = supercategories
        self.categories.index.name = 'index'
        self.categories.reset_index(inplace=True)

        # Read in annotations and image paths
        image_paths = [
            "/".join(str(path).split("/")[-2:])
            for path in Path(source).glob('*/*') if path.is_file()
        ]
        
        self.images = pd.DataFrame(image_paths, columns=['file_name'])
        self.images['id'] = [splitext(fn)[0] for fn in self.images.file_name]
        self.images['license'] = None
        self.images['source'] = source

        # Image dimensions
        img_dims = pd.read_csv(os.path.join(source, self.image_sizes))
        img_dims.columns = ['file_name', 'width', 'height']
        self.images = self.images.merge(img_dims, how='inner', on='file_name')

        self.images['set'] = 'train'
        self.images.set_index('id', inplace=True)

        self.annotations = pd.DataFrame(list(self.images.index), columns=['image_id'])
        self.annotations['segmentation'] = None
        self.annotations['bbox'] = [[]] * len(self.annotations)
        self.annotations['iscrowd'] = False
        self.annotations['ignore'] = False
        self.annotations['area'] = None
        self.annotations['set'] = 'train'
        self.annotations['id'] = range(1, len(self.annotations) + 1)
        
        self.annotations['name'] = category_transform([
            im_id.split("/")[0] for im_id in self.annotations.image_id
        ])[0]
        self.annotations = self.annotations.merge(self.categories, how='inner', on='name')
        self.annotations.rename(columns={'index':'category_id'}, inplace=True)
        self.annotations.set_index('category_id', inplace=True)

        self.categories.set_index('index', inplace=True)
        self.categories.index.name = 'id'

        # Generate unique numeric ID for this data source
        self.source_id = [str(abs(hash(source)) % (10 ** 8))]
        self.transformations = {}
        
        # Save source in list so that we can append later if datasets are merged
        self.source = [source]

        # Make image and annotation IDs unique to this data source using unique ID
        self.images.index = self.source_id[0] + '_' + self.images.index.astype(str)
        self.images.index.name = 'id'
        self.annotations.image_id = self.source_id[0] + '_' + self.annotations.image_id.astype(str)
        self.annotations.id = self.source_id[0] + self.annotations.id.astype(str)
        
        self.verify_schema()
        self.analyze()




