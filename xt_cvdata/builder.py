import os, shutil
import json
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm


class Builder(object):

    def __init__(self, source=None):
        """Base object for building and modifying object detection and segmentation datasets."""

        # Define required data schema (PK = primary key, FK = foreign key)
        self.info = [{'description': '', 'url': '', 'version': '', 'year': ''}]
        self.licenses = pd.DataFrame(
            columns=[
                'id', # PK
                'name',
                'url',
                'source'
            ]
        )
        self.categories = pd.DataFrame(
            columns=[
                'id', # PK
                'supercategory',
                'name'
            ]
        )
        self.categories.set_index('id', inplace=True)
        self.annotations = pd.DataFrame(
            columns=[
                'id', # PK
                'image_id', # FK
                'category_id', # FK
                'name', # FK
                'area',
                'bbox',
                'segmentation',
                'ignore',
                'iscrowd',
                'set'
            ]
        )
        self.annotations.set_index('category_id', inplace=True)
        self.images = pd.DataFrame(
            columns=[
                'id', # PK
                'license', # FK
                'file_name',
                'height',
                'set',
                'source',
                'width'
            ]
        )
        self.images.set_index('id', inplace=True)

        self.source = [source]
        self.transformations = {}

        self.analyze()
        
    def __str__(self):
        return (
            'Detection/segmentation dataset object:\n'
            f'{type(self).__name__}(source="{self.source}")\n\n'
            f'Classes: {self.num_classes}\n\n'
            f'Images:\n{self.num_images}\n\n'
            f'Annotations:\n{self.num_annotations}\n\n'
            f'Transformations: {json.dumps(self.transformations, indent=2)}'
        )
    
    def __repr__(self):
        return self.__str__()
    
    def verify_schema(self):
        """Function to verify schema.

        Use this function to check that attributes in inheriting classes follow the required
        format.
        """

        # Categories
        assert all(c in ['supercategory', 'name'] for c in self.categories.columns)
        assert self.categories.index.name == 'id'

        # Annotations
        assert all(
            c in self.annotations.columns for c in [
                'id', 'image_id', 'name',  'area', 'bbox',
                'segmentation', 'ignore', 'iscrowd', 'set'
            ]
        )
        assert self.annotations.index.name == 'category_id'

        # Images
        assert all(
            c in self.images.columns for c in [
                'license', 'file_name', 'height', 'set',
                'source', 'width'
            ]
        )
        assert self.images.index.name == 'id'
    
    def analyze(self):
        """Compute dataset statistics.
        
        This should be called each time the dataset object is modified.
        """

        # Get aggregate statistics
        self.num_classes = len(self.categories)

        agg_fn = lambda l: {'set': [('size', 'size'), ('prop', lambda x: len(x) / l)]}
        self.num_images = self.images.groupby('set').agg(agg_fn(len(self.images)))
        self.num_annotations = self.annotations.groupby('set').agg(agg_fn(len(self.annotations)))

        # Get class-wise statistics
        class_dist = self.annotations.groupby(['set', 'name']).size()
        self.class_distribution = None
        for s in class_dist.index.levels[0]:
            set_dist = class_dist[s]
            set_dist.name = s
            set_dist[f'{s}_prop'] = set_dist / set_dist.sum()
            if self.class_distribution is None:
                self.class_distribution = pd.DataFrame(set_dist)
            else:
                self.class_distribution = self.class_distribution.join(set_dist, how='outer')
        self.class_distribution = self.class_distribution.fillna(0)
    
    def subset(self, classes: list, keep_intersecting: bool=False):
        """Subset object categories.
        
        Arguments:
            classes {list} -- List of classes to keep. These must all be existing categories in the
                dataset.

        Keywork Arguments:
            keep_intersecting {bool} -- Whether or not to keep intersecting classes. When true, first
                finds all images in which annotations for `classes` exist, then also includes all
                other annotations found in those images. (default: {False})
        
        Raises:
            Exception: If not all `classes` exist in dataset.
        """

        for cls in classes:
            if cls not in self.categories.name.values:
                raise Exception(f'"{cls}" not found in current dataset')

        # Apply subset
        subset_categories = self.categories.loc[self.categories.name.isin(classes)]
        subset_annotations = self.annotations.join(subset_categories[[]], how='inner')
        subset_annotations.index.name = 'category_id'
        self.images = self.images.filter(subset_annotations.image_id.unique(), axis=0)

        if not keep_intersecting:
            self.annotations = subset_annotations
            self.categories = subset_categories
        else:
            self.annotations = self.annotations.join(self.images[[]], on='image_id', how='inner')

        self.analyze()
        self.transformations[len(self.transformations)] = ('Subset classes', str(classes))

        return self
    
    def rename(self, mapping: dict):
        """Modify category names. When mapping multiple existing categories to the same name,
        they will be combined into a single category.
        
        Arguments:
            mapping {dict} -- Mapping from old to new names. E.g., {'old_name': 'new_name', ...}
        """

        for cls in mapping:
            if cls not in self.categories.name.values:
                raise Exception(f'"{cls}" not found in current dataset')

        self.categories['new_name'] = self.categories.name
        self.categories['new_id'] = self.categories.index
        for old_name, new_name in mapping.items():
            mask = self.categories.name == old_name
            self.categories.loc[mask, 'new_name'] = new_name
            new_mask = self.categories.new_name == new_name
            self.categories.loc[new_mask, 'new_id'] = self.categories.loc[new_mask].index.min().astype(int)

        self.annotations.drop(columns='name', inplace=True)
        self.annotations = self.annotations.join(self.categories[['new_name', 'new_id']], how='inner')
        self.annotations.rename(columns={'new_name': 'name', 'new_id': 'category_id'}, inplace=True)
        self.annotations.set_index('category_id', inplace=True)

        self.categories.name = self.categories.new_name
        self.categories.set_index('new_id', inplace=True)
        self.categories.drop(columns='new_name', inplace=True)
        self.categories.index.name = 'id'
        self.categories.drop_duplicates(inplace=True)

        self.analyze()
        self.transformations[len(self.transformations)] = ('Rename classes',  str(mapping))

        return self
    
    def sample(self, n_train: int=None, n_val: int=None):
        """Sample a fixed number of images from dataset
        
        Keyword Arguments:
            n_train {int} -- Number of training images to sample. (default: {None})
            n_val {int} -- Number of validation images to sample. (default: {None})
        
        Returns:
            Builder -- Sampled dataset builder object.
        """

        if n_train is not None:
            images_train = self.images.loc[self.images.set == 'train'].sample(n_train)
        else:
            images_train = self.images.loc[self.images.set == 'train']
        if n_val is not None:
            images_val = self.images.loc[self.images.set == 'val'].sample(n_val)
        else:
            images_val = self.images.loc[self.images.set == 'val']
        self.images = pd.concat((images_train, images_val))
        self.annotations = self.annotations.merge(self.images[[]], left_on='image_id', right_index=True)
        self.annotations.index.name = 'category_id'

        self.analyze()
        self.transformations[len(self.transformations)] = ('Sample', [n_train, n_val])

        return self

    def split(self, val_frac):
        """Apply train-val split to dataset images.
        
        Arguments:
            val_frac {float} -- Proportion of data to use for validation.
        """

        # Apply random split to images
        self.images.set = np.random.choice(
            ['train', 'val'],
            p=[1 - val_frac, val_frac],
            size=len(self.images)
        )

        # Join new image split to annotations
        self.annotations.drop(columns='set', inplace=True)
        self.annotations['category_id'] = self.annotations.index
        self.annotations.set_index('image_id', inplace=True)
        self.annotations = self.annotations.join(self.images[['set']])
        self.annotations['image_id'] = self.annotations.index
        self.annotations.set_index('category_id', inplace=True)

        self.analyze()
        self.transformations[len(self.transformations)] = ('Split', val_frac)

        return self

    def merge(self, other, names=('data0', 'data1')):
        """Merge two datasets. Unlink most other methods, this does not modify either dataset in-place.
        Hence, the result should be captured in a variable.
        
        Arguments:
            other {Builder} -- Other dataset builder object.
        
        Keyword Arguments:
            names {list} -- Names of two source datasets. If not specified (default: {('data0', 'data1')})
        
        Raises:
            TypeError: If 'other' is not of Builder type.
            Exception: If category ids cannot be merged correctly.
        
        Returns:
            Builder -- Merged dataset builder.
        """

        if not isinstance(other, Builder):
            raise TypeError('"other" must be of a type that inherits from "Builder"')
        
        # Make copy of left dataset
        merged = copy.deepcopy(self)

        # Combine metadata
        merged.source.extend(other.source)
        merged.source_id.extend(other.source_id)
        merged.info.extend(copy.deepcopy(other.info))
        merged.licenses = pd.concat((merged.licenses, other.licenses), sort=True)
        merged.image_paths.extend(other.image_paths)

        # Build new category table, updating category ids in other dataset as needed
        current_max = merged.categories.index.max()
        for i, row in other.categories.iterrows():
            if row['name'] not in merged.categories.name.values:
                current_max += 1
                row.name = current_max
                merged.categories = merged.categories.append(row)
        merged.categories['new_id'] = merged.categories.index

        # Update category ids in other annotations
        other_annotations = other.annotations.merge(merged.categories[['name', 'new_id']], on='name', how='left')
        merged.categories.drop(columns='new_id', inplace=True)
        if other_annotations.new_id.isna().any():
            raise Exception('Problem with merge: missing category ids')
        other_annotations.set_index('new_id', inplace=True)
        other_annotations.index.name = 'category_id'

        # Combine annotations
        merged.annotations = pd.concat((merged.annotations, other_annotations), sort=True)
        merged.annotations = merged.annotations[~merged.annotations.id.duplicated(keep='first')]

        # Combine images
        merged.images = pd.concat((merged.images, other.images), sort=True)
        merged.images = merged.images[~merged.images.index.duplicated(keep='first')]

        merged.analyze()
        merged.transformations = {
            0: {names[0]: copy.deepcopy(merged.transformations), names[1]: copy.deepcopy(other.transformations)},
            1: 'Merge datasets'
        }

        return merged
    
    def build(self, target_dir: str, use_links=True):
        """Build defined dataset. Annotation JSON files are exported and images are copied
        using hard links.
        
        Arguments:
            target_dir {str} -- Target directory for dataset.

        Keyword arguments:
            use_links {bool} -- Whether to create copies or hard links to image files.
                (default: {True})
        
        Raises:
            Exception: When target directory is not empty.
        """

        if use_links:
            cp_fn = os.link
        else:
            cp_fn = shutil.copy

        os.makedirs(target_dir, exist_ok=True)
        if len(os.listdir(target_dir)) > 0:
            raise Exception('Target directory is not empty')
        os.makedirs(os.path.join(target_dir, 'annotations'))
        for split in ['train', 'val']:
            os.makedirs(os.path.join(target_dir, split))
            for source_id in np.unique(self.source_id):
                os.makedirs(os.path.join(target_dir, split, source_id))
            
        instances_train = {
            'info': self.info,
            'licenses': self.licenses.to_dict(orient='records'),
            'categories': self.categories.reset_index().to_dict(orient='records'),
            'images': []
        }
        instances_val = copy.deepcopy(instances_train)
        instances_train['annotations'] = (
            self.annotations
                .loc[self.annotations.set == 'train']
                .drop(columns=['set', 'name'])
                .reset_index()
                .to_dict(orient='records')
        )
        instances_val['annotations'] = (
            self.annotations
                .loc[self.annotations.set == 'val']
                .drop(columns=['set', 'name'])
                .reset_index()
                .to_dict(orient='records')
        )

        for i, row in tqdm(self.images.reset_index().iterrows(), desc='Building dataset'):
            image = row.to_dict()
            src_path = image['file_name']
            dest_path = os.path.join(image['id'].split('_')[0], image['file_name'])
            cp_fn(
                os.path.abspath(
                    os.path.join(image['source'],
                    self.image_paths[self.source.index(image['source'])][image['set']], src_path)
                ),
                os.path.join(target_dir, image['set'], dest_path)
            )
            image['file_name'] = dest_path
            if image['set'] == 'train':
                instances_train['images'].append(image)
            else:
                instances_val['images'].append(image)

        print('Saving annotation JSON files')
        with open(os.path.join(target_dir, 'annotations/instances_train.json'), 'w') as f:
            json.dump(instances_train, f)
        with open(os.path.join(target_dir, 'annotations/instances_val.json'), 'w') as f:
            json.dump(instances_val, f)

        self.analyze()
        self.transformations[len(self.transformations)] = ('Build', target_dir)
