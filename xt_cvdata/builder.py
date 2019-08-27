import os
import json
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm


class Builder(object):

    def __init__(self):
        """Base object for building and modifying object detection and segmentation datasets."""

        raise NotImplementedError
        
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
    
    def analyze(self):
        """Compute dataset statistics. This is called each time the dataset object is modified."""

        # Get aggregate statistics
        self.num_classes = len(self.categories)

        agg_fn = lambda l: {'set': [('size', 'size'), ('prop', lambda x: len(x) / l)]}
        self.num_images = self.images.groupby('set').agg(agg_fn(len(self.images)))
        self.num_annotations = self.annotations.groupby('set').agg(agg_fn(len(self.annotations)))

        # Get class-wise statistics
        class_dist = self.annotations.groupby(['set', 'name']).size()
        self.class_distribution = pd.DataFrame({
            'train': class_dist['train'],
            'val': class_dist['val']
        })
        self.class_distribution['train_prop'] = self.class_distribution.train / self.class_distribution.train.sum()
        self.class_distribution['val_prop'] = self.class_distribution.val / self.class_distribution.val.sum()
    
    def subset(self, classes: list):
        """Subset object categories.
        
        Arguments:
            classes {list} -- List of classes to keep. These must all be existing categories in the
                dataset.
        
        Raises:
            Exception: If not all `classes` exist in dataset.
        """

        for cls in classes:
            if cls not in self.categories.name.values:
                raise Exception(f'"{cls}" not found in current dataset')

        # Apply subset
        self.categories = self.categories.loc[self.categories.name.isin(classes)]
        self.annotations = self.annotations.join(self.categories[[]], how='inner')
        self.annotations.index.name = 'category_id'
        self.images = self.images.filter(self.annotations.image_id.unique(), axis=0)

        self.analyze()
        self.transformations[len(self.transformations)] = ('Subset classes', str(classes))

        return self
    
    def rename(self, mapping: dict):
        """Modify category names.
        
        Arguments:
            mapping {dict} -- Mapping from old to new names. E.g., {'old_name': 'new_name', ...}
        """

        for cls in mapping:
            if cls not in self.categories.name.values:
                raise Exception(f'"{cls}" not found in current dataset')

        map_fn = lambda n: mapping.get(n, n)
        self.categories.name = self.categories.name.map(map_fn)
        self.annotations.drop(columns='name', inplace=True)
        self.annotations = self.annotations.join(self.categories[['name']], how='inner')
        self.annotations.index.name = 'category_id'

        self.analyze()
        self.transformations[len(self.transformations)] = ('Rename classes',  str(mapping))

        return self
    
    def sample(self, n_train: int=None, n_val: int=None):
        """Sample a fixed number of images from dataset
        
        Keyword Arguments:
            n_train {int} -- Number of training images to sample. (default: {None})
            n_val {int} -- Number of validation images to sample (default: {None})
        
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
        

    def merge(self, other, names=('data0', 'data1')):
        """Merge two datasets.
        
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
        
        # Combine metadata
        self.source.extend(other.source)
        self.source_id.extend(other.source_id)
        self.info.extend(copy.deepcopy(other.info))
        self.licenses.extend(copy.deepcopy(other.licenses))

        # Build new category table, updating category ids in other dataset as needed
        current_max = self.categories.index.max()
        for i, row in other.categories.iterrows():
            if row['name'] not in self.categories.name.values:
                current_max += 1
                row.name = current_max
                self.categories = self.categories.append(row)
        self.categories['new_id'] = self.categories.index

        # Update category ids in other annotations
        other_annotations = other.annotations.merge(self.categories[['name', 'new_id']], on='name', how='left')
        self.categories.drop(columns='new_id', inplace=True)
        if other_annotations.new_id.isna().any():
            raise Exception('Problem with merge: missing category ids')
        other_annotations.set_index('new_id', inplace=True)
        other_annotations.index.name = 'category_id'

        # Combine annotations
        self.annotations = pd.concat((self.annotations, other_annotations), sort=True)
        self.annotations = self.annotations[~self.annotations.id.duplicated(keep='first')]

        # Combine images
        self.images = pd.concat((self.images, other.images), sort=True)
        self.images = self.images[~self.images.index.duplicated(keep='first')]

        self.analyze()
        self.transformations = {
            0: {names[0]: copy.deepcopy(self.transformations), names[1]: copy.deepcopy(other.transformations)},
            1: 'Merge datasets'
        }

        return self
    
    def build(self, target_dir: str):
        """Build defined dataset. Annotation JSON files are exported and images are copied
        using symlinks.
        
        Arguments:
            target_dir {str} -- Target directory for dataset.
        
        Raises:
            Exception: When target directory is not empty.
        """

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
            'licenses': self.licenses,
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

        for i, row in tqdm(self.images.reset_index().iterrows(), desc='Generating symlinks'):
            image = row.to_dict()
            src_path = image['file_name']
            dest_path = os.path.join(image['id'].split('_')[0], image['file_name'])
            os.symlink(
                os.path.abspath(
                    os.path.join(image['source'], self.image_paths[image['set']], src_path)
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
