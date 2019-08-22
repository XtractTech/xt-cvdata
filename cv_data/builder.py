import numpy as np
import pandas as pd
import json
import copy

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
    
    def subset(self, classes):
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
        self.images = self.images.filter(self.annotations.image_id.unique(), axis=0)

        self.analyze()
        self.transformations[len(self.transformations)] = ('Subset classes', str(classes))
    
    def rename(self, mapping):
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

        self.analyze()
        self.transformations[len(self.transformations)] = ('Rename classes',  str(mapping))

    def merge(self, other):
        self.info = {1: self.info, 2: other.info}
        self.licenses = {1: self.licenses, 2: other.licenses}

        current_max = self.categories.index.max()
        for i, row in other.categories.iterrows():
            if row['name'] not in self.categories.name.values:
                current_max += 1
                row.name = current_max
                self.categories = self.categories.append(row)
        self.categories['new_id'] = self.categories.index

        other.annotations = other.annotations.merge(self.categories[['name', 'new_id']], on='name', how='left')
        self.categories.drop(columns='new_id', inplace=True)
        if other.annotations.new_id.isna().any():
            raise Exception('Problem with merge: missing category ids')
        other.annotations.set_index('new_id', inplace=True)
        other.annotations.index.name = 'category_id'

        self.annotations = pd.concat((self.annotations, other.annotations))

        self.analyze()
        self.transformations = {
            0: {1: copy.deepcopy(self.transformations), 2: copy.deepcopy(other.transformations)},
            1: 'Merge datasets'
        }
        

