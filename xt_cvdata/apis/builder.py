import os, shutil
import json
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import maskUtils


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
        self.class_distribution = pd.DataFrame()
        for s in class_dist.index.levels[0]:
            set_dist = class_dist[s]
            set_dist = pd.DataFrame({s: set_dist, f'{s}_prop': set_dist / set_dist.sum()})
            self.class_distribution = self.class_distribution.join(set_dist, how='outer')
        self.class_distribution = self.class_distribution.fillna(0)
    
    def subset(self, classes: list, keep_images: bool=False, keep_intersecting: bool=False):
        """Subset object categories.
        
        Arguments:
            classes {list} -- List of classes to keep. These must all be existing categories in the
                dataset.

        Keywork Arguments:
            keep_images {bool} -- Whether or not to keep images in the dataset that end up with no
                annotations. (default: {False})
            keep_intersecting {bool} -- Whether or not to keep intersecting classes. When true,
                first finds all images in which annotations for `classes` exist, then also includes
                all other annotations found in those images. (default: {False})
        
        Raises:
            Exception: If not all `classes` exist in dataset.
        
        Returns:
            Builder -- Modified dataset builder object.

        Example:
        >>> api = Builder(<source>)
        >>> api.subset(['person', 'cat', 'dog'])
        """

        if isinstance(classes, str):
            classes = [classes]

        for cls in classes:
            if cls not in self.categories.name.values:
                raise Exception(f'"{cls}" not found in current dataset')

        # Apply subset
        subset_categories = self.categories.loc[self.categories.name.isin(classes)]
        subset_annotations = self.annotations.join(subset_categories[[]], how='inner')
        subset_annotations.index.name = 'category_id'
        subset_images = self.images.filter(subset_annotations.image_id.unique(), axis=0)

        if not keep_intersecting:
            self.annotations = subset_annotations
            self.categories = subset_categories
        else:
            self.annotations = self.annotations.join(subset_images[[]], on='image_id', how='inner')
        
        if not keep_images:
            self.images = subset_images

        self.analyze()
        self.transformations[len(self.transformations)] = ('Subset classes', str(classes))

        return self
    
    def rename(self, mapping: dict):
        """Modify category names. When mapping multiple existing categories to the same name,
        they will be combined into a single category.
        
        Arguments:
            mapping {dict} -- Mapping from old to new names. E.g., {'old_name': 'new_name', ...}
        
        Returns:
            Builder -- Modified dataset builder object.

        Example:
        >>> api = Builder(<source>)
        >>> api.rename({'person': 'human', 'cat': 'pet', 'dog': 'pet'})
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
            self.categories.loc[new_mask, 'new_id'] = (
                self.categories.loc[new_mask].index.min().astype(int)
            )

        self.annotations.drop(columns='name', inplace=True)
        self.annotations = self.annotations.join(
            self.categories[['new_name', 'new_id']],
            how='inner'
        )
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
    
    def sample(self, n):
        """Sample a fixed number of images from dataset. This method does not shift images between
        data splits (e.g., from train to val), only samples within them. Hence, if there are
        currently 1000 images in train and 100 in val, up to 100 images can be sampled for the val
        set.
        
        Keyword Arguments:
            n {dict or int} -- Number of images to sample if an int is passed, or number
                of images to sample from each image set if a dict is passed. (default: {{}}})
        
        Returns:
            Builder -- Modified dataset builder object.

        Example:
        >>> api = Builder(<source>)
        >>> # Overall sampling
        >>> api.sample(10000)
        >>> # Set-specific sampling
        >>> api.sample({'train': 1000, 'val': 10, 'test': 100})
        """

        if isinstance(n, int):
            self.images = self.images.sample(n)
        else:
            for split, split_n in n.items():
                images_split = self.images.loc[self.images.set == split].sample(split_n)
                self.images = pd.concat((self.images[self.images.set != split], images_split))

        self.annotations = self.annotations.merge(
            self.images[[]], left_on='image_id', right_index=True
        )
        self.annotations.index.name = 'category_id'

        self.analyze()
        self.transformations[len(self.transformations)] = ('Sample', n)

        return self

    def split(self, sets: list, p: list, seed: int=None):
        """Apply a random split to dataset images. This will redistribute the existing images
        between the specified splits (train, val, test, etc.).
        
        Uses numpy's `random.choice` to generate the new split.
        
        Arguments:
            sets {list} -- List of names for image sets (e.g., ['train', 'val', 'test']).
            p {list} -- List of proportions to assign to each set in `sets`. Should sum to 1.
        
        Keywork Arguments:
            seed {int} -- Can supply an int to be used as the random seed. (default: {None})
        Returns:
            Builder -- Modified dataset builder object.

        Example:
        >>> api = Builder(<source>)
        >>> api.split(['train', 'val', 'test'], [0.8, 0.1, 0.1])
        """

        # Save original set distribution (to enable finding source image files later)
        if 'orig_set' not in self.images.columns:
            self.images['orig_set'] = self.images.set

        # Apply random split to images
        if seed:
            np.random.seed(seed)
        self.images.set = np.random.choice(sets, p=p, size=len(self.images))

        # Join new image split to annotations
        self.annotations.drop(columns='set', inplace=True)
        self.annotations['category_id'] = self.annotations.index
        self.annotations.set_index('image_id', inplace=True)
        self.annotations = self.annotations.join(self.images[['set']])
        self.annotations['image_id'] = self.annotations.index
        self.annotations.set_index('category_id', inplace=True)

        self.analyze()
        self.transformations[len(self.transformations)] = ('Split', sets, p)

        return self

    def merge(self, other, names=('data0', 'data1')):
        """Merge two datasets. Unlink most other methods, this does not modify either dataset
        in-place. Hence, the result should be captured in a variable.
        
        Arguments:
            other {Builder} -- Other dataset builder object.
        
        Keyword Arguments:
            names {list} -- Names of two source datasets. (default: {('data0', 'data1')})
        
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
        other_annotations = other.annotations.merge(
            merged.categories[['name', 'new_id']],
            on='name',
            how='left'
        )
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
            0: {
                names[0]: copy.deepcopy(merged.transformations),
                names[1]: copy.deepcopy(other.transformations)
            },
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

        # Assign copy function based on use_links
        if use_links:
            cp_fn = os.link
        else:
            cp_fn = shutil.copy

        # Get unique sets
        splits = self.images.set.unique()

        # Create/check target directory
        os.makedirs(target_dir, exist_ok=True)
        if len(os.listdir(target_dir)) > 0:
            raise Exception('Target directory is not empty')
        os.makedirs(os.path.join(target_dir, 'annotations'))
        
        for split in splits:
            os.makedirs(os.path.join(target_dir, split))
            for source_id in np.unique(self.source_id):
                os.makedirs(os.path.join(target_dir, split, source_id))

            # Initialize annotation JSON for this split
            instances_split = {
                'info': self.info,
                'licenses': self.licenses.to_dict(orient='records'),
                'categories': self.categories.reset_index().to_dict(orient='records'),
                'images': []
            }
            instances_split['annotations'] = (
                self.annotations
                    .loc[self.annotations.set == split]
                    .drop(columns=['set', 'name'])
                    .reset_index()
                    .to_dict(orient='records')
            )

            images_split = self.images.loc[self.images.set == split].reset_index().iterrows()

            # Copy images
            for i, row in tqdm(images_split, desc=f'Building {split} dataset'):
                image = row.dropna().to_dict()
                source = image['source']

                # Find set location in original dataset (e,g., train, val, or test)
                source_ind = self.source.index(source)
                source_set = self.image_paths[source_ind][image.get('orig_set', split)]

                src_path = os.path.abspath(os.path.join(source, source_set, image['file_name']))
                dest_path = os.path.join(
                    target_dir,
                    split,
                    image['id'].split('_')[0],
                    image['file_name']
                )

                # Copy image and add to annotation JSON
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                cp_fn(src_path, dest_path)
                image['file_name'] = os.path.join(image['id'].split('_')[0], image['file_name'])
                instances_split['images'].append(image)

            print('Saving annotation JSON files')
            with open(os.path.join(target_dir, f'annotations/instances_{split}.json'), 'w') as f:
                json.dump(instances_split, f)

        self.analyze()
        self.transformations[len(self.transformations)] = ('Build', target_dir)
    
    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        im = self.images.xs(ann.image_id)
        h, w = im.height, im.width
        segm = ann.segmentation
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann):
        rle = self.annToRLE(ann)
        m = maskUtils.decode(rle)
        
        return m
