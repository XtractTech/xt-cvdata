# xt-cvdata
  
## Description

This repo contains utilities for building and working with computer vision datasets, developed by [Xtract AI](https://xtract.ai/).

So far, APIs for the following open-source datasets are included:
1. COCO 2017 (detection and segmentation): `xt_cvdata.apis.COCO`
1. Open Images V5 (detection and segmentation): `xt_cvdata.apis.OpenImages`
1. Visual Object Tagging Tool (VoTT) CSV output (detection): `xt_cvdata.apis.VoTTCSV`

More to come.

## Installation

From PyPI:
```bash
pip install xt-cvdata
```

From source:
```bash
git clone https://github.com/XtractTech/xt-cvdata.git
pip install ./xt-cvdata
```

## Usage

See specific help on a dataset class using `help`. E.g., `help(xt_cvdata.apis.COCO)`.

#### Building a dataset

```python
from xt_cvdata.apis import COCO, OpenImages

# Build an object populated with the COCO image list, categories, and annotations
coco = COCO('/nasty/data/common/COCO_2017')
print(coco)
print(coco.class_distribution)

# Same for Open Images
oi = OpenImages('/nasty/data/common/open_images_v5')
print(oi)
print(coco.class_distribution)

# Get just the person classes
coco.subset(['person'])
oi.subset(['Person']).rename({'Person': 'person'})

# Merge and build
merged = coco.merge(oi)
merged.build('./data/new_dataset_dir')
```

This package follows pytorch chaining rules, meaning that methods operating on an object modify it in-place, but also return the modified object. The exception is the `merge()` method which does not modify in-place and returns a new merged object. Hence, the above operations can also be completed using:

```python
from xt_cvdata.apis import COCO, OpenImages

merged = (
    COCO('/nasty/data/common/COCO_2017')
        .subset(['person'])
        .merge(
            OpenImages('/nasty/data/common/COCO_2017')
                .subset(['Person'])
                .rename({'Person': 'person'})
        )
)
merged.build('./data/new_dataset_dir')
```

In practice, somewhere between the two approaches will probably be most readable.

The current set of dataset operations are:
* `analyze`: recalculate dataset statistics (e.g., class distributions, train/val split)
* `verify_schema`: check if class attributes follow required schema
* `subset`: remove all but a subset of classes from the dataset
* `rename`: rename/combine dataset classes
* `sample`: sample a specified number of images from the train and validation sets
* `split`: define the proportion of data in the validation set
* `merge`: merge two datasets together, returning merged dataset
* `build`: create the currently defined dataset using either symlinks or by copying images

#### Implementing a new dataset type

New dataset types should inherit from the base `xt_cvdata.Builder` class. See the `Builder`, `COCO` and `OpenImages` classes as a guide. Specifically, the class initializer should define `info`, `licenses`, `categories`, `annotations`, and `images` attributes such that `self.verify_schema()` runs without error. This ensures that all of the methods defined in the `Builder` class will operate correctly on the inheriting class.
  
## Data Sources

[descriptions and links to data]
  
## Dependencies/Licensing

[list of dependencies and their licenses, including data]

## References

[list of references]
