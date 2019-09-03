import torch.utils.data as data
import os.path
import numpy as np
import matplotlib.cm
from PIL import Image

class VOC(data.Dataset):
    """
    Pascal VOC 2012 Dataset class. You can get the data from: 
        http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit

    Args:
        root (str): Path to data folder.
        transform (object): Data transforms to apply to images.
        target_transform (object): Data transforms to apply to labels.
        image_set (str): Which split to load, either 'train or val'

    Example:
        >>> from xt_cvdata.transforms import ToLabel, Relabel
        >>> from torchvision import transforms
        >>> 
        >>> data_transforms = transforms.Compose([
        >>>     transforms.ToTensor()
        >>> ])
        >>>
        >>> # These custom transforms are in xt_cvdata.transforms
        >>> label_transforms = transforms.Compose([
        >>>     ToLabel(),
        >>>     Relabel(255, 21)
        >>> ])
        >>> 
        >>> dataset = xcvd.datasets.VOC(
            '/nasty/data/common/VOC2012', 
            transform=data_transforms, 
            target_transform=label_transforms
            )
    """
    def __init__(self, root, transform=None, target_transform=None, image_set='train'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self._image_set = image_set
        self._imsetpath = os.path.join(self.root, 'ImageSets', 'Segmentation', '%s.txt')
        self._annopath = os.path.join(self.root, 'SegmentationClass', '%s.png')
        self._impath = os.path.join(self.root, 'JPEGImages', '%s.jpg')
 
        with open(self._imsetpath % self._image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

        self.labels = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor', 'void'
        ]
        self.cmap = self._color_map()

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = Image.open(self._annopath % img_id)
        img = Image.open(self._impath % img_id).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return len(self.ids)

    def _color_map(self, normalized=True, base_map_name='tab20'):
        """
        Custom Colormap for the labels.
        """

        base_map = matplotlib.cm.get_cmap(base_map_name, 22).colors
        cmap = np.zeros_like(base_map)
        cmap[0,-1] = 1
        cmap[1:-1] = base_map[:20]
        cmap[-1] = [1, 1, 1, 1]

        return matplotlib.colors.ListedColormap(cmap)
    
