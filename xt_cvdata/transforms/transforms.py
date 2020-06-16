
import torch 
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from albumentations.core.transforms_interface import BasicTransform as ATransform

import random

__all__ = ['Relabel', 'ToLabel', 'Squeeze', 'OneHot', 'MaskCompose', 'XTCompose'
]

class XTCompose:
    """ Custom Composer of transforms used for applying transforms to the img, mask, or both.
    Can use both Albuentations transforms, or any transform on PIL images

    Basic Usage:
    >>> transforms = XTCompose([
        GaussianBlur(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    Advanced Usage:
    You can supply the transforms as a tuple if you want to use it for segmentation
    >>> transforms = MaskCompose([
        (Resize(24, 24), 'both'), 
        (ColorJitter(), 'img'), 
        (lambda x: x.long(), 'mask')]
    )
    """
    target_options = ['img', 'mask', 'both']
    def __init__(self, transforms):
        self.transforms = transforms

        assert all([
            t_target[1] in self.target_options 
            for t_target in transforms 
            if isinstance(t_target, tuple)
        ])
    
    @staticmethod 
    def convert_to_proper_type(img, to_numpy):
        """Converts a PIL or ndarray to the proper type before a transformation

        Args:
            img (PIL.Image or np.ndarray): The image to be converted
            to_numpy (bool): Convert to np array or to PIL Image

        Returns:
            PIL.Image or np.ndarray: The converted image
        """

        if to_numpy:
            if not isinstance(img, np.ndarray):
                img = np.asarray(img)
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)

        return img


    def __call__(self, img, mask=None):
        for t_target in self.transforms:
            if isinstance(t_target, tuple):
                t, target = t_target
            else: 
                t = t_target
                target = 'img'

            is_album = isinstance(t, ATransform)

            if target == 'img':
                img = self.convert_to_proper_type(img, is_album)
                if is_album:
                    img = t(image=img)['image']
                else:
                    img = t(img)
            elif target == 'mask':
                mask = self.convert_to_proper_type(mask, is_album)
                if is_album:
                    mask = t(image=mask)['image']
                else:
                    mask = t(mask)
            else:
                img = self.convert_to_proper_type(img, is_album)
                mask = self.convert_to_proper_type(mask, is_album)
                if is_album:
                    augmented = t(image=img, mask=mask)
                    img = augmented['image']
                    mask = augmented['mask']
                else:
                    # Concat Mask, Transform, Split
                    img.putalpha(mask)
                    img = t(img)
                    red, green, blue, mask = img.split()
                    img = Image.merge('RGB', [red, green, blue])

        if mask is None:
            return img
        
        return img, mask


class Relabel:
    """ Transform to replace all occurances of `old_label` with `new_label` in a torch.tensor.
    
    Relabel(old_label, new_label)
    """

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor

class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)

class Squeeze:
    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, x):
        return torch.squeeze(x, dim=self.dim)

class OneHot:

    def __init__(self, C):
        self.C = C

    def __call__(self, labels):
        '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.
        
        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size. 
            Each value is an integer representing correct classification.
        C : integer. 
            number of classes in labels.
        
        Returns
        -------
        target : torch.autograd.Variable of torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        '''


        one_hot = torch.LongTensor(self.C, labels.size(1), labels.size(2)).zero_()
        target = one_hot.scatter_(0, labels.data, 1)
        
        target = torch.autograd.Variable(target)
            
        return target



# For backwards compatability
MaskCompose = XTCompose