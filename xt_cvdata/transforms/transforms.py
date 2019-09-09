__all__ = [
    'Relabel',
    'ToLabel',
    'Squeeze',
    'OneHot'
]

import torch 
import numpy as np

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