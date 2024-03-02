import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
class MedianPool1d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    Args:
         kernel_size: size of pooling kernel, int
         stride: pool stride, int
         padding: pool padding, int
         same: override padding and enforce same padding, boolean#
    """

    def __init__(self, kernel_size=4, stride=1, padding=2, same=True):
        super(MedianPool1d, self).__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = _pair(padding)
        self.same = same

    def _padding(self, x):
        if self.same:
            il = x.size()[1]
            # If seq_len can divide steps exactly
            if il % self.stride == 0:
                pl = max(self.k - self.stride, 0)
            else:#If seq_len does not divide the step exactly
                pl = max(self.k - (il % self.stride), 0)
            pleft = pl // 2
            pright = pl - pleft
            padding = (pleft, pright)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = x.permute(1, 0)
        x = F.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(1, self.k, self.stride)
        x = x.contiguous().median(dim=-1)[0]
        x = x.permute(1, 0)
        return x

class MinPool2d(nn.Module):#
    def __init__(self, scale):
        super(MinPool2d, self).__init__()
        self.scale=scale

    def forward(self, x):
        x = -F.max_pool2d(-x,self.scale,ceil_mode=True)
        return x

