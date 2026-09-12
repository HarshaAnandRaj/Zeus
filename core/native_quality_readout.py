"""LCM5: a new public-quality reader on an immutable qualified action/store parent."""
import torch
from torch import nn

VERSION='native-quality-reader-v1-20260913'


class NativeQualityReadout(nn.Module):
    def __init__(self,base,parent_hash):
        super().__init__();self.base=base;self.base.requires_grad_(False);self.parent_hash=parent_hash
        self.reader=nn.Linear(8,2);self.register_buffer('revision',torch.zeros((),dtype=torch.long))

    def get_extra_state(self):return dict(version=VERSION,parent_hash=self.parent_hash)
    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('incompatible native quality reader')

    def quality_logits(self,z):return self.reader(z)
