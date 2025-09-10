import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np




class Binarize(Function):

    @staticmethod
    def forward(ctx, input, quant_mode='det', allow_scale=False):
        output = input.clone()      

        scale = output.abs().max() if allow_scale else 1

        if quant_mode=='det':
            return output.div(scale).sign().mul(scale)
        else:
            return output.div(scale).add_(1).div_(2).add_(torch.rand(output.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1).mul(scale)
        
    @staticmethod
    def backward(ctx, grad_output):
        #STE (Straight Through Estimator)
        grad_input = grad_output
        return grad_input, None, None


class Quantize(Function):
    @staticmethod
    def forward(ctx, input, quant_mode='det', numBits=4):
        output = input.clone()
        scale = (2**numBits-1)/(output.max()-output.min())
        output = output.mul(scale).clamp(-2**(numBits-1)+1, 2**(numBits-1))
        if quant_mode=='det':
            output = output.round().div(scale)
        else:
            output = output.round().add(torch.rand(output.size()).add(-0.5)).div(scale)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        #STE (Straight Through Estimator)
        grad_input = grad_output
        return grad_input, None, None

def binarized(input,quant_mode='det'):
      return Binarize.apply(input,quant_mode)  

def quantize(input,quant_mode,numBits):
      return Quantize.apply(input,quant_mode,numBits) 

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    @staticmethod
    def forward(ctx, input, target):
        margin = 1.0
        output = margin - input.mul(target)
        output[output.le(0)] = 0
        ctx.save_for_backward(input, target)
        loss = output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        margin = 1.0
        output = margin - input.mul(target)
        output[output.le(0)] = 0
        grad_input = target.mul(-2).mul(output)
        grad_input.mul_(output.ne(0).float())
        grad_input.div_(input.numel())
        grad_input = grad_input * grad_output
        return grad_input, None



class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input_b=binarized(input)
        weight_b=binarized(self.weight)
        out = nn.functional.linear(input_b,weight_b)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input_b = binarized(input)
        else:
            input_b=input
        weight_b=binarized(self.weight)

        out = nn.functional.conv2d(input_b, weight_b, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
