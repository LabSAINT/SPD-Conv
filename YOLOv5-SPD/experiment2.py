"""
Experimental Modules
"""

import logging
import math
import warnings
from copy import copy
from pathlib import Path
import sys
from models.yolo import Model

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative


from utils.datasets import exif_transpose, letterbox
from utils.general import colorstr, increment_path, make_divisible, non_max_suppression, save_one_box, \
    scale_coords, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync

LOGGER = logging.getLogger(__name__)


def autopad(k, p=None):  # kernel, padding
	# Pad to 'same'
	if p is None:
		p = k //2 if isinstance(k, int) else [x//2 for x in k] # autopad

	return p

class Conv(nn.Module):
	# standard convolution
	def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True): 
		#ch_in, ch_ out, kernel, stride, padding, groups
		super().__init__()
		self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
		self.bn = nn.BatchNorm2d(c2)
		self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Model) else nn.Identity())

	def forward(self,x):
		return self.act(self.bn(self.conv(x)))

	def forward_fuse(self,x):
		return self.act(self.conv(x))

class DWConv(Conv):
	# Depth-wise convolution class

	def __init_(self, c1, c2, k=1, s=1, act=True): 
		super().__init__(c1,c2,k,s,g=math.gcd(c1,c2),act=act)


class TransformerLayer(nn.Module):
	# Transforment layer
	
	def __init__(self, c, num_heads):
		super().__init__()
		self.q = nn.Linear(c, c, bias = False)
		self.k = nn.Linear(c, c, bias=False)
		self.v = nn.Linear(c, c, bias=False)
		self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
		self.fc1 = nn.Linear(c, c, bias=False)
		self.fc2 = nn.Linear(c, c, bias=False)

	def forward(self,x):
		x= self.ma(self.q(x),self.k(x),self.v(x))[0] + x
		x = self.fc2(self.fc1(x)) + x
		return x

class TransformerBlock(nn.Module):
	
	def __init__(self, c1, c2, num_heads, num_layers):
		super().__init__()
		self.conv = None
		if c1 != c2:
			self.conv = Conv(c1,c2)
		self.linear = nn.Linear(c2, c2)  # learnable position embedding
		self.tr=nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2=c2

	def forward(self,x):
		if self.conv is not None:
			x= self.conv(x)

		b, _, w, h = x.shape

		p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
		return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)

		
class Bottleneck(nn.Module):
	# Standard bottleneck

	def __init__(self, c1, c2, shortcut= True, g=1, e=0.5):
		# ch_in, ch_out, shortcut, groups, expansion
		super().__init__()
		c_ = int(c2*e) # hidden channels

		self.cv1 = Conv(c1, c_, 1, 1)
		self.cv2 = Conv(c_, c2, 3, 1, g=g)
		self.add = shortcut and c1==c2

	def forward(self, x):

		return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
	# CSP Bottleneck

	def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
		# ch_in, ch_out, number, shortcut, groups, expansion
		super().__init__()
		c_ = int(c2*e) # hidden channels
		self.cv1 = Conv(c1, c_, 1, 1)
		self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias = False)
		self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
		self.cv4 = Conv(2 * c_, c2, 1, 1)
		self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
		self.act = nn.LeakyReLU(0.1, inplace=True)
		self.m = nn.Sequential(*[Bottleneck(c_,c_,shortcut, g, e=1.0) for _ in range(n)])

	def forward(self,x):
		y1 =  self.cv3(self.m(self.cv1(x)))
		y2 = self.cv2(x)

		return  self.cv4(self.act(self.bn(torch.cat((y1,y2),dim=1))))

class C3(nn.Module):
	# CSP Bottleneck with 3 convolutions
	def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
		# ch_in, ch_out, number, shortcut, groups, expansion
		super().__init__()
		c_ = int(c2 * e)  # Hidden channels
		self.cv1 = Conv(c1, c_, 1, 1)
		self.cv2 = Conv(c1, c_, 1, 1)
		self.cv3 = Conv(2* c_, c2, 1)
		self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

	def forward(self,x):
		return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
	# C3 module with TransformerBlock()

	def __init__(self,c1, c2, n=1, shortcut= True, g=1, e=0.5):
		super().__init__(c1, c2, n=1, shortcut, g=1, e)
		c_ = int(c2*e)
		self.m = TransformerBlock(c_, c_, 4, n)

class SPP(nn.Module):
	# spatial pyramid pooling layer

	def __init__(self, c1, c2, k=(5,9,13)):
		super().__init__()
		c_ = c1 //2 # hidden channels

		self.cv1 = Conv(c1, c_, 1, 1)
		self.cv2 = Conv(c_ *(len(k)+1), c2, 1, 1)
		self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding = x//2 ) for x in k])

	def forward(self, x):
		x = self.cv1(x)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')  #  suppress torch 1.9.0 max_pool2d() warning
			return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPPF(nn.Module):
	# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOV5

	def __init__(self,c1,c2, k=5):
		super().__init__()

		c_ = c1 // 2
		self.cv1 = Conv(c1, c_, 1, 1)
		self.cv2 = Conv(c_ *4, c2, 1, 1)
		self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k //2)

	def forward(self,x):
		x = self.cv1(x)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			y1= self.m(x)
			y2 = self.m(y1)
			return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class Focus(nn.Module):
	# Focus wh information into c-space
	def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
		super().__init__
		self.conv(c1*4, c2, k, s, p, g, act)

	def forward(self,x): # x(b,c,w,h) -> y(b,4c,w/2,h/2)
		return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class GhostBottleneck(nn.Module):
	# Chost bottleneck

	def __init__(self, c1, c2, k=3, s=1):
		super().__init__()
		c_ = c2 // 2
		self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),
		                  DWConv(c_,c_,k,s, act=False) if s==2 else nn.Identity(),
						  GhostConv(c_, c2, 1, 1, act=False))

		self.shortcut = nn.Sequential(DWConv(c2, c1, k, s, act= False),
		Conv(c1, c2, 1, 1, act=False) if s == 2 else nn.Identity())


	def forward(self,x):
		return self.conv(x) + self.shortcut(x)


class GhostConv(nn.Module):
	# Chost convolution
	def __init__(self, c1, c2, k=1, s=1, g=1, act= True):
		super().__init__()
		c_ = c1 // 2
		self.cv1 = Conv(c1, c_, k, s, None, g, act)
		self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

	def forward(self,x):
		y = self.cv1(x)
		return torch.cat([y, self.cv2(y)], 1)


class Contract(nn.Module):
	# contract width=height into channels, i.e x(1,64,80,80) to x(1,256,40,40)

	def __init__(self,gain=2):
		super().__init__()
		self.gain = gain
	
	def forward(self,x):
		b, c, h, w = x.size()
		s = self.gain
		x = x.view(b, c, h //s, s, w //s, s)
		x =x.permute(0, 3, 5, 1, 2, 4).contiguous()
		return x.view(b, c * s *s, h//s, w//s) 

class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
	# Concatenate a list of tensors along dimension
	def __init__(self, dimension=1):
		super().__init__()
		self.d = dimension

	def forward(self, x):
		return torch.cat(x, self.d)


class AutoShape(nn.Module):
	# YOLOv5 input- robust model warapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
	conf = 0.25  # NMS confidence threshold
	iou = 0.45   # NMS IoU threshold
	classes = None  
	multi_label = False 
	mx_det = 1000  # maximum number of detection per image

	def __init__(self, model):
		super().__init__()
		self.model = model.eval()

	def autoshape(self):
		LOGGER.info('AutoShape already enables, skiping...')

	def _apply(self,fn):
		
		self = super()._apply(fn)
		m = self.model.model[-1]
		m.stride = fn(m.stride)
		m.grid = list(map(fn, m.grid))
		if isinstance(m.anchor_grid, list):
			m.anchor_grid = list(map(fn, m.anchor_grid))
		return self

	
	@torch.no_grad()
	def forward(self, imgs, size=640, augment = False, profile= False):
		# Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

		t = [time_sync()]
		p = next(self.model.parameters())  # for device and type

		if isinstance(imgs,torch.Tensor): # torch
			with amp.autocast(enabled=p.device.type != 'cpu'):
				return self.model(imgs.to(p.device).type_as(p), augment, profile) 
			

















