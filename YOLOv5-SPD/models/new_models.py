from random import weibullvariate
import torch
import torch.nn as nn

class AFF(nn.Module):
	"""
	Implimenting AFF module
	"""

	def __init__(self, channels=64, r=4):
		super(AFF, self).__init__()
		inter_channels = int(channels //r )

		self.local_att = nn.Sequential(
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)

		self.global_att = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)

		self.sigmoid = nn.Sigmoid()

	def forward(self, input):
		x= input[0]
		y= input[1]
		xa = x + y
		xl = self.local_att(xa)
		xg= self.global_att(xa)
		xlg = xl + xg
		m = self.sigmoid(xlg)

		x_union_y = 2* x * m + 2* y * (1-m)

		return x_union_y



class iAFF(nn.Module):

	"""
	implimenting iAFF module
	"""

	def __init__(self, channels=64, r=4):
		super(iAFF, self).__init__()
		inter_channels = int(channels // r)

		self.local_attention1 = nn.Sequential(
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)
		self.global_attention1 = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)

		self.local_attention2 = nn.Sequential(
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)
		self.global_attention2 = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)

		self.sigmoid = nn.Sigmoid()


	def forward(self, input):
		"""
		Implimenting the iAFF forward step
		"""
		x = input[0]
		y = input[1]
		xa = x+y
		xl = self.local_attention1(xa)
		xg = self.global_attention1(xa)
		xlg = xl+xg
		m1 = self.sigmoid(xlg)
		xuniony = x * m1 + y * (1-m1)

		xl2 = self.local_attention2(xuniony)
		xg2 = self.global_attention2(xuniony)
		xlg2 = xl2 + xg2
		m2 = self.sigmoid(xlg2)
		z = x * m2 + y * (1-m2)
		return z


if __name__ == '__main__':
	import os
	x = torch.randn(8,64,32,32)
	y = torch.randn(8,64,32,32)
	channels = x.shape[1]

	model = iAFF(channels=channels)
	output = model(x,y)
	print(output)
	print(output.shape)




