import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Net(nn.Module):
	def __init__(self, n_class):
		super(Net, self).__init__()
		self.n_class = n_class
		self.encoder = models.alexnet(pretrained=True)
		self.classifier = nn.Sequential(
			nn.Linear(1000, 100),
			nn.ReLU(),
			nn.Linear(100, self.n_class)
		)

	def forward(self, x):
		f = self.encoder(x)
		return self.classifier(f)