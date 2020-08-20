import yaml
from argparse import Namespace
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import Net
import mlflow
import mlflow.pytorch
import pickle
import os


def train(epoch, model, trainloader):
	running_loss = 0.0
	tcorrect = 0
	ttotal = 0
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	for i, data in enumerate(trainloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = model(inputs)
		_, predicted = torch.max(outputs.data, 1)
		ttotal += labels.size(0)
		tcorrect += (predicted == labels).sum().item()

		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()

		if i % args.log_interval == 0:  # print every 2000 mini-batches
			print(
				f'[{epoch + 1},{i + 1}] loss: {round(running_loss / args.log_interval, 3)} accuracy: {round(100 * tcorrect / ttotal, 2)}%')
			running_loss = 0.0
			mlflow.log_metric('loss', round(running_loss / args.log_interval, 3))
			mlflow.log_metric('train_accuracy', round(100 * tcorrect / ttotal, 2))
			break


def test(epoch, model, valloader):
	vcorrect = 0
	vtotal = 0
	with torch.no_grad():
		for i, data in enumerate(valloader, 0):
			images, labels = data
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			vtotal += labels.size(0)
			vcorrect += (predicted == labels).sum().item()
	print(f'Epoch {epoch + 1} - Validation accuracy: {round(100 * vcorrect / vtotal, 2)}%')
	mlflow.log_metric('val_accuracy', round(100 * vcorrect / vtotal, 2))


def main(args):
	# GET DATA
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	dataset = datasets.ImageFolder(args.train_dir, transform=transform)

	# GET MODEL
	model = Net(n_class=10)

	# TRAINING
	print('Start Training')
	with mlflow.start_run():

		for k, v in vars(args).items():
			mlflow.log_param(k, v)

		for epoch in range(args.epochs):  # loop over the dataset multiple times
			train_size = int(.8 * len(dataset))
			test_size = len(dataset) - train_size
			train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
			trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
			valloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

			train(epoch, model, trainloader)
			test(epoch, model, valloader)


	print('Finished Training')
	mlflow.pytorch.log_model(model, artifact_path="pytorch-model", pickle_module=pickle)
	print("\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(), "pytorch-model"))



if __name__ == "__main__":
	params = yaml.safe_load(open('params.yaml'))['train']
	args = Namespace(**params)
	main(args)