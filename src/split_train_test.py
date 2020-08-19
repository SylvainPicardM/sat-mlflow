import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import shutil
from typing import Tuple

def clear_dir(path:Path) -> None:
	"""
	Remove all files in directory
	:param path: Path to directory to clean
	:return: None
	"""
	for filename in os.listdir(path):
		file_path = path / filename
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))

def read_original_files(original_path:Path) -> pd.DataFrame:
	"""
	Read all files in original data directory to pandas Dataframe
	:param original_path: PAth to original data
	:return: pandas Dataframe with columns ["path", "label", "filename"]
	"""
	files = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(original_path)] for val in sublist]
	files_data = []

	for f in files:
		label = f.split('/')[-2]
		fname = f.split('/')[-1]
		files_data.append([f, label, fname])

	files_data = np.array(files_data)
	df = pd.DataFrame(files_data, columns=['path', 'label', 'filename'])
	return df

def split_df(files_df:pd.DataFrame, split_size:float=.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""

	:param files_df:
	:param split_size:
	:return:
	"""
	labels = files_df.label.unique()
	grouped = files_df.groupby(files_df.label)

	# Split df by label value
	labels_dfs = {'original': {}, 'train': {}, 'test': {}}
	for label in labels:
		ldf = grouped.get_group(label)
		labels_dfs['original'][label] = ldf

	# For each label, split into train and test
	for k, v in labels_dfs['original'].items():
		assert k == v.label.unique()[0]
		data = v.values
		np.random.shuffle(data)
		train, test = np.split(data, [int(split_size * data.shape[0])])

		assert train.shape[0] + test.shape[0] == data.shape[0]

		labels_dfs['train'][k] = train
		labels_dfs['test'][k] = test

	# Concat all labels for train and test
	train_data = None
	test_data = None

	# STACK DATA FOR EACH LABELS OF TRAIN AND TEST
	for k, v in labels_dfs['train'].items():
		train_data = v if train_data is None else np.vstack((train_data, v))

	for k, v in labels_dfs['test'].items():
		test_data = v if test_data is None else np.vstack((test_data, v))

	assert train_data.shape[0] + test_data.shape[0] == len(files_df)

	# CONVERT BACK TO PANDAS DF
	train_data = pd.DataFrame(train_data, columns=['path', 'label', 'filename'])
	test_data = pd.DataFrame(test_data, columns=['path', 'label', 'filename'])
	return train_data, test_data


def main(args):
	original_path = Path(args.data_dir)
	output_path = Path(args.out_dir)
	train_path = output_path / 'train'
	test_path = output_path / 'test'

	# Create dir if not exists
	os.makedirs(train_path, exist_ok=True)
	os.makedirs(test_path, exist_ok=True)

	# Remove files in train and test dir if exists
	clear_dir(train_path)
	clear_dir(test_path)

	files_df = read_original_files(original_path)

	for l in files_df.label.unique():
		os.makedirs(os.path.join(output_path, "train", l), exist_ok=True)
		os.makedirs(os.path.join(output_path, "test", l), exist_ok=True)

	train_data, test_data = split_df(files_df, split_size=args.train_size)

	print('Writing training data')
	for i, row in train_data.iterrows():
		newpath = os.path.join(output_path, "train", row.label, row.filename)
		shutil.copyfile(row.path, newpath)

	print('Writing test data')
	for i, row in test_data.iterrows():
		newpath = os.path.join(output_path, "test", row.label, row.filename)
		shutil.copyfile(row.path, newpath)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", "-d", type=str, required=True, help="Path to original data")
	parser.add_argument("--out_dir", "-o", type=str, required=True, help="Path to output directory")
	parser.add_argument("--train_size", "-t", type=float, default=.7, help="Size of training set in pourcentage of total")
	args = parser.parse_args()
	main(args)
