import torch

import codecs
import numpy as np
import pickle as pkl
import os


class BaseLoader(object):
	def __init__(self, mode, split_file, res=32, pointcloud_samples=3000, data_path=None, suffix='',
	             batch_size=64, num_sample_points=1024, num_workers=12, sample_distribution=[1], sample_sigmas=[0.005],
	             ext='', cache_suffix=None):
		# sample distribution should contain the percentage of uniform samples at index [0]
		# and the percentage of N(0,sample_sigma[i-1]) samples at index [i] (>0).
		self.sample_distribution = np.array(sample_distribution)
		self.sample_sigmas = np.array(sample_sigmas)

		assert np.sum(self.sample_distribution) == 1
		assert np.any(self.sample_distribution < 0) == False
		assert len(self.sample_distribution) == len(self.sample_sigmas)

		self.mode = mode
		self.path = data_path
		with open(split_file, "rb") as f:
			self.split = pkl.load(f)

		self.data = self.split[mode]
		self.res = res
		self.num_sample_points = num_sample_points
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.pointcloud_samples = pointcloud_samples
		self.ext = ext
		self.suffix = suffix
		self.cache_suffix = cache_suffix

		# compute number of samples per sampling method
		self.num_samples = np.rint(self.sample_distribution * self.num_sample_points).astype(np.uint32)

	def __len__(self):
		return len(self.data)

	def get_loader(self, shuffle=True):
		return torch.utils.data.DataLoader(
			self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
			worker_init_fn=self.worker_init_fn)

	def worker_init_fn(self, worker_id):
		''' Worker init function to ensure true randomness.
		'''
		# base_seed = int(os.urandom(4).encode('hex'), 16)
		base_seed = int(codecs.encode(os.urandom(4), 'hex'), 16)
		np.random.seed(base_seed + worker_id)

	def load_sampling_points(self, file):
		raise NotImplementedError

	def load_voxel_input(self, file):
		raise NotImplementedError

	def __getitem__(self, idx):
		raise NotImplementedError
