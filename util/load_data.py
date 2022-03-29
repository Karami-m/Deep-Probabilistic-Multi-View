

"""Simple wrap counter: grabs chunks of indices, repermuted after every pass"""

import numpy as np
import scipy.io as spio
import tensorflow as tf
import gzip, os


class wrapcounter():

	def __init__(self, gap, length, shuffle=True, seed=None):
		self.gap = gap
		self.length = length
		self.order = np.arange(length)
		if shuffle:
			np.random.seed(seed=seed)
			np.random.shuffle(self.order)
		self.start = 0
		self.wraps = 0

	def next_inds(self, seed=None):
		start = self.start
		end = start + self.gap
		if end >  self.length:
			self.wraps += 1
			self.start = start = 0
			end = start + self.gap
			np.random.shuffle(self.order)
		self.start += self.gap
		return self.order[start:end]

class mv_data_mnist():
	def __init__(self, file_name, args, SEED):
		self.args = args
		dtype = {tf.float32: np.float32,
				 tf.float16: np.float16,
				 tf.float64: np.float64}[args.inp_dtype]

		MAT = spio.loadmat(file_name, struct_as_record=False, squeeze_me=True)
		self.x1_tr = MAT['X1'].astype(dtype)
		self.x1_tu = MAT['XV1'].astype(dtype)
		self.x1_te = MAT['XTe1'].astype(dtype)
		self.x2_tr = MAT['X2'].astype(dtype)
		self.x2_tu = MAT['XV2'].astype(dtype)
		self.x2_te = MAT['XTe2'].astype(dtype)
		self.label_tr = MAT['trainLabel']
		self.label_tu = MAT['tuneLabel']
		self.label_te = MAT['testLabel']

		self.t_tr, self.d1_in = self.x1_tr.shape[0], self.x1_tr.shape[1:]
		self.d2_in = self.x2_tr.shape[1:]
		self.t_te = self.x1_te.shape[0]
		self.t_tu = self.x1_tu.shape[0]
		self.batch_size = args.batch_size

		self.sampler = wrapcounter(self.batch_size, self.t_tr, seed=SEED)

		if args.n_train == -1:
			args.n_train = {'n-mnist': 50000}[args.dataset]
		args.n_tune = {'n-mnist': 10000}[args.dataset]
		args.n_test = {'n-mnist': 10000}[args.dataset]
		# Get number of training and validation iterations
		args.n_batches = int(np.ceil(args.n_train / (self.batch_size * args.n_gpu)))
		if args.debug_mode:
			args.n_batches = 5
		self.n_batches = args.n_batches
		# args.n_batches_ts = int(np.ceil(args.n_test / (self.batch_size * args.n_gpu)))
		# args.n_batches_tu = int(np.ceil(args.n_tune / (self.batch_size * args.n_gpu)))
		#
		# # Do a full validation run
		# assert args.n_test % args.local_batch_test == 0
		# assert args.n_valid % args.local_batch_valid == 0
		# full_test_its = args.n_test // args.local_batch_test
		# full_valid_its = args.n_valid // args.local_batch_valid

	def set_next_batch(self):
		self.batch_ind = self.sampler.next_inds()
		self.x1_ba = self.x1_tr[self.batch_ind, :]
		self.x2_ba = self.x2_tr[self.batch_ind, :]
		self.label_ba = self.label_tr[self.batch_ind]
		if hasattr(self, 'Phi_tr'):
			self.Phi_ba = self.Phi_tr[self.batch_ind, :]
		return self.batch_ind

	def train_iterator(self, batch_size = None):
		if batch_size:
			self.batch_size = batch_size
		batch_ind = self.sampler.next_inds()
		self.batch_ind = batch_ind
		x1_ba = self.x1_tr[batch_ind, :]
		x2_ba = self.x2_tr[batch_ind, :]
		label_ba = self.label_tr[batch_ind]
		return x1_ba, x2_ba, batch_ind, label_ba

	def validation_set_fold(self, fold, n_folds):
		x1 = np.concatenate((self.x1_tr, self.x1_tu), axis=0)
		x2 = np.concatenate((self.x2_tr, self.x2_tu), axis=0)
		label = np.concatenate((self.label_tr, self.label_tu), axis=0)
		t_tu = (self.t_tr+self.t_tu)//n_folds
		t_tr = self.t_tr + self.t_tu - t_tu
		s_tu = t_tu * fold
		self.t_tr, self.t_tu = t_tr, t_tu

		idx_tu = np.s_[fold:: n_folds] # = np.s_[s_tu : s_tu+t_tu]
		self.x1_tu = x1[idx_tu,:]
		self.x1_tr = np.delete(x1, idx_tu, axis=0)

		self.x2_tu = x2[idx_tu, :]
		self.x2_tr = np.delete(x2, idx_tu, axis=0)

		self.label_tu = label[idx_tu]
		self.label_tr = np.delete(label, idx_tu, axis=0)

		self.args.n_train = self.t_tr
		self.args.n_tune = self.t_tu
		# Get number of training and validation iterations
		self.args.n_batches = int(np.ceil(self.t_tr / (self.batch_size * self.args.n_gpu)))
		self.n_batches = self.args.n_batches
		return


class mv_data_mnist2():
	def __init__(self, file_name, args, SEED):
		self.args = args
		MAT = spio.loadmat(file_name, struct_as_record=False, squeeze_me=True)

		shape_in = [-1, 28, 28, 1]
		shape_out = [-1, args.image_size, args.image_size, 1]
		def pad(x, view):
			if shape_in == shape_in:
				return x

			pad_up = (shape_out[1] - 28) // 2
			pad_down = (shape_out[1] - 28) - pad_up
			pad_left = (shape_out[2] - 28) // 2
			pad_right = (shape_out[2] - 28) - pad_left
			if view == 1:
				return np.lib.pad(x, ((0, 0), (pad_up, pad_down), (pad_left, pad_right), (0,0)), 'minimum')
			elif view== 2:
				return np.lib.pad(x, ((0, 0), (pad_up, pad_down), (pad_left, pad_right), (0,0)), 'symmetric')

		self.x1_tr = pad(np.reshape(MAT['X1'], shape_in), view=1)
		self.x1_tu = pad(np.reshape(MAT['XV1'], shape_in), view=1)
		self.x1_te = pad(np.reshape(MAT['XTe1'], shape_in), view=1)
		self.x2_tr = pad(np.reshape(MAT['X2'], shape_in), view=2)
		self.x2_tu = pad(np.reshape(MAT['XV2'], shape_in), view=2)
		self.x2_te = pad(np.reshape(MAT['XTe2'], shape_in), view=2)
		self.label_tr = MAT['trainLabel']
		self.label_tu = MAT['tuneLabel']
		self.label_te = MAT['testLabel']

		self.x1_init = self.x1_tr[0:args.batch_size_init]
		self.x2_init = self.x2_tr[0:args.batch_size_init]
		self.label_init = self.label_tr[0:args.batch_size_init]

		self.t_tr, self.d1_in = self.x1_tr.shape[0], self.x1_tr.shape[1:]
		self.d2_in = self.x2_tr.shape[1:]
		self.t_te = self.x1_te.shape[0]
		self.t_tu = self.x1_tu.shape[0]
		self.batch_size = args.batch_size

		self.sampler = wrapcounter(self.batch_size, self.t_tr, seed=SEED)

		if args.n_train == -1:
			args.n_train = {'mnist': 50000}[args.dataset]
		args.n_tune = {'mnist': 10000}[args.dataset]
		args.n_test = {'mnist': 10000}[args.dataset]
		# Get number of training and validation iterations
		args.n_batches = int(np.ceil(args.n_train / (self.batch_size * args.n_gpu)))
		if args.debug_mode:
			args.n_batches = 5
		self.n_batches = args.n_batches
		# args.n_batches_ts = int(np.ceil(args.n_test / (self.batch_size * args.n_gpu)))
		# args.n_batches_tu = int(np.ceil(args.n_tune / (self.batch_size * args.n_gpu)))
		#
		# # Do a full validation run
		# assert args.n_test % args.local_batch_test == 0
		# assert args.n_valid % args.local_batch_valid == 0
		# full_test_its = args.n_test // args.local_batch_test
		# full_valid_its = args.n_valid // args.local_batch_valid

	def set_next_batch(self):
		self.batch_ind = self.sampler.next_inds()
		self.x1_ba = self.x1_tr[self.batch_ind, :]
		self.x2_ba = self.x2_tr[self.batch_ind, :]
		self.label_ba = self.label_tr[self.batch_ind]
		if hasattr(self, 'Phi_tr'):
			self.Phi_ba = self.Phi_tr[self.batch_ind, :]
		return self.batch_ind

	def train_iterator(self, batch_size = None):
		if batch_size:
			self.batch_size = batch_size
		batch_ind = self.sampler.next_inds()
		self.batch_ind = batch_ind
		x1_ba = self.x1_tr[batch_ind, :]
		x2_ba = self.x2_tr[batch_ind, :]
		label_ba = self.label_tr[batch_ind]
		return x1_ba, x2_ba, batch_ind, label_ba

	def validation_set_fold(self, fold, n_folds):
		x1 = np.concatenate((self.x1_tr, self.x1_tu), axis=0)
		x2 = np.concatenate((self.x2_tr, self.x2_tu), axis=0)
		label = np.concatenate((self.label_tr, self.label_tu), axis=0)
		t_tu = (self.t_tr+self.t_tu)//n_folds
		t_tr = self.t_tr + self.t_tu - t_tu
		s_tu = t_tu * fold
		self.t_tr, self.t_tu = t_tr, t_tu

		self.x1_init = self.x1_tr[0: self.args.batch_size_init]
		self.x2_init = self.x2_tr[0: self.args.batch_size_init]
		self.label_init = self.label_tr[0: self.args.batch_size_init]

		idx_tu = np.s_[fold:: n_folds] # = np.s_[s_tu : s_tu+t_tu]
		self.x1_tu = x1[idx_tu,:]
		self.x1_tr = np.delete(x1, idx_tu, axis=0)

		self.x2_tu = x2[idx_tu, :]
		self.x2_tr = np.delete(x2, idx_tu, axis=0)

		self.label_tu = label[idx_tu]
		self.label_tr = np.delete(label, idx_tu, axis=0)

		self.args.n_train = self.t_tr
		self.args.n_tune = self.t_tu
		# Get number of training and validation iterations
		self.args.n_batches = int(np.ceil(self.t_tr / (self.batch_size * self.args.n_gpu)))
		self.n_batches = self.args.n_batches
		return

from util.load_data_MultiModal import mv_data_flickr
class mv_data_flickr2(mv_data_flickr):
	def __init__(self, file_name, args, SEED):
		super(mv_data_flickr2, self).__init__(file_name, args, SEED)

		# self.x1_tr, self.x2_tr = self.x_tr
		# self.x1_tu, self.x2_tu = self.x_tu
		# self.x1_te, self.x2_te = self.x_te
		# self.x1_trlb, self.x2_trlb = self.x_trlb
		#
		# self.d1_in, self.d2_in = self.d_in

	@property
	def x1_tr(self):
		return self.x_tr[0]

	@property
	def x1_tu(self):
		return self.x_tu[0]

	@property
	def x1_te(self):
		return self.x_te[0]

	@property
	def x1_trlb(self):
		return self.x_trlb[0]

	@property
	def x2_tr(self):
		return self.x_tr[1]

	@property
	def x2_tu(self):
		return self.x_tu[1]

	@property
	def x2_te(self):
		return self.x_te[1]

	@property
	def x2_trlb(self):
		return self.x_trlb[1]

	@property
	def d1_in(self):
		return self.d_in[0]

	@property
	def d2_in(self):
		return self.d_in[1]




# Get number of training and validation iterations
def get_its(args):
	# These run for a fixed amount of time. As anchored batch is smaller, we've actually seen fewer examples
	train_its = int(np.ceil(args.n_train / (args.batch_size * args.n_gpu)))
	test_its = int(np.ceil(args.n_test / (args.batch_size * args.n_gpu)))
	train_epoch = train_its * args.batch_size * args.n_gpu
	print("Train epoch size: " + str(train_epoch))

	# Do a full validation run
	assert args.n_test % args.local_batch_test == 0
	assert args.n_valid % args.local_batch_valid == 0
	full_test_its = args.n_test // args.local_batch_test
	full_valid_its = args.n_valid // args.local_batch_valid

	return train_its, full_valid_its, test_its, full_test_its
