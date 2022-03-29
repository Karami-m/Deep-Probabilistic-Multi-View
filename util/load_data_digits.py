
import numpy as np
import scipy.io as spio
import gzip
import tensorflow as tf

"""Simple wrap counter: grabs chunks of indices, repermuted after every pass"""
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

class mv_data_digits():
	def __init__(self, file_name, args, SEED):
		self.args = args
		dtype = {tf.float32: np.float32,
				 tf.float16: np.float16,
				 tf.float64: np.float64}[args.inp_dtype]
		self.n_labels = 10
		self.to_reshape = True if args.dataset != 'digits_fc' else False

		# load MNIST digits:
		shape_in = [-1, 28, 28, 1]
		shape_out = [-1, args.image_size, args.image_size, 1] if args.image_size > 0 else shape_in

		from keras.datasets import mnist
		(x_tr_, y_tr_), (x_te_, y_te_) = mnist.load_data()
		# idx_tu = np.s_[0:: 6] first grab tune then use for train np.delete(x_tr_, idx_tu, axis=0)
		self.x1_tr = self._pad_reshape(x_tr_[:50000], view=1, shape_in=shape_in, shape_out=shape_out)
		self.x1_tu = self._pad_reshape(x_tr_[50000:], view=1, shape_in=shape_in, shape_out=shape_out)
		self.x1_te = self._pad_reshape(x_te_, view=1, shape_in=shape_in, shape_out=shape_out)
		self.label_tr = y_tr_[:50000]
		self.label_tu = y_tr_[50000:]
		self.label_te = y_te_
		self.x1_tr, self.label_tr = self._reorder(self.x1_tr, self.label_tr, rescale=255., dtype=np.float32)
		self.x1_tu, self.label_tu = self._reorder(self.x1_tu, self.label_tu, rescale=255., dtype=np.float32)
		self.x1_te, self.label_te = self._reorder(self.x1_te, self.label_te, rescale=255., dtype=np.float32)

		# load USPS digits:
		data_files = {
			'train': 'zip.train.gz',
			'test': 'zip.test.gz'}
		shape_in = [-1, 16, 16, 1]
		shape_out = [-1, args.image_size, args.image_size, 1] if args.image_size > 0 else shape_in

		n_each_class = [np.where(self.label_tr == i)[0].size for i in range(self.n_labels)]
		x2_tr, _ = self._read_datafile(file_name + data_files['train'], n_each_class, portion=[0., 5./6.])
		n_each_class = [np.where(self.label_tu == i)[0].size for i in range(self.n_labels)]
		x2_tu, _ = self._read_datafile(file_name + data_files['train'], n_each_class, portion=[5./6., 1.])
		n_each_class = [np.where(self.label_te == i)[0].size for i in range(self.n_labels)]
		x2_te, _ = self._read_datafile(file_name + data_files['test'], n_each_class, portion=[0., 1.])
		self.x2_tr = self._pad_reshape(x2_tr, view=2, shape_in=shape_in, shape_out=shape_out)
		self.x2_tu = self._pad_reshape(x2_tu, view=2, shape_in=shape_in, shape_out=shape_out)
		self.x2_te = self._pad_reshape(x2_te, view=2, shape_in=shape_in, shape_out=shape_out)

		self.t_tr, self.d1_in = self.x1_tr.shape[0], self.x1_tr.shape[1:]
		self.d2_in = self.x2_tr.shape[1:]
		self.t_te = self.x1_te.shape[0]
		self.t_tu = self.x1_tu.shape[0]
		self.batch_size = args.batch_size

		self.sampler = wrapcounter(self.batch_size, self.t_tr, seed=SEED)

		if args.n_train == -1:
			args.n_train = {'mnist': 50000,
							'digits': 50000,
							'digits_fc': 50000}[args.dataset]
		args.n_tune = {'mnist': 10000,
						'digits': 10000,
						'digits_fc': 10000}[args.dataset]
		args.n_test = {'mnist': 10000,
						'digits': 10000,
						'digits_fc': 10000}[args.dataset]
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

	def _read_datafile(self, path, n_each_class, portion=[0. , 1.]):
		"""
		Read the proprietary USPS digits data file.
		Some references:
		https://github.com/haeusser/learning_by_association/blob/master/semisup/tools/usps.py
		https://www.kaggle.com/bistaumanga/usps-dataset
		"""
		labels, images = [], []
		with gzip.GzipFile(path) as f:
			for line in f:
				vals = line.strip().split()
				labels.append(float(vals[0]))
				images.append([float(val) for val in vals[1:]])
		labels = np.array(labels, dtype=np.int32)
		labels[labels == 10] = 0  # fix weird 0 labels
		images = np.array(images, dtype=np.float32)
		# images = np.array(images, dtype=np.float32).reshape(-1, 16, 16, 1)
		images = (images + 1) / 2

		n = labels.shape[0]
		n0, n1 = int(portion[0] * n), int(portion[1] * n)
		images = images[n0:n1]
		labels = labels[n0:n1]

		img_list = []
		lbl_list = []
		for i in range(self.n_labels):
			inds = np.where(labels == i)[0]
			img_ = images[inds]
			repeat = int(np.ceil((1.0 * n_each_class[i]) / len(inds)))
			img_ = np.tile(img_, (repeat, 1))[0:n_each_class[i]]
			# img_ = np.tile(img_, (repeat, 1, 1, 1))[0:n_each_class[i]]
			img_list += [img_]

			lbl_ = labels[inds]
			lbl_ = np.tile(lbl_, repeat)[0:n_each_class[i]]
			lbl_list += [lbl_]

		return np.concatenate(img_list, axis=0), np.concatenate(lbl_list, axis=0)

	def _reorder(self, images, labels, rescale=1, dtype=None):
		if dtype:
			images = images.astype(dtype)
		images = images/rescale
		img_list = []
		lbl_list = []
		for i in range(self.n_labels):
			inds = np.where(labels == i)[0]
			img_ = images[inds]
			img_list += [img_]
			lbl_ = labels[inds]
			lbl_list += [lbl_]
		return np.concatenate(img_list, axis=0), np.concatenate(lbl_list, axis=0)

	def _pad_reshape(self, x, view, shape_in, shape_out):
		if self.to_reshape == False:
			return x.reshape([-1, shape_in[1]*shape_in[2]])

		x = np.reshape(x, shape_in)
		if shape_in == shape_out:
			return x

		pad_up = (shape_out[1] - shape_in[1]) // 2
		pad_down = (shape_out[1] - shape_in[1]) - pad_up
		pad_left = (shape_out[2] - shape_in[2]) // 2
		pad_right = (shape_out[2] - shape_in[2]) - pad_left
		return np.lib.pad(x, ((0, 0), (pad_up, pad_down), (pad_left, pad_right), (0, 0)), 'minimum')
		# if view == 1:
		# 	return np.lib.pad(x, ((0, 0), (pad_up, pad_down), (pad_left, pad_right), (0, 0)), 'minimum')
		# elif view == 2:
		# 	return np.lib.pad(x, ((0, 0), (pad_up, pad_down), (pad_left, pad_right), (0, 0)), 'symmetric')

	def set_next_batch(self):
		self.batch_ind = self.sampler.next_inds()
		self.x1_ba = self.x1_tr[self.batch_ind]
		self.x2_ba = self.x2_tr[self.batch_ind]
		self.label_ba = self.label_tr[self.batch_ind]
		if hasattr(self, 'Phi_tr'):
			self.Phi_ba = self.Phi_tr[self.batch_ind]
		return self.batch_ind

	def train_iterator(self, batch_size = None):
		if batch_size:
			self.batch_size = batch_size
		batch_ind = self.sampler.next_inds()
		self.batch_ind = batch_ind
		x1_ba = self.x1_tr[batch_ind]
		x2_ba = self.x2_tr[batch_ind]
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
		self.x1_tu = x1[idx_tu]
		self.x1_tr = np.delete(x1, idx_tu, axis=0)

		self.x2_tu = x2[idx_tu]
		self.x2_tr = np.delete(x2, idx_tu, axis=0)

		self.label_tu = label[idx_tu]
		self.label_tr = np.delete(label, idx_tu, axis=0)

		self.args.n_train = self.t_tr
		self.args.n_tune = self.t_tu
		# Get number of training and validation iterations
		self.args.n_batches = int(np.ceil(self.t_tr / (self.batch_size * self.args.n_gpu)))
		self.n_batches = self.args.n_batches
		return


class mv_data_mnist():
	def __init__(self, file_name, args, SEED):
		self.args = args
		dtype = {tf.float32: np.float32,
				 tf.float16: np.float16,
				 tf.float64: np.float64}[args.inp_dtype]
		self.n_labels = 10
		self.to_reshape = True if args.dataset != 'mnist_fc' else False

		# load MNIST digits:
		shape_in = [-1, 28, 28, 1]
		shape_out = [-1, args.image_size, args.image_size, 1] if args.image_size > 0 else shape_in

		from keras.datasets import mnist
		(x_tr_, y_tr_), (x_te_, y_te_) = mnist.load_data()
		# idx_tu = np.s_[0:: 6] first grab tune then use for train np.delete(x_tr_, idx_tu, axis=0)
		self.x1_tr = self._pad_reshape(x_tr_[:50000], view=1, shape_in=shape_in, shape_out=shape_out)
		self.x1_tu = self._pad_reshape(x_tr_[50000:], view=1, shape_in=shape_in, shape_out=shape_out)
		self.x1_te = self._pad_reshape(x_te_, view=1, shape_in=shape_in, shape_out=shape_out)
		self.label_tr = y_tr_[:50000]
		self.label_tu = y_tr_[50000:]
		self.label_te = y_te_
		self.x1_tr, self.label_tr = self._reorder(self.x1_tr, self.label_tr, rescale=255., dtype=np.float32)
		self.x1_tu, self.label_tu = self._reorder(self.x1_tu, self.label_tu, rescale=255., dtype=np.float32)
		self.x1_te, self.label_te = self._reorder(self.x1_te, self.label_te, rescale=255., dtype=np.float32)

		self.t_tr, self.d1_in = self.x1_tr.shape[0], self.x1_tr.shape[1:]
		self.t_te = self.x1_te.shape[0]
		self.t_tu = self.x1_tu.shape[0]
		self.batch_size = args.batch_size

		self.sampler = wrapcounter(self.batch_size, self.t_tr, seed=SEED)

		if args.n_train == -1:
			args.n_train = {'mnist': 50000,
							'digits': 50000,
							'digits_fc': 50000}[args.dataset]
		args.n_tune = {'mnist': 10000,
						'digits': 10000,
						'digits_fc': 10000}[args.dataset]
		args.n_test = {'mnist': 10000,
						'digits': 10000,
						'digits_fc': 10000}[args.dataset]
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

	def _reorder(self, images, labels, rescale=1, dtype=None):
		if dtype:
			images = images.astype(dtype)
		images = images/rescale
		img_list = []
		lbl_list = []
		for i in range(self.n_labels):
			inds = np.where(labels == i)[0]
			img_ = images[inds]
			img_list += [img_]
			lbl_ = labels[inds]
			lbl_list += [lbl_]
		return np.concatenate(img_list, axis=0), np.concatenate(lbl_list, axis=0)

	def _pad_reshape(self, x, view, shape_in, shape_out):
		if self.to_reshape == False:
			return x.reshape([-1, shape_in[1]*shape_in[2]])

		x = np.reshape(x, shape_in)
		if shape_in == shape_out:
			return x

		pad_up = (shape_out[1] - shape_in[1]) // 2
		pad_down = (shape_out[1] - shape_in[1]) - pad_up
		pad_left = (shape_out[2] - shape_in[2]) // 2
		pad_right = (shape_out[2] - shape_in[2]) - pad_left
		return np.lib.pad(x, ((0, 0), (pad_up, pad_down), (pad_left, pad_right), (0, 0)), 'minimum')

	def set_next_batch(self):
		self.batch_ind = self.sampler.next_inds()
		self.x1_ba = self.x1_tr[self.batch_ind]
		self.label_ba = self.label_tr[self.batch_ind]
		if hasattr(self, 'Phi_tr'):
			self.Phi_ba = self.Phi_tr[self.batch_ind]
		return self.batch_ind

	def train_iterator(self, batch_size = None):
		if batch_size:
			self.batch_size = batch_size
		batch_ind = self.sampler.next_inds()
		self.batch_ind = batch_ind
		x1_ba = self.x1_tr[batch_ind]
		label_ba = self.label_tr[batch_ind]
		return x1_ba, batch_ind, label_ba

	def validation_set_fold(self, fold, n_folds):
		x1 = np.concatenate((self.x1_tr, self.x1_tu), axis=0)
		label = np.concatenate((self.label_tr, self.label_tu), axis=0)
		t_tu = (self.t_tr+self.t_tu)//n_folds
		t_tr = self.t_tr + self.t_tu - t_tu
		s_tu = t_tu * fold
		self.t_tr, self.t_tu = t_tr, t_tu

		idx_tu = np.s_[fold:: n_folds] # = np.s_[s_tu : s_tu+t_tu]
		self.x1_tu = x1[idx_tu]
		self.x1_tr = np.delete(x1, idx_tu, axis=0)

		self.label_tu = label[idx_tu]
		self.label_tr = np.delete(label, idx_tu, axis=0)

		self.args.n_train = self.t_tr
		self.args.n_tune = self.t_tu
		# Get number of training and validation iterations
		self.args.n_batches = int(np.ceil(self.t_tr / (self.batch_size * self.args.n_gpu)))
		self.n_batches = self.args.n_batches
		return

class mv_data_usps():
	def __init__(self, file_name, args, SEED):
		self.args = args
		dtype = {tf.float32: np.float32,
				 tf.float16: np.float16,
				 tf.float64: np.float64}[args.inp_dtype]
		self.n_labels = 10
		self.to_reshape = True if args.dataset != 'usps_fc' else False

		# load USPS digits:
		data_files = {
			'train': 'zip.train.gz',
			'test': 'zip.test.gz'}
		shape_in = [-1, 16, 16, 1]
		shape_out = [-1, args.image_size, args.image_size, 1] if args.image_size > 0 else shape_in

		x1_tr, self.label_tr = self._read_datafile(file_name + data_files['train'], portion=[0., 5. / 6.])
		x1_tu, self.label_tu = self._read_datafile(file_name + data_files['train'], portion=[5. / 6., 1.])
		x1_te, self.label_te = self._read_datafile(file_name + data_files['test'], portion=[0., 1.])
		self.x1_tr = self._pad_reshape(x1_tr, shape_in=shape_in, shape_out=shape_out)
		self.x1_tu = self._pad_reshape(x1_tu, shape_in=shape_in, shape_out=shape_out)
		self.x1_te = self._pad_reshape(x1_te, shape_in=shape_in, shape_out=shape_out)
		self.x1_tr, self.label_tr = self._reorder(self.x1_tr, self.label_tr, rescale=1., dtype=np.float32)
		self.x1_tu, self.label_tu = self._reorder(self.x1_tu, self.label_tu, rescale=1., dtype=np.float32)
		self.x1_te, self.label_te = self._reorder(self.x1_te, self.label_te, rescale=1., dtype=np.float32)


		self.t_tr, self.d1_in = self.x1_tr.shape[0], self.x1_tr.shape[1:]
		self.t_te = self.x1_te.shape[0]
		self.t_tu = self.x1_tu.shape[0]
		self.batch_size = args.batch_size

		self.sampler = wrapcounter(self.batch_size, self.t_tr, seed=SEED)

		if args.n_train == -1:
			args.n_train = {'mnist': 50000,
							'usps':6075,
							'digits': 50000,
							'digits_fc': 50000}[args.dataset]
		args.n_tune = {'mnist': 10000,
					   'usps': 1216,
					   'digits': 10000,
					   'digits_fc': 10000}[args.dataset]
		args.n_test = {'mnist': 10000,
					   'usps': 2007,
					   'digits': 10000,
					   'digits_fc': 10000}[args.dataset]
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

	def _read_datafile(self, path, portion=[0. , 1.]):
		"""
		Read the proprietary USPS digits data file.
		Some references:
		https://github.com/haeusser/learning_by_association/blob/master/semisup/tools/usps.py
		https://www.kaggle.com/bistaumanga/usps-dataset
		"""
		labels, images = [], []
		with gzip.GzipFile(path) as f:
			for line in f:
				vals = line.strip().split()
				labels.append(float(vals[0]))
				images.append([float(val) for val in vals[1:]])
		labels = np.array(labels, dtype=np.int32)
		labels[labels == 10] = 0  # fix weird 0 labels
		images = np.array(images, dtype=np.float32)
		# images = np.array(images, dtype=np.float32).reshape(-1, 16, 16, 1)
		images = (images + 1) / 2

		n = labels.shape[0]
		n0, n1 = int(portion[0] * n), int(portion[1] * n)
		images = images[n0:n1]
		labels = labels[n0:n1]
		return images, labels


	def _reorder(self, images, labels, rescale=1, dtype=None):
		if dtype:
			images = images.astype(dtype)
		images = images/rescale
		img_list = []
		lbl_list = []
		for i in range(self.n_labels):
			inds = np.where(labels == i)[0]
			img_ = images[inds]
			img_list += [img_]
			lbl_ = labels[inds]
			lbl_list += [lbl_]
		return np.concatenate(img_list, axis=0), np.concatenate(lbl_list, axis=0)

	def _pad_reshape(self, x, shape_in, shape_out):
		if self.to_reshape == False:
			return x.reshape([-1, shape_in[1]*shape_in[2]])

		x = np.reshape(x, shape_in)
		if shape_in == shape_out:
			return x

		pad_up = (shape_out[1] - shape_in[1]) // 2
		pad_down = (shape_out[1] - shape_in[1]) - pad_up
		pad_left = (shape_out[2] - shape_in[2]) // 2
		pad_right = (shape_out[2] - shape_in[2]) - pad_left
		return np.lib.pad(x, ((0, 0), (pad_up, pad_down), (pad_left, pad_right), (0, 0)), 'minimum')
		# if view == 1:
		# 	return np.lib.pad(x, ((0, 0), (pad_up, pad_down), (pad_left, pad_right), (0, 0)), 'minimum')
		# elif view == 2:
		# 	return np.lib.pad(x, ((0, 0), (pad_up, pad_down), (pad_left, pad_right), (0, 0)), 'symmetric')

	def set_next_batch(self):
		self.batch_ind = self.sampler.next_inds()
		self.x1_ba = self.x1_tr[self.batch_ind]
		self.label_ba = self.label_tr[self.batch_ind]
		if hasattr(self, 'Phi_tr'):
			self.Phi_ba = self.Phi_tr[self.batch_ind]
		return self.batch_ind

	def train_iterator(self, batch_size = None):
		if batch_size:
			self.batch_size = batch_size
		batch_ind = self.sampler.next_inds()
		self.batch_ind = batch_ind
		x1_ba = self.x1_tr[batch_ind]
		label_ba = self.label_tr[batch_ind]
		return x1_ba, batch_ind, label_ba

	def validation_set_fold(self, fold, n_folds):
		x1 = np.concatenate((self.x1_tr, self.x1_tu), axis=0)
		label = np.concatenate((self.label_tr, self.label_tu), axis=0)
		t_tu = (self.t_tr+self.t_tu)//n_folds
		t_tr = self.t_tr + self.t_tu - t_tu
		s_tu = t_tu * fold
		self.t_tr, self.t_tu = t_tr, t_tu

		idx_tu = np.s_[fold:: n_folds] # = np.s_[s_tu : s_tu+t_tu]
		self.x1_tu = x1[idx_tu]
		self.x1_tr = np.delete(x1, idx_tu, axis=0)

		self.label_tu = label[idx_tu]
		self.label_tr = np.delete(label, idx_tu, axis=0)

		self.args.n_train = self.t_tr
		self.args.n_tune = self.t_tu
		# Get number of training and validation iterations
		self.args.n_batches = int(np.ceil(self.t_tr / (self.batch_size * self.args.n_gpu)))
		self.n_batches = self.args.n_batches
		return


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
