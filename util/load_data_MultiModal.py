
import numpy as np
import scipy.io as spio
import gzip, os, time
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


class mv_data(object):
	def __init__(self, file_name, args, SEED):
		self.args = args
		self.SEED = SEED
		dtype = {tf.float32: np.float32,
				 tf.float16: np.float16,
				 tf.float64: np.float64}[args.inp_dtype]


	def _set_params(self):
		# self.n_labels = self.label_tr.max()

		self.t_tr = self.x_tr[0].shape[0]
		self.d_in = [x_temp.shape[1:] for x_temp in self.x_tr]
		self.t_te = self.x_te[0].shape[0]
		self.t_tu = self.x_tu[0].shape[0]
		self.batch_size = self.args.batch_size

		self.sampler = wrapcounter(self.batch_size, self.t_tr, seed=self.SEED)

		if self.args.n_train == -1:
			self.args.n_train = {
				'yaleb':int(2424*self.alpha_),
				'mnist': 50000,
				'usps':6075,
				'digits': 50000,
				'digits_fc': 50000,
				'flickr': self.t_tr,
				'noisy_mnist': 50000,
			}[self.args.dataset]
		self.args.n_tune = {
			'yaleb':2424-int(2424*self.alpha_),
			'mnist': 10000,
			'usps': 1216,
			'digits': 10000,
			'digits_fc': 10000,
			'flickr': self.t_tu,
			'noisy_mnist': 10000,
		}[self.args.dataset]
		self.args.n_test = {
			'yaleb':2424,
			'mnist': 10000,
			'usps': 2007,
			'digits': 10000,
			'digits_fc': 10000,
			'flickr': self.t_te,
			'noisy_mnist': 10000,
		}[self.args.dataset]
		# Get number of training and validation iterations
		self.args.n_batches = int(np.ceil(self.args.n_train / (self.batch_size * self.args.n_gpu)))
		if self.args.debug_mode:
			self.args.n_batches = 5
		self.n_batches = self.args.n_batches

	def _pad_reshape(self, x, shape_in, shape_out, transpose=[0, 1, 2, 3]):
		if self.to_reshape == False:
			return x.reshape([-1, shape_in[1]*shape_in[2]])

		# x = np.reshape(x, shape_in)
		x = np.transpose(np.reshape(x, shape_in), transpose)
		if shape_in == shape_out:
			return x

		pad_up = (shape_out[1] - shape_in[1]) // 2
		pad_down = (shape_out[1] - shape_in[1]) - pad_up
		pad_left = (shape_out[2] - shape_in[2]) // 2
		pad_right = (shape_out[2] - shape_in[2]) - pad_left
		return np.lib.pad(x, ((0, 0), (pad_up, pad_down), (pad_left, pad_right), (0, 0)), 'minimum')

	def set_next_batch(self):
		self.batch_ind = self.sampler.next_inds()
		self.x_ba = []
		for i in range(self.n_modalities):
			self.x_ba.append(self.x_tr[i][self.batch_ind])
		self.label_ba = self.label_tr[self.batch_ind]
		if hasattr(self, 'Phi_tr'):
			self.Phi_ba = self.Phi_tr[self.batch_ind]
		return self.batch_ind

	def train_iterator(self, batch_size = None):
		if batch_size:
			self.batch_size = batch_size
		batch_ind = self.sampler.next_inds()
		self.batch_ind = batch_ind
		x_ba = []
		for i in range(self.n_modalities):
			x_ba.append(self.x_tr[i][batch_ind])
		label_ba = self.label_tr[batch_ind]
		return tuple(x_ba) + (batch_ind, label_ba)

	def validation_set_fold(self, fold, n_folds):
		label = np.concatenate((self.label_tr, self.label_tu), axis=0)
		# t_tu = (self.t_tr+self.t_tu)//n_folds
		# t_tr = self.t_tr + self.t_tu - t_tu
		# s_tu = t_tu * fold
		# self.t_tr, self.t_tu = t_tr, t_tu
		idx_tu = np.s_[fold:: n_folds]  # = np.s_[s_tu : s_tu+t_tu]

		for i in range(self.n_modalities):
			x_ = np.concatenate((self.x_tr[i], self.x_tu[i]), axis=0)
			self.x_tu[i] = x_[idx_tu]
			self.x_tr[i] = np.delete(x_, idx_tu, axis=0)

		self.label_tu = label[idx_tu]
		self.label_tr = np.delete(label, idx_tu, axis=0)
		self.t_tr, self.t_tu = self.label_tr.shape[0], self.label_tu.shape[0]

		self.args.n_train = self.t_tr
		self.args.n_tune = self.t_tu
		# Get number of training and validation iterations
		self.args.n_batches = int(np.ceil(self.t_tr / (self.batch_size * self.args.n_gpu)))
		self.n_batches = self.args.n_batches

		self.sampler = wrapcounter(self.batch_size, self.t_tr, seed=self.SEED)
		return


class mv_data_yaleb(mv_data):
	def __init__(self, file_name, args, SEED):
		super(mv_data_yaleb, self).__init__(file_name, args, SEED)

		self.to_reshape = True if args.dataset != 'yaleb_fc' else False

		# load yaleb images:
		shape_in = [-1, 32, 32, 1]
		shape_out = [-1, args.image_size, args.image_size, 1] if args.image_size > 0 else shape_in
		self.alpha_ = .908

		self.x_tr, self.x_tu, self.x_te, self.label_tr, self.label_tu, self.label_te = \
			self._read_datafile(file_name+'EYB_fc.mat', splits=[[0., self.alpha_], [self.alpha_, 1.], [0., 1.]],
								shape_in=shape_in, shape_out=shape_out, rescale=255., dtype=np.float32)

		self.n_labels = self.label_tr.max()
		self._set_params()

	def _read_datafile(self, path, shape_in, shape_out, splits=[[0., 1.], [0., 1.], [0., 1.]], rescale=1, dtype=None):
		"""
		Read the proprietary Extended Yale-B data file.
		Some references:
		"""
		data = spio.loadmat(path)
		self.n_modalities = data['num_modalities'][0][0]
		Labels = np.array(data['Label'][0])
		n = Labels.shape[0]

		perm_ = np.arange(n)
		np.random.shuffle(perm_); np.random.shuffle(perm_) #np.random.permutation(n)


		n0_tr, n1_tr = int(splits[0][0] * n), int(splits[0][1] * n)
		n0_tu, n1_tu = int(splits[1][0] * n), int(splits[1][1] * n)
		n0_te, n1_te = int(splits[2][0] * n), int(splits[2][1] * n)
		ind_tr = list(np.sort(perm_[n0_tr:n1_tr]))
		ind_tu = list(np.sort(perm_[n0_tu:n1_tu]))
		ind_te = list(np.sort(perm_[n0_te:n1_te]))
		X_tr, X_tu, X_te = [], [], []
		for i in range(self.n_modalities):
			x_ = np.array(data['modality_' + str(i)])
			x_ = x_.T
			x_ = self._pad_reshape(x_, shape_in=shape_in, shape_out=shape_out, transpose=[0, 2, 1, 3])
			if dtype:
				x_ = x_.astype(dtype)
			x_ = x_ / rescale

			X_tr.append(x_[ind_tr])
			X_tu.append(x_[ind_tu])
			X_te.append(x_[ind_te])

		y_tr = Labels[ind_tr]
		y_tu = Labels[ind_tu]
		y_te = Labels[ind_te]

		return X_tr, X_tu, X_te, y_tr, y_tu, y_te


class mv_data_digits(mv_data):
	def __init__(self, file_name, args, SEED):
		super(mv_data_digits, self).__init__(file_name, args, SEED)
		self.n_modalities = 2
		self.n_labels = 10
		self.to_reshape = True if args.dataset != 'digits_fc' else False
		self.alpha_ = 1. # not used in this class

		# load MNIST digits:
		shape_in = [-1, 28, 28, 1]
		shape_out = [-1, args.image_size, args.image_size, 1] if args.image_size > 0 else shape_in

		from keras.datasets import mnist
		(x_tr_, y_tr_), (x_te_, y_te_) = mnist.load_data()
		# idx_tu = np.s_[0:: 6] first grab tune then use for train np.delete(x_tr_, idx_tu, axis=0)
		_x1_tr = self._pad_reshape(x_tr_[:50000], shape_in=shape_in, shape_out=shape_out)
		_x1_tu = self._pad_reshape(x_tr_[50000:], shape_in=shape_in, shape_out=shape_out)
		_x1_te = self._pad_reshape(x_te_, shape_in=shape_in, shape_out=shape_out)
		self.label_tr = y_tr_[:50000]
		self.label_tu = y_tr_[50000:]
		self.label_te = y_te_
		_x1_tr, self.label_tr = self._reorder(_x1_tr, self.label_tr, rescale=255., dtype=np.float32)
		_x1_tu, self.label_tu = self._reorder(_x1_tu, self.label_tu, rescale=255., dtype=np.float32)
		_x1_te, self.label_te = self._reorder(_x1_te, self.label_te, rescale=255., dtype=np.float32)

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
		_x2_tr = self._pad_reshape(x2_tr, shape_in=shape_in, shape_out=shape_out)
		_x2_tu = self._pad_reshape(x2_tu, shape_in=shape_in, shape_out=shape_out)
		_x2_te = self._pad_reshape(x2_te, shape_in=shape_in, shape_out=shape_out)


		self.x_tr = [_x1_tr, _x2_tr]
		self.x_tu = [_x1_tu, _x2_tu]
		self.x_te = [_x1_te, _x2_te]

		self._set_params()

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


class mv_data_mnist(mv_data):
	def __init__(self, file_name, args, SEED):
		super(mv_data_mnist, self).__init__(file_name, args, SEED)
		self.n_modalities = 1
		self.n_labels = 10
		self.to_reshape = True if args.dataset != 'mnist_fc' else False
		self.alpha_ = 1. # not used in this class

		# load MNIST digits:
		shape_in = [-1, 28, 28, 1]
		shape_out = [-1, args.image_size, args.image_size, 1] if args.image_size > 0 else shape_in

		from keras.datasets import mnist
		(x_tr_, y_tr_), (x_te_, y_te_) = mnist.load_data()
		# idx_tu = np.s_[0:: 6] first grab tune then use for train np.delete(x_tr_, idx_tu, axis=0)
		_x1_tr = self._pad_reshape(x_tr_[:50000], shape_in=shape_in, shape_out=shape_out)
		_x1_tu = self._pad_reshape(x_tr_[50000:], shape_in=shape_in, shape_out=shape_out)
		_x1_te = self._pad_reshape(x_te_, shape_in=shape_in, shape_out=shape_out)
		self.label_tr = y_tr_[:50000]
		self.label_tu = y_tr_[50000:]
		self.label_te = y_te_
		_x1_tr, self.label_tr = self._reorder(_x1_tr, self.label_tr, rescale=255., dtype=np.float32)
		_x1_tu, self.label_tu = self._reorder(_x1_tu, self.label_tu, rescale=255., dtype=np.float32)
		_x1_te, self.label_te = self._reorder(_x1_te, self.label_te, rescale=255., dtype=np.float32)

		self.x_tr = [_x1_tr]
		self.x_tu = [_x1_tu]
		self.x_te = [_x1_te]

		self._set_params()

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


class mv_data_usps(mv_data):
	def __init__(self, file_name, args, SEED):
		super(mv_data_usps, self).__init__(file_name, args, SEED)
		self.n_modalities = 1
		self.n_labels = 10
		self.to_reshape = True if args.dataset != 'usps_fc' else False
		self.alpha_ = 1. # not used in this class

		# load USPS digits:
		data_files = {
			'train': 'zip.train.gz',
			'test': 'zip.test.gz'}
		shape_in = [-1, 16, 16, 1]
		shape_out = [-1, args.image_size, args.image_size, 1] if args.image_size > 0 else shape_in

		x1_tr, self.label_tr = self._read_datafile(file_name + data_files['train'], portion=[0., 5. / 6.])
		x1_tu, self.label_tu = self._read_datafile(file_name + data_files['train'], portion=[5. / 6., 1.])
		x1_te, self.label_te = self._read_datafile(file_name + data_files['test'], portion=[0., 1.])
		_x1_tr = self._pad_reshape(x1_tr, shape_in=shape_in, shape_out=shape_out)
		_x1_tu = self._pad_reshape(x1_tu, shape_in=shape_in, shape_out=shape_out)
		_x1_te = self._pad_reshape(x1_te, shape_in=shape_in, shape_out=shape_out)
		_x1_tr, self.label_tr = self._reorder(_x1_tr, self.label_tr, rescale=1., dtype=np.float32)
		_x1_tu, self.label_tu = self._reorder(_x1_tu, self.label_tu, rescale=1., dtype=np.float32)
		_x1_te, self.label_te = self._reorder(_x1_te, self.label_te, rescale=1., dtype=np.float32)

		self.x_tr = [_x1_tr]
		self.x_tu = [_x1_tu]
		self.x_te = [_x1_te]
		self._set_params()

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


class mv_data_noisy_mnist(mv_data):
	def __init__(self, file_name, args, SEED):
		super(mv_data_noisy_mnist, self).__init__(file_name, args, SEED)
		self.n_modalities = 2
		self.n_labels = 10
		self.to_reshape = False # True if args.dataset != 'digits_fc' else False
		self.alpha_ = 1. # not used in this class
		# load MNIST digits:
		shape_in = [-1, 784]
		shape_out = [-1, 784]
		dtype = {tf.float32: np.float32,
				 tf.float16: np.float16,
				 tf.float64: np.float64}[args.inp_dtype]

		MAT = spio.loadmat(file_name, struct_as_record=False, squeeze_me=True)
		_x1_tr = MAT['X1'].astype(dtype)
		_x1_tu = MAT['XV1'].astype(dtype)
		_x1_te = MAT['XTe1'].astype(dtype)
		_x2_tr = MAT['X2'].astype(dtype)
		_x2_tu = MAT['XV2'].astype(dtype)
		_x2_te = MAT['XTe2'].astype(dtype)
		self.label_tr = MAT['trainLabel']
		self.label_tu = MAT['tuneLabel']
		self.label_te = MAT['testLabel']

		self.x_tr = [_x1_tr, _x2_tr]
		self.x_tu = [_x1_tu, _x2_tu]
		self.x_te = [_x1_te, _x2_te]

		self._set_params()


import scipy.sparse as sp
def LoadSparse(inputfile, verbose=False):
	"""
	Loads a sparse matrix stored as npz file.
	from: http://www.cs.toronto.edu/~nitish/multimodal/
	"""
	npzfile = np.load(inputfile)
	mat = sp.csr_matrix((npzfile['data'], npzfile['indices'],
						 npzfile['indptr']),
						shape=tuple(list(npzfile['shape'])))
	if verbose:
		print('Loaded sparse matrix from %s of shape %s' % (inputfile,
														  mat.shape.__str__()))
	return mat

class mv_data_flickr(mv_data):
	def __init__(self, file_name, args, SEED):
		super(mv_data_flickr, self).__init__(file_name, args, SEED)

		self.n_modalities = 2
		min_tag_counts = 2
		self.flickr_img_unlabeled_path 	= os.path.join(file_name, 'image/unlabelled/')
		self.flickr_img_labeled_path 	= os.path.join(file_name, 'image/labelled/')
		self.flickr_txt_path 			= os.path.join(file_name, 'text/')
		self.validation_split_path 		= os.path.join(file_name, 'splits/')
		self.alpha_ = 1. # not used in this class

		###############
		# load unlabelled images #
		###############
		files_count = 1
		path = self.flickr_img_unlabeled_path
		flickr_img = np.load(os.path.join(path, 'combined-00003_1-of-00100.npy'))
		for fract in range(4, 101):
			f = 'combined-%05d-of-00100.npy' % fract
			if self.args.debug_mode and (files_count == 2):  # only 2 files for debug
				break
			files_count += 1
			_img = np.load(os.path.join(path, f))
			flickr_img = np.concatenate([flickr_img, _img], axis=0)

		###############
		# load unlabelled text   #
		###############
		path = self.flickr_txt_path
		flickr_txt = LoadSparse(os.path.join(path, 'text_all_2000_unlabelled.npz'))
		flickr_txt = (flickr_txt > 0.).astype('float32') # change all tags to 0. or 1.

		###############
		# load labelled images #
		###############
		path = self.flickr_img_labeled_path
		flickr_img_lbl = np.load(os.path.join(path, 'combined-00001-of-00100.npy'))
		_img       = np.load(os.path.join(path, 'combined-00002-of-00100.npy'))
		_img3      = np.load(os.path.join(path, 'combined-00003_0-of-00100.npy'))
		flickr_img_lbl = np.concatenate([flickr_img_lbl, _img, _img3], axis=0)

		###############
		# load labelled text   #
		###############
		path = self.flickr_txt_path
		flickr_txt_lbl = LoadSparse(os.path.join(path, 'text_all_2000_labelled.npz'))
		flickr_txt_lbl = flickr_txt_lbl.toarray()
		flickr_txt_lbl = (flickr_txt_lbl > 0.).astype('float32')


		labels_ = np.load(os.path.join(file_name, 'labels.npy'))
		self.n_labels = 38

		#####################
		# standardize the data
		#####################
		moments_file_ = 'mean_std_img2'
		moments_file_path = os.path.join(file_name, moments_file_ + '.npz')
		if os.path.isfile(moments_file_path):
			npzfile = np.load(moments_file_path)
			[img_mean, img_std] = [npzfile['img_mean'], npzfile['img_std']]
		else:
			t0_ = time.time()
			img_mean, img_std = np.mean(flickr_img, axis=0), np.std(flickr_img, axis=0)
			np.savez(moments_file_path, img_mean=img_mean, img_std=img_std)
			print("moment file is saved within %g seconds" %(time.time()-t0_))
		# img_std += 1e-6
		img_var_ = np.square(img_std)
		img_std = np.sqrt(img_var_ + min(img_var_)*1e-3) # it is used in Sohn's Matlab code
		flickr_img -= img_mean
		flickr_img /= img_std
		flickr_img_lbl -= img_mean
		flickr_img_lbl /= img_std

		#######################
		# remove samples with small tags
		#######################
		idx = np.where(np.sum(flickr_txt, axis=1) >= min_tag_counts)[0]
		_t_tr = flickr_img.shape[0]
		idx = idx[np.where(idx < _t_tr)[0]]
		flickr_img = flickr_img[idx]
		flickr_txt = flickr_txt[idx]

		pos_average_ = np.mean(np.sum(flickr_txt, axis=1))
		self.tag_positive_ratio = pos_average_/(flickr_txt.shape[1]-pos_average_)
		# it is SIMILAR to:
		# pos_average_ = np.mean(np.sum(flickr_txt, axis=0))
		# self.tag_positive_ratio = pos_average_ / (flickr_txt.shape[0] - pos_average_)

		self.x_tr = [flickr_img, flickr_txt]
		del([flickr_img])
		self.x_trlb = [flickr_img_lbl[0:10000],     flickr_txt_lbl[0:10000]]
		self.x_tu   = [flickr_img_lbl[10000:15000], flickr_txt_lbl[10000:15000]]
		self.x_te   = [flickr_img_lbl[15000:25000], flickr_txt_lbl[15000:25000]]
		self.label_tr   = np.array([None] * flickr_txt.shape[0])
		self.label_trlb = labels_[0:10000]
		self.label_tu   = labels_[10000:15000]
		self.label_te   = labels_[15000:25000]

		self._set_params()
		self.t_trlb = self.args.n_trainlb = self.label_trlb.shape[0]

	def train_iterator(self, batch_size = None):
		if batch_size:
			self.batch_size = batch_size
		batch_ind = self.sampler.next_inds()
		self.batch_ind = batch_ind
		x_ba = []
		for i in range(self.n_modalities):
			x_ba_i_ = self.x_tr[i][batch_ind]
			x_ba.append(x_ba_i_.toarray() if sp.issparse(x_ba_i_) else x_ba_i_)
		label_ba = self.label_tr[batch_ind]
		return tuple(x_ba) + (batch_ind, label_ba)

	def validation_set_fold(self, fold, n_folds):
		if n_folds > 5:
			raise ValueError('excedded the max number of folds that is 5')

		labels = np.concatenate((self.label_trlb, self.label_tu, self.label_te), axis=0)

		fld = fold + 1 # fold is in {0,1,2,3,4} so incremented by 1
		idx_tr = np.load(os.path.join(self.validation_split_path, 'train_indices_%1d.npy'%fld))
		idx_tu = np.load(os.path.join(self.validation_split_path, 'valid_indices_%1d.npy'%fld))
		idx_te = np.load(os.path.join(self.validation_split_path, 'test_indices_%1d.npy'%fld))

		x_ = np.concatenate((self.x_trlb[0], self.x_tu[0], self.x_te[0]), axis=0)
		self.x_trlb[0], self.x_tu[0], self.x_te[0] = x_[idx_tr], x_[idx_tu], x_[idx_te]
		x_ = np.concatenate((self.x_trlb[1], self.x_tu[1], self.x_te[1]), axis=0)
		# x_ = sp.vstack((self.x_trlb[1], self.x_tu[1], self.x_te[1]))
		# x_ = x_.toarray()
		self.x_trlb[1], self.x_tu[1], self.x_te[1] = x_[idx_tr], x_[idx_tu], x_[idx_te]

		self.label_trlb = labels[idx_tr]
		self.label_tu = labels[idx_tu]
		self.label_te = labels[idx_te]
		self.t_trlb, self.t_tu, self.t_te = self.label_trlb.shape[0], self.label_tu.shape[0], self.label_te.shape[0]

		self.args.n_trainlb = self.t_trlb
		self.args.n_tune = self.t_tu
		self.args.n_test = self.t_te

		# Get number of training and validation iterations
		self.args.n_batches = int(np.ceil(self.t_tr / (self.batch_size * self.args.n_gpu)))
		self.n_batches = self.args.n_batches
		self.sampler = wrapcounter(self.batch_size, self.t_tr, seed=self.SEED)
		return

	def permute_folds(self, inp_tr, inp_tu, inp_te, fold, n_folds):
		if n_folds > 5:
			raise ValueError('excedded the max number of folds that is 5')

		labels = np.concatenate((self.label_trlb, self.label_tu, self.label_te), axis=0)

		fld = fold + 1 # fold is in {0,1,2,3,4} so incremented by 1
		idx_tr = np.load(os.path.join(self.validation_split_path, 'train_indices_%1d.npy'%fld))
		idx_tu = np.load(os.path.join(self.validation_split_path, 'valid_indices_%1d.npy'%fld))
		idx_te = np.load(os.path.join(self.validation_split_path, 'test_indices_%1d.npy'%fld))

		out_tr, out_tu, out_te = [], [], []
		for x_tr_, x_tu_, x_te_ in zip(inp_tr, inp_tu, inp_te):
			x_ = np.concatenate((x_tr_, x_tu_, x_te_), axis=0)
			out_tr += [x_[idx_tr]]
			out_tu += [x_[idx_tu]]
			out_te += [x_[idx_te]]

		return out_tr, out_tu, out_te


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
