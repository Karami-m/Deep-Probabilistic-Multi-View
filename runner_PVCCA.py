from __future__ import division, print_function

# # TODO:


import sys, os, time, copy
from datetime import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import util
from util.misc import myprint_file, struct
from util.misc import HParams_new as HParams
from util.downstream_tasks import linear_svm_classify, linear_svm_classify2, spectral_clustering, spectral_clustering2, spectral_clustering3
from util.downstream_tasks import orth_measure, orth_measure2, orth_measure3
from util.graphics import plot_tile_image


try:
	from urllib.request import urlretrieve  # Python 3
except ImportError:
	from urllib import urlretrieve  # Python 2
homepath = os.getcwd() #+ "/.."
sys.path.append(homepath)
sys.path.append(homepath+'/util')



# Hyper-parameters default
def get_default_arch(args):
	"""Get the default hyperparameters for conv flow."""
	return HParams(
		hidden_dim=args.hidden_dim,  # hidden_dim1=60,hidden_dim2=60,
		hidden_dim_shrd = args.hidden_dim_shrd,
		var_v1=1., var_v2=1.,
		reg_var_wgt=1e-4,  # Weight decay parameter
		p_drop=0.,  # dropout probability in NN (keep_prob= 1. - p_drop)
		reg_bias=1,  # to regularize the bias with the same weight or not
		gate='relu',
		# Multi-view hyperparameters
		var_prior0=1., # variance of prior Guassina on Phi
		var_prior1=1., var_prior2_ratio=1., # variance of prior Guassina on eps1, eps2
		postenc1='sp', postenc2='sp',
		postdec2='sp', #{'exp', 'sp'}
		shrd_on_all='none', # 'none', 'early', 'late'
		shrd_est='z1', # 'nn', 'z1', 'z12', 'z2'
		predec='convT', #'fc' or 'convT'
		clstr_mode='advanced', #'simple' only used for mnist, usps and digits
		# n_basis=args.hidden_dim_shrd,
	)

class Runner(object):
	
	def __init__(self, args):

		self.args = args
		args.dtype = tf.float32

		# Initialization
		tf.reset_default_graph()

		args.ckpt_best_filename = "model_best_loss"
		args.ckpt_filename = "model_last.ckpt"
		args.data_dir = homepath + "/data/"
		# sys.path.append(data_dir)
		
		logdir_name = args.logdir
		if args.mode == "cross-validate":
			time.sleep(47 * args.fold)
			args.logdir = homepath + "/logs/{}/CV/".format(args.dataset) + logdir_name
			args.logdir += '/'
			args.log_file = args.logdir + 'resultsLogFold%d.txt' % args.fold
			self.npy_result_file = 'fold%d_output.npy' % args.fold
			self.npy_result_file_running = self.npy_result_file
		else:
			args.logdir = homepath + "/logs/{}/".format(args.dataset) + logdir_name \
				if (args.logdir[0] != '/') and (args.logdir[0:2] != './') else logdir_name
			args.logdir += '/'
			args.log_file = args.logdir + \
								('results_log.txt' if args.mode == "train" else 'test_log.txt')
			self.npy_result_file = 'final_output.npy'
			self.npy_result_file_running = 'running_output.npy'
		
		if not os.path.exists(args.logdir): os.makedirs(args.logdir)  # , exist_ok=True)
		
		hp = get_default_arch(self.args).update_config(args.hpconfig)
		args.hp = hp
		args._print = self._print = lambda str_in, verbose: myprint_file(args.log_file, verbose, str_in)
		self._print("\n\n ------------------------------------------------- \n\n", verbose='log')
		self._print("Results will be saved in sub_directory: %s \n" % logdir_name, verbose='log')
		self._print("\n    	MODEL ARGUMENTS ARE:", verbose='log')
		self._print(args.__dict__, verbose='log')
		self._print("\n 		NN CONFIG HYPER-PARAMETERS ARE:", verbose='log')
		self._print(args.hp.__dict__['dict_'], verbose='log')
		for key in hp.dict_.keys():
			args.__setattr__(key, hp[key])
		
		args.SEED_gra = args.seed if args.seed >= 0 else np.random.randint(1, 100000)  # None for random seed
		args.SEED_op = args.SEED_gra
		tf.set_random_seed(args.SEED_gra)
		np.random.seed(args.SEED_gra)

		self.data = self.load_data()

		args.eval_methods = args.eval_methods.split('_')
		args.eval_mtd_init ={
			'class': 1.1,
			'NMI': 0., 'ACC': 0., 'ARI': 0.,
			'NMI0': 0., 'ACC0': 0., 'ARI0': 0., 'NMI00': 0., 'ACC00': 0., 'ARI00': 0.,
			'NMI2': 0., 'ACC2': 0., 'ARI2': 0., 'NMI3': 0., 'ACC3': 0., 'ARI3': 0.,
			'classv1': 1.1, 'NMIv1': 0., 'ACCv1': 0., 'classCCA': 1.1, 'classEYB':1.1,
			'nllrec': np.inf}
		args.eval_mtd_init = {key: args.eval_mtd_init[key] for key in args.eval_methods}
		
		args.evalCCA = True if 'classCCA' in args.eval_methods else False
		
		if args.mode == "cross-validate":
			self.data.validation_set_fold(args.fold, args.n_folds)
			self._print("FOLD number=%d" % args.fold, verbose='log')

		self.load_model()

	def load_data(self):
		#########################
		# initialize data loaders for train/test splits
		#########################
		hp = self.args.hp
		self.is_nmnist = False
				
		if self.args.dataset == 'n-mnist':
			self.is_nmnist = True
			from util.load_data import mv_data_mnist as mv_data
			# args.score1, args.score2 = 'NMI', 'ACC'
			self.args.inp_dtype = tf.float32
			data = mv_data(self.args.data_dir + 'MNIST.mat', self.args, self.args.SEED_gra)
			self.args.modalities = 2

			self.args.ae1, self.args.ae2 = struct(), struct()
			self.args.ae1.shape_enc = np.array([784, 1024, 1024, 1024, hp.hidden_dim])
			self.args.ae1.gate_enc = [hp.gate, hp.gate, hp.gate, 'linear']
			self.args.ae1.shape_dec = np.array([hp.hidden_dim, 1024, 1024, 1024, 784])
			self.args.ae1.gate_dec = [hp.gate, hp.gate, hp.gate, 'linear']
			self.args.ae1.postenc_gate = ['linear', hp.postenc1]
			self.args.ae1.postdec_gate = ['sigmoid', None]
			self.args.ae1.loss_rec = 'CE'
			self.args.ae1.var_rec, self.args.ae1.var_prior = hp.var_v1, hp.var_prior1

			self.args.ae2.shape_enc, self.args.ae2.gate_enc = self.args.ae1.shape_enc, self.args.ae1.gate_enc
			self.args.ae2.shape_dec = self.args.ae1.shape_dec
			self.args.ae2.gate_dec = [hp.gate, hp.gate, hp.gate, 'linear']
			self.args.ae2.postenc_gate = ['linear', hp.postenc2]
			self.args.ae2.postdec_gate = ['sigmoid', hp.postdec2]
			self.args.ae2.loss_rec = 'WLS'
			self.args.ae2.var_rec, self.args.ae2.var_prior = hp.var_v2, hp.var_prior1 * hp.var_prior2_ratio

			self.args.ae_shrd = struct()
			self.args.ae_shrd.shape_enc = np.array([784 if not self.args.hp.shrd_on_all else 2 * 784,
												   1024, 1024, 1024, self.args.hidden_dim_shrd])
			self.args.ae_shrd.gate_enc = [hp.gate, hp.gate, hp.gate, 'linear']  # 'sigmoid']
			self.args.ae_shrd.var_prior = hp.var_prior0
			self.args.ae_shrd.loss_rec, self.args.ae_shrd.var_rec, = None, None
			self.args.ae_shrd.postenc_gate = ['linear', 'sigmoid'] if hp.shrd_est == 'nn' else [None, 'sigmoid']
			self.args.ae_shrd.postdec_gate = [None, None]
			self.args.ae_shrd.shape_dec = [-1]
			self.args.ae_shrd.gate_dec = [None]

			self.args.inp_shape1 = [None] + list(data.d1_in)
			self.args.inp_shape2 = [None] + list(data.d2_in)

		elif self.args.dataset == 'yaleb':
			from util.load_data_MultiModal import mv_data_yaleb as mv_data
			self.args.modalities = 5
			self.args.inp_dtype = tf.float32
			data = mv_data(self.args.data_dir, self.args, self.args.SEED_gra)
			self.args.inp_shapes = [[None] + list(d_in_) for d_in_ in data.d_in]

			data.cluster = struct()
			data.cluster.k_knn = [5]  # [5, 10, 20, 30]
			data.cluster.assign_labels = 'discretize'  # 'kmeans'
			data.cluster.affinity = 'pre_nearest_neighbors'
			self.args.cluster = data.cluster

			self.args.ae_list = []
			# ---------- 1st view -------------
			ae_ = struct()
			ae_.enc_spec = [
				{'gate': hp.gate, 'dim_out': 10, 'filter_size': [5, 5], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 20, 'filter_size': [3, 3], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 30, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 30, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
			]
			# the order othe decoder layer is reveresed, last spec is first one the decoder network
			ae_.dec_spec = [
				{'gate': 'linear', 'dim_out': 1, 'filter_size': [5, 5], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 10, 'filter_size': [3, 3], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 20, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 30, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
			]
			ae_.postenc_gate = ['linear', hp.postenc1]
			ae_.postenc_dims = [hp.hidden_dim, hp.hidden_dim]
			ae_.postdec_gate = ['sigmoid', None]
			ae_.loss_rec = 'CE'
			ae_.var_rec, ae_.var_prior = hp.var_v1, hp.var_prior1

			self.args.ae_list += [ae_]

			# ---------- 2nd view -------------
			ae_.var_prior = hp.var_prior1 * hp.var_prior2_ratio  # todo: to fix, using deEp copy and on all views
			self.args.ae_list += [copy.deepcopy(ae_)]

			# ---------- 3rd view -------------
			self.args.ae_list += [copy.deepcopy(ae_)]

			# ---------- 4rd view -------------
			self.args.ae_list += [copy.deepcopy(ae_)]

			# ---------- 5rd view -------------
			self.args.ae_list += [copy.deepcopy(ae_)]

			# ---------- shared factor --------
			self.args.ae_shrd = struct()
			self.args.ae_shrd.enc_spec = self.args.ae_list[0].enc_spec
			if self.args.hp.shrd_on_all == 'late':
				self.args.ae_shrd.enc_spec = [
					{'gate': hp.gate, 'dim_out': 5 * hp.hidden_dim_shrd, 'filter_size': [1, 1], 'stride': [1, 1],
					 'pad': 'SAME'},
					{'gate': hp.gate, 'dim_out': 5 * hp.hidden_dim_shrd, 'filter_size': [1, 1], 'stride': [1, 1],
					 'pad': 'SAME'},
					{'gate': hp.gate, 'dim_out': 5 * hp.hidden_dim_shrd, 'filter_size': [1, 1], 'stride': [1, 1],
					 'pad': 'SAME'},
				]
			self.args.ae_shrd.dec_spec = self.args.ae_shrd.enc_spec  # dummy, it is not used
			self.args.ae_shrd.postenc_gate = ['linear', 'sigmoid'] if hp.shrd_est == 'nn' else [None, 'sigmoid']
			self.args.ae_shrd.postenc_dims = [hp.hidden_dim_shrd, hp.hidden_dim_shrd] if hp.shrd_est == 'nn' \
				else [0, hp.hidden_dim_shrd]
			self.args.ae_shrd.var_prior = hp.var_prior0
			self.args.ae_shrd.loss_rec, self.args.ae_shrd.var_rec, = None, None
			self.args.ae_shrd.postdec_gate = [None, None]

			if self.args.reconst in ['all2one', '345to12']:
				for jj in self.args.reconst_views[0]:
					self.args.ae_list[jj].enc_spec = [
						{'gate': hp.gate, 'dim_out': 5 * hp.hidden_dim, 'filter_size': [1, 1], 'stride': [1, 1],
						 'pad': 'SAME'},
						{'gate': hp.gate, 'dim_out': 5 * hp.hidden_dim, 'filter_size': [1, 1], 'stride': [1, 1],
						 'pad': 'SAME'},
					]

		elif self.args.dataset == 'digits':
			from util.load_data_MultiModal import mv_data_digits as mv_data
			self.args.modalities = 2
			self.args.inp_dtype = tf.float32
			data = mv_data(self.args.data_dir, self.args, self.args.SEED_gra)
			self.args.modalities = 2
			self.args.inp_shapes = [[None] + list(d_in_) for d_in_ in data.d_in]

			data.cluster = struct()
			if hp.clstr_mode == 'simple':
				data.cluster.k_knn = [5]  # [5, 10, 20, 30]
				data.cluster.assign_labels = 'discretize'  # 'kmeans'
			else:
				data.cluster.k_knn = [5, 10, 20, 30]  # [5] #
				data.cluster.assign_labels = 'kmeans'  # 'discretize' #
			data.cluster.affinity = 'pre_nearest_neighbors'
			self.args.cluster = data.cluster

			self.args.ae_list = []
			# ---------- 1st view -------------
			ae_ = struct()
			ae_.enc_spec = [
				{'gate': hp.gate, 'dim_out': 7, 'filter_size': [7, 7], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 10, 'filter_size': [5, 5], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 30, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 30, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
			]
			# the order othe decoder layer is reveresed, last spec is first one the decoder network
			ae_.dec_spec = [
				{'gate': 'linear', 'dim_out': 1, 'filter_size': [7, 7], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 7, 'filter_size': [5, 5], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 10, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 30, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
			]
			ae_.postenc_gate = ['linear', hp.postenc1]
			ae_.postenc_dims = [hp.hidden_dim, hp.hidden_dim]
			ae_.postdec_gate = ['sigmoid', None]
			ae_.loss_rec = 'CE'
			ae_.var_rec, ae_.var_prior = hp.var_v1, hp.var_prior1

			self.args.ae_list += [ae_]

			# ---------- 2nd view -------------
			ae_ = struct()
			ae_.enc_spec = [
				{'gate': hp.gate, 'dim_out': 7, 'filter_size': [7, 7], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 10, 'filter_size': [5, 5], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 15, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 15, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
			]
			ae_.dec_spec = [
				{'gate': 'linear', 'dim_out': 1, 'filter_size': [7, 7], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 7, 'filter_size': [5, 5], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 10, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 15, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
			]
			ae_.postenc_gate = ['linear', hp.postenc2]
			ae_.postenc_dims = [hp.hidden_dim, hp.hidden_dim]
			ae_.postdec_gate = ['sigmoid', None]
			ae_.loss_rec = 'CE'
			ae_.var_rec, ae_.var_prior = hp.var_v2, hp.var_prior1 * hp.var_prior2_ratio

			self.args.ae_list += [ae_]

			# ---------- shared factor --------
			self.args.ae_shrd = struct()
			self.args.ae_shrd.enc_spec = [
				{'gate': hp.gate, 'dim_out': 7, 'filter_size': [7, 7], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 10, 'filter_size': [5, 5], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 30, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 30, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
			]
			if self.args.hp.shrd_on_all == 'late':
				self.args.ae_shrd.enc_spec = [
					{'gate': hp.gate, 'dim_out': 5 * hp.hidden_dim_shrd, 'filter_size': [1, 1], 'stride': [1, 1],
					 'pad': 'SAME'},
					{'gate': hp.gate, 'dim_out': 5 * hp.hidden_dim_shrd, 'filter_size': [1, 1], 'stride': [1, 1],
					 'pad': 'SAME'},
					{'gate': hp.gate, 'dim_out': 5 * hp.hidden_dim_shrd, 'filter_size': [1, 1], 'stride': [1, 1],
					 'pad': 'SAME'},
				]
			self.args.ae_shrd.dec_spec = self.args.ae_shrd.enc_spec  # dummy, it is not used
			self.args.ae_shrd.postenc_gate = ['linear', 'sigmoid'] if hp.shrd_est == 'nn' else [None, 'sigmoid']
			self.args.ae_shrd.postenc_dims = [hp.hidden_dim_shrd, hp.hidden_dim_shrd] if hp.shrd_est == 'nn' \
				else [0, hp.hidden_dim_shrd]
			self.args.ae_shrd.var_prior = hp.var_prior0
			self.args.ae_shrd.loss_rec, self.args.ae_shrd.var_rec, = None, None
			self.args.ae_shrd.postdec_gate = [None, None]

		elif self.args.dataset == 'digits_fc':
			# args.score1, args.score2 = 'NMI', 'ACC'
			from util.load_data_MultiModal import mv_data_digits as mv_data
			self.args.modalities = 1
			self.args.inp_dtype = tf.float32
			data = mv_data(self.args.data_dir, self.args, self.args.SEED_gra)
			self.args.inp_shape1 = [None] + list(data.d1_in)
			self.args.inp_shape2 = [None] + list(data.d2_in)

			self.args.ae1 = struct()
			self.args.ae1.shape_enc = np.array([784, 1024, 1024, 1024, hp.hidden_dim])
			self.args.ae1.gate_enc = [hp.gate, hp.gate, hp.gate, 'linear']
			self.args.ae1.shape_dec = np.array([hp.hidden_dim, 1024, 1024, 1024, 784])
			self.args.ae1.gate_dec = [hp.gate, hp.gate, hp.gate, 'linear']
			self.args.ae1.postenc_gate = ['linear', hp.postenc1]
			self.args.ae1.postdec_gate = ['sigmoid', None]
			self.args.ae1.loss_rec = 'CE'
			self.args.ae1.var_rec, self.args.ae1.var_prior = hp.var_v1, hp.var_prior1

			self.args.ae2 = struct()
			self.args.ae2.shape_enc = np.array([256, 1024, 1024, 1024, hp.hidden_dim])
			self.args.ae2.gate_enc = [hp.gate, hp.gate, hp.gate, 'linear']
			self.args.ae2.shape_dec = np.array([hp.hidden_dim, 1024, 1024, 1024, 256])
			self.args.ae2.gate_dec = [hp.gate, hp.gate, hp.gate, 'linear']
			self.args.ae2.postenc_gate = ['linear', hp.postenc2]
			self.args.ae2.postdec_gate = ['sigmoid', None]
			self.args.ae2.loss_rec = 'CE'
			self.args.ae2.var_rec, self.args.ae2.var_prior = hp.var_v2, hp.var_prior1 * hp.var_prior2_ratio

			self.args.ae_shrd = struct()
			self.args.ae_shrd.shape_enc = np.array([784 if not self.args.hp.shrd_on_all else (784 + 256),
											   1024, 1024, 1024, self.args.hidden_dim_shrd])
			self.args.ae_shrd.gate_enc = [hp.gate, hp.gate, hp.gate, 'linear']  # 'sigmoid']
			self.args.ae_shrd.var_prior = hp.var_prior0
			self.args.ae_shrd.loss_rec, self.args.ae_shrd.var_rec, = None, None
			self.args.ae_shrd.postenc_gate = ['linear', 'sigmoid'] if hp.shrd_est == 'nn' else [None, 'sigmoid']
			self.args.ae_shrd.postdec_gate = [None, None]
			self.args.ae_shrd.shape_dec = [-1]
			self.args.ae_shrd.gate_dec = [None]

		elif self.args.dataset == 'mnist':
			from util.load_data_MultiModal import mv_data_mnist as mv_data
			self.args.modalities = 1
			self.args.inp_dtype = tf.float32
			data = mv_data(self.args.data_dir, self.args, self.args.SEED_gra)
			self.args.modalities = 1
			self.args.inp_shapes = [[None] + list(d_in_) for d_in_ in data.d_in]

			data.cluster = struct()
			if hp.clstr_mode == 'simple':
				data.cluster.k_knn = [5]  # [5, 10, 20, 30]
				data.cluster.assign_labels = 'discretize'  # 'kmeans'
			else:
				data.cluster.k_knn = [5, 10, 20, 30]  # [5] #
				data.cluster.assign_labels = 'kmeans'  # 'discretize' #
			data.cluster.affinity = 'pre_nearest_neighbors'
			self.args.cluster = data.cluster

			self.args.ae_list = []
			# ---------- 1st view -------------
			ae_ = struct()
			ae_.enc_spec = [
				{'gate': hp.gate, 'dim_out': 7, 'filter_size': [7, 7], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 10, 'filter_size': [5, 5], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 30, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 30, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
			]
			# the order othe decoder layer is reveresed, last spec is first one the decoder network
			ae_.dec_spec = [
				{'gate': 'linear', 'dim_out': 1, 'filter_size': [7, 7], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 7, 'filter_size': [5, 5], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 10, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 30, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
			]
			ae_.postenc_gate = ['linear', hp.postenc1]
			ae_.postenc_dims = [hp.hidden_dim, hp.hidden_dim]
			ae_.postdec_gate = ['sigmoid', None]
			ae_.loss_rec = 'CE'
			ae_.var_rec, ae_.var_prior = hp.var_v1, hp.var_prior1

			self.args.ae_list += [ae_]

			# ---------- shared factor --------
			self.args.ae_shrd = struct()
			self.args.ae_shrd.enc_spec = [
				{'gate': hp.gate, 'dim_out': 7, 'filter_size': [7, 7], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 10, 'filter_size': [5, 5], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 30, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 30, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
			]
			self.args.ae_shrd.dec_spec = self.args.ae_shrd.enc_spec  # dummy, it is not used
			self.args.ae_shrd.postenc_gate = ['linear', 'sigmoid'] if hp.shrd_est == 'nn' else [None, 'sigmoid']
			self.args.ae_shrd.postenc_dims = [hp.hidden_dim_shrd, hp.hidden_dim_shrd] if hp.shrd_est == 'nn' \
				else [0, hp.hidden_dim_shrd]
			self.args.ae_shrd.var_prior = hp.var_prior0
			self.args.ae_shrd.loss_rec, self.args.ae_shrd.var_rec, = None, None
			self.args.ae_shrd.postdec_gate = [None, None]

		elif self.args.dataset == 'usps':
			from util.load_data_MultiModal import mv_data_usps as mv_data
			self.args.modalities = 1
			self.args.inp_dtype = tf.float32
			data = mv_data(self.args.data_dir, self.args, self.args.SEED_gra)
			self.args.modalities = 1
			self.args.inp_shapes = [[None] + list(d_in_) for d_in_ in data.d_in]

			data.cluster = struct()
			if hp.clstr_mode == 'simple':
				data.cluster.k_knn = [5]  # [5, 10, 20, 30]
				data.cluster.assign_labels = 'discretize'  # 'kmeans'
			else:
				data.cluster.k_knn = [5, 10, 20, 30]  # [5] #
				data.cluster.assign_labels = 'kmeans'  # 'discretize' #
			data.cluster.affinity = 'pre_nearest_neighbors'
			self.args.cluster = data.cluster

			self.args.ae_list = []
			# ---------- 1st view -------------
			ae_ = struct()
			ae_.enc_spec = [
				{'gate': hp.gate, 'dim_out': 7, 'filter_size': [7, 7], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 10, 'filter_size': [5, 5], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 15, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 15, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
			]
			ae_.dec_spec = [
				{'gate': 'linear', 'dim_out': 1, 'filter_size': [7, 7], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 7, 'filter_size': [5, 5], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 10, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 15, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
			]
			ae_.postenc_gate = ['linear', hp.postenc1]
			ae_.postenc_dims = [hp.hidden_dim, hp.hidden_dim]
			ae_.postdec_gate = ['sigmoid', None]
			ae_.loss_rec = 'CE'
			ae_.var_rec, ae_.var_prior = hp.var_v1, hp.var_prior1

			self.args.ae_list += [ae_]

			# ---------- shared factor --------
			self.args.ae_shrd = struct()
			self.args.ae_shrd.enc_spec = [
				{'gate': hp.gate, 'dim_out': 7, 'filter_size': [7, 7], 'stride': [2, 2], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 10, 'filter_size': [5, 5], 'stride': [1, 1], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 15, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
				{'gate': hp.gate, 'dim_out': 15, 'filter_size': [3, 3], 'stride': [1, 1], 'pad': 'SAME'},
			]
			self.args.ae_shrd.dec_spec = self.args.ae_shrd.enc_spec  # dummy, it is not used
			self.args.ae_shrd.postenc_gate = ['linear', 'sigmoid'] if hp.shrd_est == 'nn' else [None, 'sigmoid']
			self.args.ae_shrd.postenc_dims = [hp.hidden_dim_shrd, hp.hidden_dim_shrd] if hp.shrd_est == 'nn' \
				else [0, hp.hidden_dim_shrd]
			self.args.ae_shrd.var_prior = hp.var_prior0
			self.args.ae_shrd.loss_rec, self.args.ae_shrd.var_rec, = None, None
			self.args.ae_shrd.postdec_gate = [None, None]

		else:
			raise ValueError('not a valid dataset')

		return data


	def load_model(self):
		#########################
		# load the model
		#########################
		if self.args.dataset == 'n-mnist':
			import model_VPCCA as mdl

		elif self.args.dataset == 'yaleb':
			import model_MVCCA_CNN as mdl
			if self.args.reconst == 'one2all':
				import model_MVCCA_Recons1 as mdl
			if self.args.reconst in ['all2one', '345to1', '345to12']:
				import model_MVCCA_Recons_all2one as mdl
			if self.args.reconst == 'all2one':
				self.args.reconst_views = [[0], [1, 2, 3, 4]]
			elif self.args.reconst == '345to12':
				self.args.reconst_views = [[0, 1], [2, 3, 4]]

		elif self.args.dataset == 'mnist': # for uni-modal clustering
			import model_MVCCA_CNN as mdl
		elif self.args.dataset == 'usps': # for uni-modal clustering
			import model_MVCCA_CNN as mdl
		elif self.args.dataset == 'digits_fc':
			import model_VCCA2 as mdl
		elif self.args.dataset == 'digits':
			import model_MVCCA_CNN as mdl

		self.model = mdl.model_VCCA(self.args, data=self.data)


	def train(self):

		# Train
		# sess.graph.finalize()
		valid_loss_best = np.inf

		self.model.init_global_step0()
		self.model.init_epoch = self.model.sess.run(self.model.epoch) + 1
		results_list = []
		final_output = []
		sim_measure_list = []
		# best_misclass_err_tu, best_misclass_err_te, misclass_err_tu, misclass_err_te = 2., 2., 1., 1.

		# revoke
		if self.model.init_epoch > 0:
			_old_eval_ = 'OLD BEST '
			for key_ in self.model.best_score_tu:
				_old_eval_ += ': ' + key_ + ' _tu =' + str(100 * self.model.val(self.model.best_score_tu[key_])) \
							  + ', ' + key_ + '_te =' + str(100 * self.model.val(self.model.best_score_te[key_]))
			self._print(_old_eval_, 'log')

		self.model._t0 = time.time()
		while self.model.val(self.model.epoch) < self.args.n_epoch:
			self.model.epoch_inc()
			epoch = self.model.val(self.model.epoch)

			self.args.to_eval_now = True if ((epoch % self.args.eval_interval == 0 or epoch == self.args.n_epoch - 1)
											and self.model.completed_warmup and epoch >= self.args.warmupDelay - 1) \
				else False

			results_dict = self.model.train_epoch(epoch, self.data, )
			results_list.append(results_dict)

			if self.args.compute_corr:
				n_samples_ = 1000

				# add some tensors for computing similarities
				if not hasattr(self.model, 'Phi_tensor_samples'):
					self.model.Z1_tensor_samples, self.model.Z2_tensor_samples, self.model.EPS1_tensor_samples, self.model.EPS2_tensor_samples, self.model.Phi_tensor_samples = \
						self.model.CA.draw_2v_samples(
							Phi_mean=self.model.Phi_mean, Phi_sigmasq=self.model.Phi_sigmasq,
							Eps1_mean=self.model.Eps1_mean, Eps1_sigmasq=self.model.Eps1_sigmasq,
							Eps2_mean=self.model.Eps2_mean, Eps2_sigmasq=self.model.Eps2_sigmasq, n_samples=n_samples_)

				latent_tensors_ = [self.model.Phi_mean, self.model.Eps1_mean, self.model.Eps2_mean, self.model.Phi_sigmasq,
								   self.model.Eps1_sigmasq,
								   self.model.Eps2_sigmasq] + \
								  [self.model.Phi_tensor_samples, self.model.EPS1_tensor_samples, self.model.EPS2_tensor_samples] + \
								  [self.model.Z1_tensor_samples, self.model.Z2_tensor_samples]
				latent_dictKeys = ['Phi_mean', 'Eps1_mean', 'Eps2_mean', 'Phi_sigmasq', 'Eps1_sigmasq',
								   'Eps2_sigmasq'] + \
								  ['Phi', 'Eps1', 'Eps2'] + ['Z1', 'Z2']

				sim_measure_ = self._compute_correlation(
					x_value=[self.data.x1_tu[::5], self.data.x2_tu[::5]],
					n_samples=n_samples_, model=self.model, epoch=epoch, 
					latent_tensors=latent_tensors_, latent_dictKeys=latent_dictKeys
				)
				
				sim_measure_list.append(sim_measure_)
				np.save(self.args.logdir + 'similarity_measures.npy', sim_measure_list)

			if epoch < self.args.warmupDelay - 1:  # or (not completed_warmup and args.mode == "cross-validate"):
				continue  # to skip the remaining statements in the current iteration of the loop

			if (epoch % self.args.save_interval == 0) or (epoch == 1) or (
					epoch == self.args.warmupDelay + 1):  # and args.tf_save != 'ignore'
				if np.isnan(results_dict['obj_tr']):
					self._print("Loss is nan. Reverting to last saved checkpoint ...", 'log')
					self.model.restore(self.args.logdir + self.args.ckpt_filename)
					self.model.load_var(self.model.lr0, self.model.val(self.model.lr0) * 0.9)
					continue
				else:
					self.model.save(self.args.logdir + self.args.ckpt_filename)
					print("Model saved in file: %s" % self.args.logdir + self.args.ckpt_filename)

			if self.args.to_eval_now:
				# ------- inference 2v self.model-------
				self.model.latent_mean_list = [self.model.Phi_mean]
				self.model.latent_dict2ind = {'phi': 0}
				if self.args.reconst in ['one2all', 'all2one', '345to12']:
					self.model.latent_mean_list = [self.model.NLL_rec]
					self.model.latent_dict2ind = {'nllrec': 0}

				if self.is_nmnist:
					if set(self.args.eval_methods).intersection(['classv1', 'NMIv1', 'ACCv1']):
						self.model.latent_dict2ind.update({'phi_v1': len(self.model.latent_mean_list)})
						self.model.latent_mean_list.append(self.model.Phi_mean_v1)
					if set(self.args.eval_methods).intersection(['classv2', 'NMIv2', 'ACCv2']):
						self.model.latent_dict2ind.update({'phi_v2': len(self.model.latent_mean_list)})
						self.model.latent_mean_list.append(self.model.Phi_mean_v2)

					self.model.CA.latent_tr_list = self.model.encode_batched(self.model.latent_mean_list,
																	x1_value=self.data.x1_tr, x2_value=self.data.x2_tr,
																	bs=self.args.batch_size_valid)
					self.model.CA.latent_tu_list = self.model.encode_batched(self.model.latent_mean_list,
																	x1_value=self.data.x1_tu, x2_value=self.data.x2_tu,
																	bs=self.args.batch_size_valid)
					self.model.CA.latent_te_list = self.model.encode_batched(self.model.latent_mean_list,
																	x1_value=self.data.x1_te, x2_value=self.data.x2_te,
																	bs=self.args.batch_size_valid)

					# compute OBJective values for tune dataset
					elbo_tu, NLL1_tu, NLL2_tu = self.model.test(x1_value=self.data.x1_tu,
														   x2_value=self.data.x2_tu,
														   bs=self.args.batch_size_valid)
					# compute OBJective values for test dataset
					elbo_te, NLL1_te, NLL2_te = self.model.test(x1_value=self.data.x1_te,
														   x2_value=self.data.x2_te,
														   bs=self.args.batch_size_valid)

					self._print(">>>>>Epoch| TUNE: obj,  NLL1,  NLL2 | TEST: obj,  NLL1,  NLL2", 'log')
					self._print(">>>>>%05d|      %.4g   %.4g   %.4g|    %.4g   %.4g   %.4g "
									% (epoch, elbo_tu, NLL1_tu, NLL2_tu, elbo_te, NLL1_te, NLL2_te), 'log')

					dict_out = {'CV_config': self.args.hpconfig, 'epoch': epoch, 'msg': '',
								'obj_tu': elbo_tu, 'NLL1_tu': NLL1_tu, 'NLL2_tu': NLL2_tu,
								'obj_te': elbo_te, 'NLL1_te': NLL1_te, 'NLL2_te': NLL2_te}

				else:
					self.model.CA.latent_tr_list = self.model.run_batched(self.model.latent_mean_list,
																x_values=self.data.x_tr,
																bs=self.args.batch_size_valid)
					self.model.CA.latent_tu_list = self.model.run_batched(self.model.latent_mean_list,
																x_values=self.data.x_tu,
																bs=self.args.batch_size_valid)
					self.model.CA.latent_te_list = self.model.run_batched(self.model.latent_mean_list,
																x_values=self.data.x_te,
																bs=self.args.batch_size_valid)

					# compute OBJective values for tune dataset
					out_ = self.model.test(x_values=self.data.x_tu, bs=self.args.batch_size_valid)
					elbo_tu, NLL_tu = out_[0], out_[1:]
					# compute OBJective values for test dataset
					out_ = self.model.test(x_values=self.data.x_te, bs=self.args.batch_size_valid)
					elbo_te, NLL_te = out_[0], out_[1:]

					str_NLL = ""
					str_NLL = str_NLL.join([',  NLL%d' % i for i in range(1, self.args.modalities + 1)])
					self._print(">>>>> EPOCH| TUNE: obj%s || TEST: obj%s " % (str_NLL, str_NLL), 'log')
					str_ = ">>>>> %05d|    %.4g" + "   %.4g" * self.args.modalities + "|| %.4g" + "   %.4g" * self.args.modalities
					self._print(str_ % ((epoch, elbo_tu) + tuple(NLL_tu) + (elbo_te,) + tuple(NLL_te)), 'log')

					dict_out = {'CV_config': self.args.hpconfig, 'epoch': epoch, 'msg': '',
								'obj_tu': elbo_tu, 'NLL_tu': NLL_tu,
								'obj_te': elbo_te, 'NLL_te': NLL_te}

				dict_out = self._eval_method_fun(self.args.eval_methods, model=self.model,
										   data=self.data, dict_out=dict_out, epoch=epoch)

				final_output.append(dict_out)
				np.save(self.args.logdir + self.npy_result_file_running, final_output)
				if epoch == self.args.n_epoch - 1:
					np.save(self.args.logdir + self.npy_result_file, final_output)

				results_dict.update(dict_out)

	def infer(self, data=None):
		data = self.data if data is None else data
		if self.is_nmnist:
			return self._infer_NoisyMNIST(data=data)
		else:
			return self._infer_cluster(data=data)
	
	def _infer_NoisyMNIST(self, data=None):
		data = self.data if data is None else data
		epoch = self.model.val(self.model.epoch)

		self.model.latent_mean_list = [self.model.Phi_mean, self.model.Eps1_mean, self.model.Eps2_mean]
		self.model.latent_dict2ind = {'phi_mean': 0, 'eps1_mean': 1, 'eps2_mean': 2}
		if set(self.args.eval_methods).intersection(['classv1', 'NMIv1', 'ACCv1']):
			self.model.latent_dict2ind.update({'phi_v1': len(self.model.latent_mean_list)})
			self.model.latent_mean_list.append(self.model.Phi_mean_v1)
		if set(self.args.eval_methods).intersection(['classv2', 'NMIv2', 'ACCv2']):
			self.model.latent_dict2ind.update({'phi_v2': len(self.model.latent_mean_list)})
			self.model.latent_mean_list.append(self.model.Phi_mean_v2)

		_, _, EPS1_tensor_sample, EPS2_tensor_sample, Phi_tensor_sample = \
			self.model.CA.draw_2v_samples(
				Phi_mean=self.model.Phi_mean, Phi_sigmasq=self.model.Phi_sigmasq,
				Eps1_mean=self.model.Eps1_mean, Eps1_sigmasq=self.model.Eps1_sigmasq,
				Eps2_mean=self.model.Eps2_mean, Eps2_sigmasq=self.model.Eps2_sigmasq, n_samples=1)
		tensors_ = self.model.latent_mean_list + [Phi_tensor_sample, EPS1_tensor_sample, EPS2_tensor_sample]
		self.model.latent_dict2ind.update({'Phi': len(self.model.latent_mean_list),
									  'Eps1': len(self.model.latent_mean_list) + 1,
									  'Eps2': len(self.model.latent_mean_list) + 2})

		self.model.CA.latent_tr_list = self.model.encode_batched(tensors_,
														x1_value=data.x1_tr, x2_value=data.x2_tr,
														bs=self.args.batch_size_valid)
		self.model.CA.latent_tu_list = self.model.encode_batched(tensors_,
														x1_value=data.x1_tu, x2_value=data.x2_tu,
														bs=self.args.batch_size_valid)
		self.model.CA.latent_te_list = self.model.encode_batched(tensors_,
														x1_value=data.x1_te, x2_value=data.x2_te,
														bs=self.args.batch_size_valid)

		# compute OBJective values for tune dataset
		elbo_tu, NLL1_tu, NLL2_tu = self.model.test(x1_value=data.x1_tu,
											   x2_value=data.x2_tu,
											   bs=self.args.batch_size_valid)
		# compute OBJective values for test dataset
		elbo_te, NLL1_te, NLL2_te = self.model.test(x1_value=data.x1_te,
											   x2_value=data.x2_te,
											   bs=self.args.batch_size_valid)

		self._print(">>>>>Epoch| TUNE: obj,  NLL1,  NLL2 | TEST: obj,  NLL1,  NLL2", 'log')
		self._print(">>>>>%05d|      %.4g   %.4g   %.4g|    %.4g   %.4g   %.4g "
						% (epoch, elbo_tu, NLL1_tu, NLL2_tu, elbo_te, NLL1_te, NLL2_te), 'log')

		dict_out = {'CV_config': self.args.hpconfig, 'epoch': epoch, 'msg': '',
					'obj_tu': elbo_tu, 'NLL1_tu': NLL1_tu, 'NLL2_tu': NLL2_tu,
					'obj_te': elbo_te, 'NLL1_te': NLL1_te, 'NLL2_te': NLL2_te}

		from util.downstream_tasks import tsne_embeding
		downsample_tsne = 2

		# mean of the latent factors
		# Phi_tr_ = self.self.model.CA.latent_tr_list[self.self.model.latent_dict2ind['phi']]
		# Phi_tu_ = self.self.model.CA.latent_tu_list[self.self.model.latent_dict2ind['phi']]
		Phi_te_ = self.model.CA.latent_te_list[self.model.latent_dict2ind['phi_mean']]
		z_tsne_mess = tsne_embeding(Phi_te_[::downsample_tsne], data.label_te[::downsample_tsne],
									self.args.logdir + "Phi_mean_te_tsne.eps")
		if set(self.args.eval_methods).intersection(['classv1', 'NMIv1', 'ACCv1']):
			Phi_te_ = self.model.CA.latent_te_list[self.model.latent_dict2ind['phi_v1']]
			z_tsne_mess = tsne_embeding(Phi_te_[::downsample_tsne], data.label_te[::downsample_tsne],
										self.args.logdir + "Phi_mean_tev1_tsne.eps")

		Eps_te_ = self.model.CA.latent_te_list[self.model.latent_dict2ind['eps1_mean']]
		z_tsne_mess = tsne_embeding(Eps_te_[::downsample_tsne], data.label_te[::downsample_tsne],
									self.args.logdir + "Eps1_mean_te_tsne.eps")

		Eps_te_ = self.model.CA.latent_te_list[self.model.latent_dict2ind['eps2_mean']]
		z_tsne_mess = tsne_embeding(Eps_te_[::downsample_tsne], data.label_te[::downsample_tsne],
									self.args.logdir + "Eps2_mean_te_tsne.eps")

		# samples from latent factors
		Phi_te_ = self.model.CA.latent_te_list[self.model.latent_dict2ind['Phi']]
		z_tsne_mess = tsne_embeding(Phi_te_[::downsample_tsne], data.label_te[::downsample_tsne],
									self.args.logdir + "Phi_te_tsne.eps")

		Eps_te_ = self.model.CA.latent_te_list[self.model.latent_dict2ind['Eps1']]
		z_tsne_mess = tsne_embeding(Eps_te_[::downsample_tsne], data.label_te[::downsample_tsne],
									self.args.logdir + "Eps1_te_tsne.eps")

		Eps_te_ = self.model.CA.latent_te_list[self.model.latent_dict2ind['Eps2']]
		z_tsne_mess = tsne_embeding(Eps_te_[::downsample_tsne], data.label_te[::downsample_tsne],
									self.args.logdir + "Eps2_te_tsne.eps")

		dict_out = self._eval_method_fun(self.args.eval_methods, model=self.model,
								   data=data, dict_out=dict_out, epoch=epoch)

	def _infer_cluster(self, data=None):
		data = self.data if data is None else data
		epoch = self.model.val(self.model.epoch)

		self.model.Z_tensor_samples, self.model.EPS_tensor_samples, self.model.Phi_tensor_samples = \
			self.model.CA.draw_samples(
				Phi_mean=self.model.Phi_mean, Phi_sigmasq=self.model.Phi_sigmasq,
				Eps_mean=self.model.Eps_mean, Eps_sigmasq=self.model.Eps_sigmasq, n_samples=1)

		tensors_ = [self.model.Phi_mean,
					self.model.Phi_tensor_samples] + self.model.Eps_mean + self.model.EPS_tensor_samples + self.model.Z_tensor_samples  # [self.self.model.Phi_sigmasq] + self.self.model.Eps_sigmasq + \
		latent_dictKeys = ['Phi_mean', 'Phi'] + \
						  ['Eps%d_mean' % d_ for d_ in range(1, self.args.modalities + 1)] + \
						  ['Eps%d' % d_ for d_ in range(1, self.args.modalities + 1)] + \
						  ['Z%d' % d_ for d_ in range(1, self.args.modalities + 1)]

		out_np_ = self.model.run_batched(tensors_,
									x_values=data.x_tr,
									bs=self.args.batch_size_valid)
		self.model.CA.latent_tr_list = dict(zip(latent_dictKeys, out_np_))

		out_np_ = self.model.run_batched(tensors_,
									x_values=data.x_tu,
									bs=self.args.batch_size_valid)
		self.model.CA.latent_tu_list = dict(zip(latent_dictKeys, out_np_))

		out_np_ = self.model.run_batched(tensors_,
									x_values=data.x_te,
									bs=self.args.batch_size_valid)
		self.model.CA.latent_te_list = dict(zip(latent_dictKeys, out_np_))

		# compute OBJective values for tune dataset
		out_ = self.model.test(x_values=data.x_tu, bs=self.args.batch_size_valid)
		elbo_tu, NLL_tu = out_[0], out_[1:]

		out_ = self.model.test(x_values=data.x_te, bs=self.args.batch_size_valid)
		elbo_te, NLL_te = out_[0], out_[1:]

		str_NLL = ""
		str_NLL = str_NLL.join([',  NLL%d' % i for i in range(1, self.args.modalities + 1)])
		self.args._print(">>>>> EPOCH| TUNE: obj%s || TEST: obj%s " % (str_NLL, str_NLL), 'log')
		str_ = ">>>>> %05d|    %.4g" + "   %.4g" * self.args.modalities + "|| %.4g" + "   %.4g" * self.args.modalities
		self.args._print(str_ % ((epoch, elbo_tu) + tuple(NLL_tu) + (elbo_te,) + tuple(NLL_te)), 'log')

		from util.downstream_tasks import tsne_embeding
		downsample_tsne = 2

		# mean of the latent factors
		# Phi_tr_ = self.self.model.CA.latent_tr_list[self.self.model.latent_dict2ind['phi']]
		# Phi_tu_ = self.self.model.CA.latent_tu_list[self.self.model.latent_dict2ind['phi']]
		Phi_te_ = self.model.CA.latent_te_list['Phi_mean']
		z_tsne_mess = tsne_embeding(Phi_te_[::downsample_tsne], data.label_te[::downsample_tsne],
									self.args.logdir + "Phi_mean_te_tsne.eps")
		# samples from latent shared
		Phi_te_ = self.model.CA.latent_te_list['Phi']
		z_tsne_mess = tsne_embeding(Phi_te_[::downsample_tsne], data.label_te[::downsample_tsne],
									self.args.logdir + "Phi_te_tsne.eps")

		for d_ in range(1, self.args.modalities + 1):
			Eps_te_ = self.model.CA.latent_te_list['Eps%d_mean' % d_]
			z_tsne_mess = tsne_embeding(Eps_te_[::downsample_tsne], data.label_te[::downsample_tsne],
										self.args.logdir + "Eps%d_mean_te_tsne.eps" % d_)

			Eps_te_ = self.model.CA.latent_te_list['Eps%d' % d_]
			z_tsne_mess = tsne_embeding(Eps_te_[::downsample_tsne], data.label_te[::downsample_tsne],
										self.args.logdir + "Eps%d_te_tsne.eps" % d_)

	def _eval_method_fun(self, eval_methods, model, data, dict_out, epoch):

		if 'class' in eval_methods :
			key = 'class'
			ckpt_file_name_ = self.args.ckpt_best_filename + key + ".ckpt"
			Phi_tr_= model.CA.latent_tr_list[model.latent_dict2ind['phi']]
			Phi_tu_= model.CA.latent_tu_list[model.latent_dict2ind['phi']]
			Phi_te_= model.CA.latent_te_list[model.latent_dict2ind['phi']]
			class_err_tu, class_err_te = \
				linear_svm_classify(Phi_tr_, data.label_tr,
									Phi_tu_, data.label_tu,
									Phi_te_, data.label_te,
									cache_size=1000,
									alg='LinearSVC' if self.args.fast_SVM else 'SVC')
	
			model.load_var(model.score_tu[key], class_err_tu)
			model.load_var(model.score_te[key], class_err_te)
			if class_err_tu < model.val(model.best_score_tu[key]):
				model.load_var(model.best_score_tu[key], class_err_tu)
				model.load_var(model.best_score_te[key], class_err_te)
				dict_out['msg'] += ' *'
				if self.args.tf_save != 'ignore':
					model.save(self.args.logdir + ckpt_file_name_)
	
			dtime = time.time() - model._t0
			self._print("EPOCH: %d, time:%ds, tune_err = %f, test_err=%f %s"
							% (epoch, dtime, 100 * class_err_tu, 100 * class_err_te, dict_out['msg']), 'log')
	
			dict_out.update({'class_err_tu': class_err_tu, 'class_err_te': class_err_te,
							 'best_class_err_tu': model.val(model.best_score_tu[key]),
							 'best_class_err_te': model.val(model.best_score_te[key])})
	
		if 'classv1' in eval_methods :
			# im = self.self.args.eval_mtd_ind['class']
			key = 'classv1'
			ckpt_file_name_ = None #self.self.args.ckpt_best_filename+ key + ".ckpt"
			Phi_tr_= model.CA.latent_tr_list[model.latent_dict2ind['phi_v1']]
			Phi_tu_= model.CA.latent_tu_list[model.latent_dict2ind['phi_v1']]
			Phi_te_= model.CA.latent_te_list[model.latent_dict2ind['phi_v1']]
			class_err_tu, class_err_te = \
				linear_svm_classify(Phi_tr_, data.label_tr,
									Phi_tu_, data.label_tu,
									Phi_te_, data.label_te,
									cache_size=1000,
									alg='LinearSVC' if self.args.fast_SVM else 'SVC')
	
			model.load_var(model.score_tu[key], class_err_tu)
			model.load_var(model.score_te[key], class_err_te)
			if class_err_tu < model.val(model.best_score_tu[key]):
				model.load_var(model.best_score_tu[key], class_err_tu)
				model.load_var(model.best_score_te[key], class_err_te)
				dict_out['msg'] += ' *'
				if self.args.tf_save != 'ignore' and ckpt_file_name_:
					model.save(self.args.logdir + ckpt_file_name_)
	
			dtime = time.time() - model._t0
			self._print("EPOCH: %d, time:%ds, tune_err_v1 = %f, test_err_v1 =%f %s"
							% (epoch, dtime, 100 * class_err_tu, 100 * class_err_te, dict_out['msg']), 'log')
	
			dict_out.update({'class_err_tu_v1': class_err_tu, 'class_err_te_v1': class_err_te,
							 'best_class_err_tu_v1': model.val(model.best_score_tu[key]),
							 'best_class_err_te_v1': model.val(model.best_score_te[key])})

		if 'classEYB' in eval_methods:  # this is for extended yale-b multi-modal
			# im = self.self.args.eval_mtd_ind['class']
			key = 'classEYB'
			C_list_ = [1.0, 10.0]
			ckpt_file_name_ = self.args.ckpt_best_filename + key + ".ckpt"
			Phi_tr_ = model.CA.latent_tr_list[model.latent_dict2ind['phi']]
			Phi_tu_ = model.CA.latent_tu_list[model.latent_dict2ind['phi']]
			Phi_te_ = model.CA.latent_te_list[model.latent_dict2ind['phi']]
			class_err_tu, class_err_te = \
				linear_svm_classify2(Phi_tr_, data.label_tr,
									 Phi_tu_[0::2], data.label_tu[0::2],
									 Phi_tu_[1::2], data.label_tu[1::2],
									 cache_size=1000, C_list=C_list_,
									 alg='LinearSVC' if self.args.fast_SVM else 'SVC')
			# self._print("classification with alg: %s, tune_err = %f, test_err=%f " %('LinearSVC' if self.self.args.fast_SVM else 'SVC',
			# 																		 class_err_tu, class_err_te), 'log')
			#
			# class_err_tu_, class_err_te_ = \
			# 	linear_svm_classify(Phi_tr_, data.label_tr,
			# 						Phi_tu_[0::2], data.label_tu[0::2],
			# 						Phi_tu_[1::2], data.label_tu[1::2],
			# 						cache_size=1000, C_list=C_list_,
			# 						alg='LinearSVC' if not self.self.args.fast_SVM else 'SVC')
			# self._print("classification with alg: %s, tune_err = %f, test_err=%f " %('LinearSVC' if not self.self.args.fast_SVM else 'SVC',
			# 																		 class_err_tu_, class_err_te_), 'log')

			model.load_var(model.score_tu[key], class_err_tu)
			model.load_var(model.score_te[key], class_err_te)
			if class_err_tu < model.val(model.best_score_tu[key]):
				model.load_var(model.best_score_tu[key], class_err_tu)
				model.load_var(model.best_score_te[key], class_err_te)
				dict_out['msg'] += ' *'
				if self.args.tf_save != 'ignore':
					model.save(self.args.logdir + ckpt_file_name_)

			dtime = time.time() - model._t0
			self._print("EPOCH: %d, time:%ds, tune_err = %f, test_err=%f %s"
						% (epoch, dtime, 100 * class_err_tu, 100 * class_err_te, dict_out['msg']), 'log')

			dict_out.update({'class_err_tu': class_err_tu, 'class_err_te': class_err_te,
							 'best_class_err_tu': model.val(model.best_score_tu[key]),
							 'best_class_err_te': model.val(model.best_score_te[key])})

		if 'classCCA' in eval_methods :
			# im = self.self.args.eval_mtd_ind['class']
			key = 'classCCA'
			ckpt_file_name_ = None #self.self.args.ckpt_best_filename+ key + ".ckpt"
			class_err_tu, class_err_te = \
				linear_svm_classify(model.CA.Phi_tr_cca1, data.label_tr,
									model.CA.Phi_tu_cca1, data.label_tu,
									model.CA.Phi_te_cca1, data.label_te,
									cache_size=1000,
									alg='LinearSVC' if self.args.fast_SVM else 'SVC')

			model.load_var(model.score_tu[key], class_err_tu)
			model.load_var(model.score_te[key], class_err_te)
			if class_err_tu < model.val(model.best_score_tu[key]):
				model.load_var(model.best_score_tu[key], class_err_tu)
				model.load_var(model.best_score_te[key], class_err_te)
				dict_out['msg'] += ' *'
				if self.args.tf_save != 'ignore' and ckpt_file_name_:
					model.save(self.args.logdir + ckpt_file_name_)

			dtime = time.time() - model._t0
			self._print("CCA class error, time:%ds, tune_err = %f, test_err=%f %s"
							% (dtime, 100 * class_err_tu, 100 * class_err_te, dict_out['msg']), 'log')

			dict_out.update({'class_errCCA_tu': class_err_tu, 'class_errCCA_te': class_err_te,
							 'best_class_errCCA_tu': model.val(model.best_score_tu[key]),
							 'best_class_errCCA_te': model.val(model.best_score_te[key])})

		if ('NMI' in eval_methods):  # and ('ACC' in eval_methods):
			key1 = 'NMI'
			key2 = 'ACC'
			ckpt_file_name_ = self.args.ckpt_best_filename + key1 + ".ckpt"
			# spectral clustering
			Phi_tr_ = model.CA.latent_tr_list[model.latent_dict2ind['phi']]
			Phi_tu_ = model.CA.latent_tu_list[model.latent_dict2ind['phi']]
			Phi_te_ = model.CA.latent_te_list[model.latent_dict2ind['phi']]
			NMI_tu, NMI_te, ACC_tu, ACC_te = \
				spectral_clustering(Phi_tr_, data.label_tr,
									Phi_tu_, data.label_tu,
									Phi_te_, data.label_te,
									compute_knn_graph=True, n_clusters=10) # todo: is this NMI used in any clustering exp?

			model.load_var(model.score_tu[key1], NMI_tu)
			model.load_var(model.score_te[key1], NMI_te)
			if NMI_tu > model.val(model.best_score_tu[key1]):
				dict_out['msg'] += '#'
				model.load_var(model.best_score_tu[key1], NMI_tu)
				model.load_var(model.best_score_te[key1], NMI_te)
				if self.args.tf_save != 'ignore' and ckpt_file_name_:
					model.save(self.args.logdir + ckpt_file_name_)

			model.load_var(model.score_tu[key2], ACC_tu)
			model.load_var(model.score_te[key2], ACC_te)
			if ACC_tu > model.val(model.best_score_tu[key2]):
				dict_out['msg'] += '$'
				model.load_var(model.best_score_tu[key2], ACC_tu)
				model.load_var(model.best_score_te[key2], ACC_te)

			dict_out.update({'NMI_tu': NMI_tu, 'NMI_te': NMI_te,
							 'best_NMI_tu': model.val(model.best_score_tu[key1]),
							 'best_NMI_te': model.val(model.best_score_te[key1]),
							 'ACC_tu': ACC_tu, 'ACC_te': ACC_te,
							 'best_ACC_tu': model.val(model.best_score_tu[key2]),
							 'best_ACC_te': model.val(model.best_score_te[key2])})

			self._print(
				"SPECTRAL CLUSTERING, time:%ds: tune_NMI = %f, test_NMI=%f | tune_ACC = %f, test_ACC=%f   %s"
				% (time.time() - model._t0, 100 * NMI_tu, 100 * NMI_te, 100 * ACC_tu, 100 * ACC_te, dict_out['msg']),
				'log')

		if ('NMIv1' in eval_methods):  # and ('ACCv1' in eval_methods):
			key1 = 'NMIv1'
			key2 = 'ACCv1'
			# spectral clustering
			Phi_tr_ = model.CA.latent_tr_list[model.latent_dict2ind['phi_v1']]
			Phi_tu_ = model.CA.latent_tu_list[model.latent_dict2ind['phi_v1']]
			Phi_te_ = model.CA.latent_te_list[model.latent_dict2ind['phi_v1']]
			NMI_tu, NMI_te, ACC_tu, ACC_te = \
				spectral_clustering(Phi_tr_, data.label_tr,
									Phi_tu_, data.label_tu,
									Phi_te_, data.label_te,
									compute_knn_graph=True, n_clusters=10)

			model.load_var(model.score_tu[key1], NMI_tu)
			model.load_var(model.score_te[key1], NMI_te)
			if NMI_tu > model.val(model.best_score_tu[key1]):
				dict_out['msg'] += '#'
				model.load_var(model.best_score_tu[key1], NMI_tu)
				model.load_var(model.best_score_te[key1], NMI_te)

			model.load_var(model.score_tu[key2], ACC_tu)
			model.load_var(model.score_te[key2], ACC_te)
			if ACC_tu > model.val(model.best_score_tu[key2]):
				dict_out['msg'] += '$'
				model.load_var(model.best_score_tu[key2], ACC_tu)
				model.load_var(model.best_score_te[key2], ACC_te)

			dict_out.update({'NMI_tu_v1': NMI_tu, 'NMI_te_v1': NMI_te,
							 'best_NMI_tu_v1': model.val(model.best_score_tu[key1]),
							 'best_NMI_te_v1': model.val(model.best_score_te[key1]),
							 'ACC_tu_v1': ACC_tu, 'ACC_te_v1': ACC_te,
							 'best_ACC_tu_v1': model.val(model.best_score_tu[key2]),
							 'best_ACC_te_v1': model.val(model.best_score_te[key2])})

			self._print(
				"SPECTRAL CLUSTERING, time:%ds: tune_NMI_v1 = %f, test_NMI_v1=%f | tune_ACC_v1 = %f, test_ACC_v1=%f   %s"
				% (time.time() - model._t0, 100 * NMI_tu, 100 * NMI_te, 100 * ACC_tu, 100 * ACC_te, dict_out['msg']),
				'log')

		if ('NMI0' in eval_methods):  # and ('ACC' in eval_methods):
			key1 = 'NMI0'
			key2 = 'ACC0'
			key3 = 'ARI0'

			cluster_setting_ = struct()
			cluster_setting_.k_knn = [5]
			cluster_setting_.assign_labels = 'discretize'  # 'kmeans'
			cluster_setting_.affinity = 'pre_nearest_neighbors'

			ckpt_file_name_ = self.args.ckpt_best_filename + key1 + ".ckpt"
			# spectral clustering
			Phi_tr_ = model.CA.latent_tr_list[model.latent_dict2ind['phi']]
			Phi_tu_ = model.CA.latent_tu_list[model.latent_dict2ind['phi']]
			Phi_te_ = model.CA.latent_te_list[model.latent_dict2ind['phi']]
			NMI_tu, NMI_te, ACC_tu, ACC_te, ARI_tu, ARI_te = \
				spectral_clustering2(x_tr=Phi_tu_, label_tr=data.label_tu,
									 x_te=Phi_te_, label_te=data.label_te,
									 n_clusters=data.n_labels, assign_labels=cluster_setting_.assign_labels,
									 k_knn=cluster_setting_.k_knn, affinity=cluster_setting_.affinity, degree_poly=8)

			model.load_var(model.score_tu[key1], NMI_tu)
			model.load_var(model.score_te[key1], NMI_te)
			if NMI_tu > model.val(model.best_score_tu[key1]):
				dict_out['msg'] += '#'
				model.load_var(model.best_score_tu[key1], NMI_tu)
				model.load_var(model.best_score_te[key1], NMI_te)
				if self.args.tf_save != 'ignore' and ckpt_file_name_:
					model.save(self.args.logdir + ckpt_file_name_)

			model.load_var(model.score_tu[key2], ACC_tu)
			model.load_var(model.score_te[key2], ACC_te)
			if ACC_tu > model.val(model.best_score_tu[key2]):
				dict_out['msg'] += '$'
				model.load_var(model.best_score_tu[key2], ACC_tu)
				model.load_var(model.best_score_te[key2], ACC_te)

			model.load_var(model.score_tu[key3], ARI_tu)
			model.load_var(model.score_te[key3], ARI_te)
			if ARI_tu > model.val(model.best_score_tu[key3]):
				dict_out['msg'] += '@'
				model.load_var(model.best_score_tu[key3], ARI_tu)
				model.load_var(model.best_score_te[key3], ARI_te)

			dict_out.update({'NMI0_tu': NMI_tu, 'NMI0_te': NMI_te,
							 'best_NMI0_tu': model.val(model.best_score_tu[key1]),
							 'best_NMI0_te': model.val(model.best_score_te[key1]),
							 'ACC0_tu': ACC_tu, 'ACC0_te': ACC_te,
							 'best_ACC0_tu': model.val(model.best_score_tu[key2]),
							 'best_ACC0_te': model.val(model.best_score_te[key2]),
							 'ARI0_tu': ARI_tu, 'ARI0_te': ARI_te,
							 'best_ARI0_tu': model.val(model.best_score_tu[key3]),
							 'best_ARI0_te': model.val(model.best_score_te[key3])})

			self._print(
				"SPECTRAL CLUSTERING, time:%ds: tune_NMI0 = %f, test_NMI0=%f | tune_ACC0 = %f, test_ACC0=%f | tune_ARI0 = %f, test_ARI0=%f   %s"
				% (time.time() - model._t0, 100 * NMI_tu, 100 * NMI_te, 100 * ACC_tu, 100 * ACC_te, 100 * ARI_tu,
				   100 * ARI_te, dict_out['msg']), 'log')

		if ('NMI00' in eval_methods):  # and ('ACC' in eval_methods):
			""" Train and test data are swaped, clustering is validated on Phi_te 
			and (Phi_tr+Phi_tu) is used for computing the final scores of clustering
			"""
			key1 = 'NMI00'
			key2 = 'ACC00'
			key3 = 'ARI00'

			cluster_setting_ = struct()
			cluster_setting_.k_knn = [5]
			cluster_setting_.assign_labels = 'discretize'  # 'kmeans'
			cluster_setting_.affinity = 'pre_nearest_neighbors'

			ckpt_file_name_ = self.args.ckpt_best_filename + key1 + ".ckpt"
			# spectral clustering
			Phi_tr_ = model.CA.latent_tr_list[model.latent_dict2ind['phi']]
			Phi_tu_ = model.CA.latent_tu_list[model.latent_dict2ind['phi']]
			Phi_te_ = model.CA.latent_te_list[model.latent_dict2ind['phi']]
			NMI_tu, NMI_te, ACC_tu, ACC_te, ARI_tu, ARI_te = \
				spectral_clustering2(x_tr=Phi_te_, label_tr=data.label_te,
									 x_te=np.concatenate([Phi_tr_, Phi_tu_], axis=0),
									 label_te=np.concatenate([data.label_tr, data.label_tu], axis=0),
									 n_clusters=data.n_labels, assign_labels=cluster_setting_.assign_labels,
									 k_knn=cluster_setting_.k_knn, affinity=cluster_setting_.affinity, degree_poly=8)

			model.load_var(model.score_tu[key1], NMI_tu)
			model.load_var(model.score_te[key1], NMI_te)
			if NMI_tu > model.val(model.best_score_tu[key1]):
				dict_out['msg'] += '#'
				model.load_var(model.best_score_tu[key1], NMI_tu)
				model.load_var(model.best_score_te[key1], NMI_te)
				if self.args.tf_save != 'ignore' and ckpt_file_name_:
					model.save(self.args.logdir + ckpt_file_name_)

			model.load_var(model.score_tu[key2], ACC_tu)
			model.load_var(model.score_te[key2], ACC_te)
			if ACC_tu > model.val(model.best_score_tu[key2]):
				dict_out['msg'] += '$'
				model.load_var(model.best_score_tu[key2], ACC_tu)
				model.load_var(model.best_score_te[key2], ACC_te)

			model.load_var(model.score_tu[key3], ARI_tu)
			model.load_var(model.score_te[key3], ARI_te)
			if ARI_tu > model.val(model.best_score_tu[key3]):
				dict_out['msg'] += '@'
				model.load_var(model.best_score_tu[key3], ARI_tu)
				model.load_var(model.best_score_te[key3], ARI_te)

			dict_out.update({'NMI00_tu': NMI_tu, 'NMI00_te': NMI_te,
							 'best_NMI00_tu': model.val(model.best_score_tu[key1]),
							 'best_NMI00_te': model.val(model.best_score_te[key1]),
							 'ACC00_tu': ACC_tu, 'ACC00_te': ACC_te,
							 'best_ACC00_tu': model.val(model.best_score_tu[key2]),
							 'best_ACC00_te': model.val(model.best_score_te[key2]),
							 'ARI00_tu': ARI_tu, 'ARI00_te': ARI_te,
							 'best_ARI00_tu': model.val(model.best_score_tu[key3]),
							 'best_ARI00_te': model.val(model.best_score_te[key3])})

			self._print(
				"SPECTRAL CLUSTERING, time:%ds: tune_NMI00 = %f, test_NMI00=%f | tune_ACC00 = %f, test_ACC00=%f | tune_ARI00 = %f, test_ARI00=%f   %s"
				% (time.time() - model._t0, 100 * NMI_tu, 100 * NMI_te, 100 * ACC_tu, 100 * ACC_te, 100 * ARI_tu,
				   100 * ARI_te, dict_out['msg']), 'log')

		if ('NMI2' in eval_methods):  # and ('ACC' in eval_methods):
			key1 = 'NMI2'
			key2 = 'ACC2'
			key3 = 'ARI2'
			ckpt_file_name_ = self.args.ckpt_best_filename + key1 + ".ckpt"
			# spectral clustering
			Phi_tr_ = model.CA.latent_tr_list[model.latent_dict2ind['phi']]
			Phi_tu_ = model.CA.latent_tu_list[model.latent_dict2ind['phi']]
			Phi_te_ = model.CA.latent_te_list[model.latent_dict2ind['phi']]
			NMI_tu, NMI_te, ACC_tu, ACC_te, ARI_tu, ARI_te = \
				spectral_clustering2(x_tr=np.concatenate([Phi_tr_, Phi_tu_], axis=0),
									 label_tr=np.concatenate([data.label_tr, data.label_tu], axis=0),
									 x_te=Phi_te_, label_te=data.label_te,
									 n_clusters=data.n_labels, assign_labels=data.cluster.assign_labels,
									 k_knn=data.cluster.k_knn, affinity=data.cluster.affinity, degree_poly=8)

			model.load_var(model.score_tu[key1], NMI_tu)
			model.load_var(model.score_te[key1], NMI_te)
			if NMI_tu > model.val(model.best_score_tu[key1]):
				dict_out['msg'] += '#'
				model.load_var(model.best_score_tu[key1], NMI_tu)
				model.load_var(model.best_score_te[key1], NMI_te)
				if self.args.tf_save != 'ignore' and ckpt_file_name_:
					model.save(self.args.logdir + ckpt_file_name_)

			model.load_var(model.score_tu[key2], ACC_tu)
			model.load_var(model.score_te[key2], ACC_te)
			if ACC_tu > model.val(model.best_score_tu[key2]):
				dict_out['msg'] += '$'
				model.load_var(model.best_score_tu[key2], ACC_tu)
				model.load_var(model.best_score_te[key2], ACC_te)

			model.load_var(model.score_tu[key3], ARI_tu)
			model.load_var(model.score_te[key3], ARI_te)
			if ARI_tu > model.val(model.best_score_tu[key3]):
				dict_out['msg'] += '@'
				model.load_var(model.best_score_tu[key3], ARI_tu)
				model.load_var(model.best_score_te[key3], ARI_te)

			dict_out.update({'NMI_tu': NMI_tu, 'NMI_te': NMI_te,
							 'best_NMI_tu': model.val(model.best_score_tu[key1]),
							 'best_NMI_te': model.val(model.best_score_te[key1]),
							 'ACC_tu': ACC_tu, 'ACC_te': ACC_te,
							 'best_ACC_tu': model.val(model.best_score_tu[key2]),
							 'best_ACC_te': model.val(model.best_score_te[key2]),
							 'ARI_tu': ARI_tu, 'ARI_te': ARI_te,
							 'best_ARI_tu': model.val(model.best_score_tu[key3]),
							 'best_ARI_te': model.val(model.best_score_te[key3])})

			self._print(
				"SPECTRAL CLUSTERING, time:%ds: tune_NMI = %f, test_NMI=%f | tune_ACC = %f, test_ACC=%f | tune_ARI = %f, test_ARI=%f   %s"
				% (time.time() - model._t0, 100 * NMI_tu, 100 * NMI_te, 100 * ACC_tu, 100 * ACC_te, 100 * ARI_tu,
				   100 * ARI_te, dict_out['msg']), 'log')

		if ('NMI3' in eval_methods):  # and ('ACC' in eval_methods):
			key1 = 'NMI3'
			key2 = 'ACC3'
			key3 = 'ARI3'
			ckpt_file_name_ = self.args.ckpt_best_filename + key1 + ".ckpt"
			# spectral clustering
			Phi_tr_ = model.CA.latent_tr_list[model.latent_dict2ind['phi']]
			Phi_tu_ = model.CA.latent_tu_list[model.latent_dict2ind['phi']]
			Phi_te_ = model.CA.latent_te_list[model.latent_dict2ind['phi']]
			NMI_tu, ACC_tu, ARI_tu = \
				spectral_clustering3(x_tr=np.concatenate([Phi_tr_, Phi_tu_], axis=0),
									 label_tr=np.concatenate([data.label_tr, data.label_tu], axis=0),
									 n_clusters=data.n_labels)

			model.load_var(model.score_tu[key1], NMI_tu)
			# self.self.model.load_var(self.self.model.score_te[key1], NMI_te)
			if NMI_tu > model.val(model.best_score_tu[key1]):
				dict_out['msg'] += '#'
				model.load_var(model.best_score_tu[key1], NMI_tu)
				# self.self.model.load_var(self.self.model.best_score_te[key1], NMI_te)
				if self.args.tf_save != 'ignore' and ckpt_file_name_:
					model.save(self.args.logdir + ckpt_file_name_)

			model.load_var(model.score_tu[key2], ACC_tu)
			# self.self.model.load_var(self.self.model.score_te[key2], ACC_te)
			if ACC_tu > model.val(model.best_score_tu[key2]):
				dict_out['msg'] += '$'
				model.load_var(model.best_score_tu[key2], ACC_tu)
			# self.self.model.load_var(self.self.model.best_score_te[key2], ACC_te)

			model.load_var(model.score_tu[key3], ARI_tu)
			# self.self.model.load_var(self.self.model.score_te[key3], ARI_te)
			if ARI_tu > model.val(model.best_score_tu[key3]):
				dict_out['msg'] += '@'
				model.load_var(model.best_score_tu[key3], ARI_tu)
			# self.self.model.load_var(self.self.model.best_score_te[key3], ARI_te)

			dict_out.update({'NMI3_tu': NMI_tu,  # 'NMI3_te': NMI_te,
							 'best_NMI3_tu': model.val(model.best_score_tu[key1]),
							 # 'best_NMI3_te': self.self.model.val(self.self.model.best_score_te[key1]),
							 'ACC3_tu': ACC_tu,  # 'ACC3_te': ACC_te,
							 'best_ACC3_tu': model.val(model.best_score_tu[key2]),
							 # 'best_ACC3_te': self.self.model.val(self.self.model.best_score_te[key2]),
							 'ARI3_tu': ARI_tu,  # 'ARI3_te': ARI_te,
							 'best_ARI3_tu': model.val(model.best_score_tu[key3]),
							 # 'best_ARI3_te': self.self.model.val(self.self.model.best_score_te[key3])
							 })

			self._print("SPECTRAL CLUSTERING, time:%ds: tune_NMI3 = %f, tune_ACC3 = %f, tune_ARI3 = %f,   %s"
						% (time.time() - model._t0, 100 * NMI_tu, 100 * ACC_tu, 100 * ARI_tu, dict_out['msg']), 'log')

		if ('nllrec' in eval_methods):
			key1 = 'nllrec'
			_t0 = time.time()
			ckpt_file_name_ = self.args.ckpt_best_filename + key1 + ".ckpt"

			key1_tr = model.CA.latent_tr_list[model.latent_dict2ind['nllrec']][0]
			key1_tu = model.CA.latent_tu_list[model.latent_dict2ind['nllrec']][0]
			key1_te = model.CA.latent_te_list[model.latent_dict2ind['nllrec']][0]

			model.load_var(model.score_tu[key1], key1_tu)
			model.load_var(model.score_te[key1], key1_te)
			if key1_tu < model.val(model.best_score_tu[key1]):
				dict_out['msg'] += '@'
				model.load_var(model.best_score_tu[key1], key1_tu)
				model.load_var(model.best_score_te[key1], key1_te)
				if self.args.tf_save != 'ignore' and ckpt_file_name_:
					model.save(self.args.logdir + ckpt_file_name_)

				# to reconstruct the secondary views based on the main view
				img_indices = [1, 64 * 2, 64 * 8 + 10, 64 * 4 + 10, 64 * 20 + 1]
				if self.args.reconst == 'one2all':
					x_rec = model.run_batched(model.xhat_mean, bs=-1,
											  x_values=[data.x_te[j][img_indices, :, :, :] for j in
														range(data.n_modalities)])
				elif self.args.reconst in ['all2one', '345to12']:
					x_rec = model.run_batched(model.xhat_mean, bs=-1,
											  x_values=[data.x_te[j][img_indices, :, :, :] * (
												  0. if j in self.args.reconst_views[0] else 1.) \
														for j in range(data.n_modalities)])
				image_list = [[x_rec[j][i, :, :, 0] for i in range(len(x_rec[0]))] for j in range(len(x_rec))]
				plot_tile_image(image_list, savefig=True, results_subfldr=self.args.logdir,
								filenames='recons_%s_epoch' % self.args.reconst + str(epoch))

			dict_out.update({'%s_tu' % key1: key1_tu, '%s_te' % key1: key1_te,
							 'best_%s_tu' % key1: model.val(model.best_score_tu[key1]),
							 'best_%s_te' % key1: model.val(model.best_score_te[key1])})

			self._print("Sum NLL reconstruction: tune_%s = %f, test_%s=%f   %s"
						% (key1, key1_tu, key1, key1_te, dict_out['msg']), 'log')

		# z_tsne_mess = tsne_embeding(self.self.model.CA.Phi_te_mess[::2], data.label_te[::2], self.self.args.logdir + "Phi_te1v_tsne")

		return dict_out
	
	def _compute_correlation(self, x_value, n_samples, model, latent_tensors, latent_dictKeys, epoch):
	
		out_np_ = model.encode_batched(latent_tensors,
									   x1_value=x_value[0], x2_value=x_value[1],
									   bs=self.args.batch_size_valid)
		latent_np = dict(zip(latent_dictKeys, out_np_))
	
		sim_measure = struct()
		# sim_measure.PhiE1 = orth_measure(X=latent_np['Phi'], Y=latent_np['Eps1'], n_samples=n_samples)
		# sim_measure.PhiE1_orig = orth_measure(X=latent_np['Phi'], Y=latent_np['Eps1'],
		# 							X_mean=latent_np['Phi_mean'], Y_mean=latent_np['Eps1_mean'],
		# 							X_sigmasq=latent_np['Phi_sigmasq'], Y_sigmasq=latent_np['Eps1_sigmasq'], n_samples=n_samples)
		# sim_measure.PhiE1_samples = orth_measure2(X=latent_np['Phi'], Y=latent_np['Eps1'])
		# sim_measure.PhiE1_means = orth_measure2(X=latent_np['Phi_mean'], Y=latent_np['Eps1_mean'])
		#
		# sim_measure.PhiE2 = orth_measure(X=latent_np['Phi'], Y=latent_np['Eps2'], n_samples=n_samples)
		# sim_measure.PhiE2_orig = orth_measure(X=latent_np['Phi'], Y=latent_np['Eps2'],
		# 										 X_mean=latent_np['Phi_mean'], Y_mean=latent_np['Eps2_mean'],
		# 										 X_sigmasq=latent_np['Phi_sigmasq'], Y_sigmasq=latent_np['Eps2_sigmasq'],
		# 										 n_samples=n_samples)
		# sim_measure.PhiE2_samples = orth_measure2(X=latent_np['Phi'], Y=latent_np['Eps2'])
		# sim_measure.PhiE2_means = orth_measure2(X=latent_np['Phi_mean'], Y=latent_np['Eps2_mean'])
		#
		# sim_measure.E1E2 = orth_measure(X=latent_np['Eps1'], Y=latent_np['Eps2'], n_samples=n_samples)
		# sim_measure.E1E2_orig = orth_measure(X=latent_np['Eps1'], Y=latent_np['Eps2'],
		# 										 X_mean=latent_np['Eps1_mean'], Y_mean=latent_np['Eps2_mean'],
		# 										 X_sigmasq=latent_np['Eps1_sigmasq'], Y_sigmasq=latent_np['Eps2_sigmasq'],
		# 										 n_samples=n_samples)
		# sim_measure.E1E2_samples = orth_measure2(X=latent_np['Eps1'], Y=latent_np['Eps2'])
		# sim_measure.E1E2_means = orth_measure2(X=latent_np['Eps1_mean'], Y=latent_np['Eps2_mean'])
		#
		# self.args._print(">>>>>Epoch: %05d| Similarity: PhiE1=%.4g,  PhiE2=%.4g,  E1E2=%.4g | MEANS PhiE1=%.4g,  PhiE2=%.4g,  E1E2=%.4g" % \
		# 			(epoch, sim_measure.PhiE1, sim_measure.PhiE2, sim_measure.E1E2 , sim_measure.PhiE1_means, sim_measure.PhiE2_means, sim_measure.E1E2_means), 'log')
	
	
		sim_measure.PhiE1_cosineSim = orth_measure3(X=latent_np['Phi'], Y=latent_np['Eps1'])
		sim_measure.PhiE2_cosineSim = orth_measure3(X=latent_np['Phi'], Y=latent_np['Eps2'])
		sim_measure.E1E2_cosineSim = orth_measure3(X=latent_np['Eps1'], Y=latent_np['Eps2'])
		sim_measure.Z1Z2_cosineSim = orth_measure3(X=latent_np['Z1'], Y=latent_np['Z2'])
		self._print(
			">>>>>Epoch: %05d| COSINE Similarity: PhiE1=%.4g,  PhiE2=%.4g,  E1E2=%.4g, Z1Z2=%.4g" % \
			(epoch, sim_measure.PhiE1_cosineSim, sim_measure.PhiE2_cosineSim, sim_measure.E1E2_cosineSim, sim_measure.Z1Z2_cosineSim), 'log')
	
		sim_measure.epoch = epoch
		sim_measure.n_samples = n_samples
		return sim_measure
	
	
	def infer_correlation(self, data=None):
		data = self.data if data is None else data
		self.model = []

		n_samples = 1000
		epoch = self.model.val(self.model.epoch)
	
		Z1_tensor_samples, Z2_tensor_samples, EPS1_tensor_sample, EPS2_tensor_sample, Phi_tensor_sample = \
			self.model.CA.draw_2v_samples(
				Phi_mean=self.model.Phi_mean, Phi_sigmasq=self.model.Phi_sigmasq,
				Eps1_mean=self.model.Eps1_mean, Eps1_sigmasq=self.model.Eps1_sigmasq,
				Eps2_mean=self.model.Eps2_mean, Eps2_sigmasq=self.model.Eps2_sigmasq, n_samples=n_samples)
	
		latent_tensors_ = [self.model.Phi_mean, self.model.Eps1_mean, self.model.Eps2_mean, self.model.Phi_sigmasq, self.model.Eps1_sigmasq, self.model.Eps2_sigmasq] + \
						  [Phi_tensor_sample, EPS1_tensor_sample, EPS2_tensor_sample] + \
						  [Z1_tensor_samples, Z2_tensor_samples]
		latent_dictKeys = ['Phi_mean', 'Eps1_mean', 'Eps2_mean', 'Phi_sigmasq', 'Eps1_sigmasq', 'Eps2_sigmasq'] + \
						  ['Phi', 'Eps1', 'Eps2'] + ['Z1', 'Z2']

		sim_measure_tu = self._compute_correlation(
			x_value= [data.x1_tu[::10], data.x2_tu[::10]],
			n_samples=n_samples,model=self.model,epoch=epoch,
			latent_tensors=latent_tensors_, latent_dictKeys=latent_dictKeys
		)

		sim_measure_te = self._compute_correlation(
			x_value= [data.x1_te[::10], data.x2_te[::10]],
			n_samples=n_samples, model=self.model, epoch=epoch,
			latent_tensors=latent_tensors_, latent_dictKeys=latent_dictKeys,
		)
	
	
		# compute OBJective values for tune dataset
		elbo_tu, NLL1_tu, NLL2_tu = self.model.test(x1_value=data.x1_tu,
											   x2_value=data.x2_tu,
											   bs=self.args.batch_size_valid)
		# compute OBJective values for test dataset
		elbo_te, NLL1_te, NLL2_te = self.model.test(x1_value=data.x1_te,
											   x2_value=data.x2_te,
											   bs=self.args.batch_size_valid)
	
		self._print(">>>>>Epoch| TUNE: obj,  NLL1,  NLL2 | TEST: obj,  NLL1,  NLL2", 'log')
		self._print(">>>>>%05d|      %.4g   %.4g   %.4g|    %.4g   %.4g   %.4g "
						% (epoch, elbo_tu, NLL1_tu, NLL2_tu, elbo_te, NLL1_te, NLL2_te), 'log')
	
		dict_out = {'CV_config': self.args.hpconfig, 'epoch': epoch, 'msg': '',
					'obj_tu': elbo_tu, 'NLL1_tu': NLL1_tu, 'NLL2_tu': NLL2_tu,
					'obj_te': elbo_te, 'NLL1_te': NLL1_te, 'NLL2_te': NLL2_te}

