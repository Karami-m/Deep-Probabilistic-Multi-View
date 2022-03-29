# from __future__ import division, print_function
import sys, os, time

import tensorflow as tf
import numpy as np
from util.misc import merge_two_dicts, list_None2negative1
from util.CCA import linCCA
from util.layers import *

SEED_op = None
debug_mode = (os.environ['debugmode'] == 'True')

gate_dict= {
	'relu': tf.nn.relu,
	'sigmoid': clipped_sigmoid,
	'tanh': tf.tanh,
	'linear': tf.identity,
	'exp': tf.exp,
	'sp':clipped_softplus,
	None: lambda x: x,
	# 'sigm_linear':
}

optimizers={'MomentumOptimizer': (lambda lr, hp: tf.train.MomentumOptimizer(lr, hp.momentum)),
			'AdamOptimizer': (lambda lr, hp : tf.train.AdamOptimizer(lr, beta1=hp.beta1, beta2=hp.beta2,
																			  epsilon=hp.eps_opt)),
			# 'AdaMaxOptimizer': (lambda lr, hp: tf.contrib.opt.AdaMaxOptimizer(lr, beta1=hp.beta1, beta2=hp.beta2,
			# 										  epsilon=hp.eps_opt)),
			'AdamaxOptimizer': (lambda lr, hp: tf.keras.optimizers.Adamax(lr, beta_1=hp.beta1, beta_2=hp.beta2,
																		  epsilon=hp.eps_opt)),
			'RMSPropOptimizer': (lambda lr, hp: tf.train.RMSPropOptimizer(lr, hp.momentum)),
			'SGD': (lambda lr, hp: tf.train.GradientDescentOptimizer(lr))
			}


class model_VCCA(object):
	def __init__(self, args, data):
		self.args = args
		self.hp = args.hp
		self.modalities = args.modalities

		# Create tensorflow session
		sess = tensorflow_session(args)
		self.sess = sess
		self.completed_warmup = False

		#########################
		# load the model and make the train op using multi GPU towers
		#########################
		graph = tf.get_default_graph()
		with graph.as_default():

			# some extra variables
			self.epoch = tf.get_variable("epoch", dtype=tf.int32, initializer=0, trainable=False)
			global_step = tf.get_variable("global_step", shape=[], dtype=tf.int64,
											 initializer=tf.zeros_initializer(), trainable=False)
			global_step0 = tf.get_variable("global_step0", shape=[], dtype=tf.int64,
											 initializer=tf.zeros_initializer(), trainable=False)
			self.ae_train_only = tf.get_variable("ae_train_only", shape=[], dtype=tf.bool,
											 initializer=tf.constant_initializer(False), trainable=False)
			self.beta_plh = tf.placeholder(tf.float32, name='beta_plh')
			self.keepprob = tf.placeholder(tf.float32, name='keepprob')
			self.lr0 = tf.get_variable("lr0", shape=[], dtype=tf.float32,
										initializer=tf.initializers.constant(args.lr), trainable=False)

			self.score_tu = {key: tf.get_variable("%s_tu" %key, dtype=tf.float32, initializer=init_,
													  trainable=False)
							  for key, init_ in args.eval_mtd_init.items()}
			self.score_te = {key: tf.get_variable("%s_te" %key, dtype=tf.float32, initializer=init_,
													  trainable=False)
							  for key, init_ in args.eval_mtd_init.items()}
			self.best_score_tu = {key: tf.get_variable("best_%s_tu" %key, dtype=tf.float32, initializer=init_,
													  trainable=False)
							  for key, init_ in args.eval_mtd_init.items()}
			self.best_score_te = {key: tf.get_variable("best_%s_te" %key, dtype=tf.float32, initializer=init_,
													  trainable=False)
							  for key, init_ in args.eval_mtd_init.items()}


			# ====== Create model
			self.AE0 = model_VAE(args, args.ae_shrd, name='VAE_shared')
			self.AE1 = model_VAE(args, args.ae1, name='VAE1')
			self.AE2 = model_VAE(args, args.ae2, name='VAE2')
			self.CA = model_CCA(args, name ='CCA')

			# optimization routine
			lr_fn = lr_scheduler(lr=self.lr0, global_step=global_step, global_step0=global_step0,
								 lr_decay=args.lr_decay, lr_decaymin=args.lr_decaymin, step_size=args.lr_stpsize,
								 epochs_warmup=args.warmup_lr, n_batches=data.n_batches, lr_scheduler=args.lr_scheduler,
								 epoch_delay=args.warmupDelay)

			increment_epoch_op = tf.assign_add(self.epoch, 1, name='epoch_inc')
			self.epoch_inc = lambda: sess.run(increment_epoch_op)
			self.lr_val = lambda: sess.run(lr_fn)

			optimizer = optimizers[args.optimizer](lr_fn, args)
			args._print(args.optimizer, verbose='log')

			with tf.variable_scope(tf.get_variable_scope(), reuse=None):
				x1 = tf.placeholder(args.inp_dtype, args.inp_shape1, name='x1')
				x2 = tf.placeholder(args.inp_dtype, args.inp_shape2, name='x2')
				self.x1, self.x2 = x1, x2

				# inference model view 1
				self.z1_mean, self.z1_sigmasq = self.AE1.encoder(x1, keepprob=self.keepprob, reuse=None)
				# self.z1_sigmasq = tf.exp(self.z1_log_sigmasq)

				# inference model view 2
				self.z2_mean, self.z2_sigmasq = self.AE2.encoder(x2, keepprob=self.keepprob, reuse=None)
				# self.z2_sigmasq = tf.exp(self.z2_log_sigmasq)

				# inference model for shared factor
				x_in_shrd = x1 if not self.hp.shrd_on_all else tf.concat([x1, x2], axis=1)
				Phi_mean_, self.p_corr = self.AE0.encoder(x_in_shrd, keepprob=self.keepprob, reuse=None)
				# self.p_corr = clipped_sigmoid(p_corr_)

				self.Eps1_mean, self.Eps1_sigmasq, self.Eps2_mean, self.Eps2_sigmasq, \
				self.Phi_mean, self.Phi_sigmasq, self.Phi_mean_v1, self.Phi_mean_v2 = \
					self.CA.fit_CCA(self.z1_mean, self.z1_sigmasq, self.z2_mean, self.z2_sigmasq, Phi_mean_, self.p_corr)

				# Calculate KL divergence for shared variable Phi and Eps1 and Eps2.
				self.KL_Phi = self.AE0.KL_gaussian(mu = self.Phi_mean, sigsq= self.Phi_sigmasq, logscale = False)
				self.KL_Eps1 = self.AE1.KL_gaussian(mu = self.Eps1_mean, sigsq= self.Eps1_sigmasq, logscale = False)
				self.KL_Eps2 = self.AE2.KL_gaussian(mu = self.Eps2_mean, sigsq= self.Eps2_sigmasq, logscale = False)

				self.latent_KL_loss = tf.reduce_mean(self.KL_Phi + self.KL_Eps1 + self.KL_Eps2)

				# Draw L samples of z1 and z2 based on the generative CCA model
				self.Z1, self.Z2, _, _, _ = self.CA.draw_2v_samples(Phi_mean=self.Phi_mean, Phi_sigmasq=self.Phi_sigmasq,
																	Eps1_mean=self.Eps1_mean, Eps1_sigmasq=self.Eps1_sigmasq,
																	Eps2_mean=self.Eps2_mean, Eps2_sigmasq=self.Eps2_sigmasq)

				# generator network for view 1
				self.xhat1_mean, self.xhat1_sigmasq = self.AE1.decoder(self.Z1, keepprob=self.keepprob, reuse=None)
				# self.xhat1_sigmasq = tf.exp(xhat1_log_sigmasq)

				# generator network for view 2
				self.xhat2_mean, self.xhat2_sigmasq = self.AE2.decoder(self.Z2, keepprob=self.keepprob, reuse=None)
				# self.xhat2_sigmasq = tf.exp(xhat2_log_sigmasq)

				# Compute negative log-likelihood (NLL) for both input views.
				self.NLL1 = self.AE1.NLL(tf.tile(x1, [args.n_samples, 1]), mu=self.xhat1_mean, log_sigsq= self.xhat1_sigmasq)
				self.NLL2 = self.AE2.NLL(tf.tile(x2, [args.n_samples, 1]), mu=self.xhat2_mean, log_sigsq= self.xhat2_sigmasq)

				# regularizer on the variables
				weigth_reg = self.hp.reg_var_wgt * tf.reduce_sum(
					tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
				# weigth_reg = self.hp.reg_var_wgt * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

				# ELBO is indeed negative elbo that we are going to minimize
				self.ELBO = self.NLL1 + self.NLL2 + self.beta_plh * self.latent_KL_loss + weigth_reg
				self.train_op = optimizer.minimize(self.ELBO, global_step=global_step, var_list= tf.compat.v1.trainable_variables())

			self.is_training = tf.Variable(True, trainable=False, name='is_training')
			self.set_isTraining = tf.assign(self.is_training, True, name='set_is_training')
			self.clear_isTraining = tf.assign(self.is_training, False, name='unset_is_training')

		######################
		# log summaries, summaries for tensorboard
		######################

		# Add ops to save and restore all the variables
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
		# saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
		self.save = lambda path: saver.save(sess, path,
											write_meta_graph=False if args.tf_save!='save_graph' else True)
		self.restore = lambda path: saver.restore(sess, path)

		def load_var(var, val):
			return var.load(val, self.sess)
		self.load_var = load_var

		def val(var):
			return sess.run(var)
		self.val = val

		# Initialize or restore the parameters
		sess.run(tf.global_variables_initializer())
		is_loaded = False
		if not args.restore_path in ['', 'none']:
			if args.restore_path == 'default': # todo
				ckpt_state = tf.train.get_checkpoint_state(args.logdir)
				if ckpt_state and ckpt_state.model_checkpoint_path:
					args._print("Loading the model from %s" % ckpt_state.model_checkpoint_path, 'log')
					self.restore(args.logdir + args.ckpt_filename) # self.restore(ckpt_state.model_checkpoint_path)
					is_loaded = True
					# args.init_CCA = 'infere'
			else:
				if os.path.isfile(args.restore_path + '.index'):
					self.restore(args.restore_path)
					args._print("Loading the model from %s" % args.restore_path, 'log')
					is_loaded = True
					# args.init_CCA = 'infere'
				elif args.mode == 'inference': # only stop in infrence mode otherwise don't load if no checkpoint exists and continue
					raise ValueError('checkpoint path is not true or not existant')

		# some extra definitions and variables
		self.global_step_val = lambda: sess.run(global_step)
		init_global_step0_op = tf.assign(global_step0, self.global_step_val(), name='init_global_step0')
		self.init_global_step0 = lambda: sess.run(init_global_step0_op)

	#### Training
	def train(self, beta, x1_value, x2_value):
		self.sess.run(self.set_isTraining)

		dict_tr = lambda x: merge_two_dicts(
			{self.keepprob: 1. - self.args.p_drop,
			 self.beta_plh: beta},
			x)

		feed_dict = {self.x1: x1_value, self.x2: x2_value}
		return self.sess.run([self.train_op, self.ELBO, self.NLL1, self.NLL2],
							 feed_dict=dict_tr(feed_dict))[1:]

	def train_epoch(self, epoch, data, ):
		t = self._t0  # time.time()
		global_step_val = self.global_step_val()

		beta = beta_fn(epoch, self.args)
		# self.args._print('beta = {:5.4f}'.format(beta),'log')
		if beta <= 0.:
			self.load_var(self.ae_train_only, True)
		else:
			self.load_var(self.ae_train_only, False)

		if beta >= 1. and (not self.completed_warmup):
			if self.args.to_reset_optimizer:
				self.reset_optimizer()
				self.args._print('------- Optimizer has been reset ---------', 'log')
			self.completed_warmup = True

		# ---------- train one epoch -----------
		elbo_tr, NLL1_tr, NLL2_tr = [], [], []
		for it in range(data.n_batches):

			_x1_ba, _x2_ba, _batch_ind, _ = data.train_iterator()
			# Run a training step.
			_obj, _nll1, _nll2 = self.train(beta, x1_value=_x1_ba, x2_value=_x2_ba)[0:3]
			elbo_tr.append(_obj)
			NLL1_tr.append(_nll1)
			NLL2_tr.append(_nll2)

			global_step_val = self.global_step_val()
			if (it % 200) == 0:
				print('  ------- step: %d, time = %ds, training OBJ=%g' %
					  (global_step_val, time.time() - t, elbo_tr[-1]))

		elbo_tr = np.mean(np.asarray(elbo_tr), axis=0)
		NLL1_tr = np.mean(np.asarray(NLL1_tr), axis=0)
		NLL2_tr = np.mean(np.asarray(NLL2_tr), axis=0)

		dtrain = time.time() - t

		if (beta == 0.) and (beta_fn(epoch + 1, self.args) == 0.):  # epoch < self.args.warmupDelay - 1
			self.args._print('>>> epoch:%d, step:%d, time=%ds, elbo=%g, NLL1=%g, NLL2=%g. lr =< %g'
						% (epoch, global_step_val, int(dtrain), elbo_tr, NLL1_tr, NLL2_tr, self.lr_val()), 'log')
			results_dict = dict(obj_tr=elbo_tr, NLL1_tr=NLL1_tr, NLL2_tr=NLL2_tr)
			# obj_tu=None, NLL1_tu=None, NLL2_tu=None)
			return results_dict

		dtrain = time.time() - t
		if epoch % 10 == 1:
			self.args._print("Epoch,  TIME| TRAIN: obj,  NLL1,  NLL2 || LR,  beta",
						'log')
		self.args._print("%05d: %5d|      %.4g   %.4g   %.4g|| %.4g  %.2g "
					% (epoch, dtrain, elbo_tr, NLL1_tr, NLL2_tr,  # elbo_tu, NLL1_tu, NLL2_tu,
					   self.lr_val(), beta), 'log')

		results_dict = dict(obj_tr=elbo_tr, NLL1_tr=NLL1_tr, NLL2_tr=NLL2_tr)
		# obj_tu=elbo_tu, NLL1_tu=NLL1_tu, NLL2_tu=NLL2_tu)

		return results_dict

	def encode_batched(self, output_list, x1_value, x2_value, bs=None):
		dict_te = lambda x: merge_two_dicts({self.keepprob: 1., self.beta_plh: 1.}, x)

		n = x1_value.shape[0]
		bs = self.args.batch_size_valid if bs is None else bs
		bs = n if bs == -1 else bs
		n_iter = int(np.ceil( float(n)/float(bs) ))
		n_out = len(output_list)

		out_val_list = [[] for i in range(n_out)]
		for i in range(n_iter):
			_from = i * bs
			_to   = min((i+1) * bs , n)
			if _to == _from: continue
			feed_dict = {self.x1: x1_value[_from:_to], self.x2: x2_value[_from:_to]}
			out_val = self.sess.run(output_list,
										  feed_dict=dict_te(feed_dict))

			for i in range(n_out):
				out_val_list[i].append(out_val[i])

		return [np.concatenate(out_val_list[i], axis=0) for i in range(n_out)]

	#### Testing
	def test(self, x1_value, x2_value, bs=None):
		self.sess.run(self.clear_isTraining)

		dict_te = lambda x: merge_two_dicts({self.keepprob: 1., self.beta_plh: 1.}, x)
		n = x1_value.shape[0]
		bs = self.args.batch_size_valid if bs is None else bs
		bs = n if bs == -1 else bs

		n_iter = n // bs
		Loss_, NLL1_, NLL2_ = [], [], []
		for i in range(n_iter):
			_from, _to = i * bs, (i+1) * bs
			feed_dict = {self.x1: x1_value[_from:_to], self.x2: x2_value[_from:_to]}
			loss_, nll1_, nll2_  = self.sess.run([self.ELBO, self.NLL1, self.NLL2],
												 feed_dict=dict_te(feed_dict))
			Loss_.append(loss_)
			NLL1_.append(nll1_)
			NLL2_.append(nll2_)
		return np.mean(np.asarray(Loss_), axis=0), np.mean(np.asarray(NLL1_), axis=0), np.mean(np.asarray(NLL2_), axis=0)


class model_CCA():
	def __init__(self, args, name='model_CCA'):
		self.name = name
		self.args = args
		self.hp = args.hp

		self.n_basis = args.hidden_dim_shrd
		self.hidden_dim1 = args.hidden_dim
		self.hidden_dim2 = args.hidden_dim
		self.n_samples = args.n_samples
		self.alpha = [1./args.ae_shrd.var_prior, 1./args.ae1.var_prior, 1./args.ae2.var_prior]

	def load_var(self, var, val):
		return var.load(val, self.sess)

	def pad2z1(self, x):
		dim_in = x.get_shape().as_list()[1]
		return tf.pad(x, paddings=tf.constant([[0, 0], [0, self.hidden_dim1 - dim_in]]), mode='CONSTANT')

	def pad2z2(self, x):
		dim_in = x.get_shape().as_list()[1]
		return tf.pad(x, paddings=tf.constant([[0, 0], [0, self.hidden_dim2 - dim_in]]), mode='CONSTANT')

	def fit_CCA(self, z1_mean, z1_sigmasq, z2_mean, z2_sigmasq, Phi_mean_in, p_corr):

		self.M = tf.sqrt(p_corr)
		self.W1 = self.pad2z1(tf.sqrt(z1_sigmasq)[:, 0:self.n_basis] * self.M)
		self.W2 = self.pad2z2(tf.sqrt(z2_sigmasq)[:, 0:self.n_basis] * self.M)

		Eps1_sigmasq = z1_sigmasq - tf.square(self.W1)
		Eps2_sigmasq = z2_sigmasq - tf.square(self.W2)

		a0, a1, a2 = self.alpha[0], self.alpha[1], self.alpha[2]
		Phi_mean_v12 = tf.reciprocal((a0 + a1 * tf.square(self.W1) + a2 * tf.square(self.W2))[:, 0:self.n_basis] + 1e-6) * \
				   (a1 * self.W1 * z1_mean + a2 * self.W2 * z2_mean)[:, 0:self.n_basis]
		# a0, a1, a2 = self.alpha[0], self.alpha[1], 0.
		Phi_mean_v1 = tf.reciprocal((a0 + a1 * tf.square(self.W1))[:, 0:self.n_basis] + 1e-6) * \
				   (a1 * self.W1 * z1_mean)[:, 0:self.n_basis]
		# a0, a1, a2 = self.alpha[0], 0, self.alpha[2]
		Phi_mean_v2 = tf.reciprocal((a0 + a2 * tf.square(self.W2))[:, 0:self.n_basis] + 1e-6) * \
				   (a2 * self.W2 * z2_mean)[:, 0:self.n_basis]
		if self.hp.shrd_est == 'nn':
			Phi_mean = Phi_mean_in
		elif self.hp.shrd_est == 'z12':
			Phi_mean = Phi_mean_v12
		elif self.hp.shrd_est == 'z1':
			Phi_mean = Phi_mean_v1
		elif self.hp.shrd_est == 'z2':
			Phi_mean = Phi_mean_v2
		else:
			raise ValueError('not a valid shrd_est type')
		Eps1_mean = z1_mean - self.W1 * self.pad2z1(Phi_mean)
		Eps2_mean = z2_mean - self.W2 * self.pad2z2(Phi_mean)

		Phi_sigmasq = tf.ones_like(Phi_mean) #tf.ones(shape=[1, self.n_basis])
		return Eps1_mean, Eps1_sigmasq, Eps2_mean, Eps2_sigmasq, Phi_mean, Phi_sigmasq, Phi_mean_v1, Phi_mean_v2

	def draw_2v_samples(self, Phi_mean, Phi_sigmasq, Eps1_mean, Eps1_sigmasq, Eps2_mean, Eps2_sigmasq, n_samples=None):
		if n_samples is None:
			n_samples = self.n_samples

		# Draw samples of phi.
		phishape = tf.multiply(tf.shape(Phi_mean), [n_samples, 1])
		Phi = tf.random_normal(phishape, 0, 1, dtype=self.args.dtype)
		Phi = tf.tile(Phi_mean, [n_samples, 1]) +\
			  tf.multiply(tf.tile( tf.sqrt(Phi_sigmasq), [n_samples, 1]), Phi) #todo simplify with broadcasting

		# Draw samples of z1.
		epsshape = tf.multiply(tf.shape(Eps1_mean), [n_samples, 1])
		Eps1 = tf.random_normal(epsshape, 0, 1, dtype=self.args.dtype)
		Eps1 = tf.tile(Eps1_mean, [n_samples, 1]) + \
			   tf.multiply(tf.tile( tf.sqrt(Eps1_sigmasq), [n_samples, 1]), Eps1) #todo simplify with broadcasting

		# Draw samples of z2.
		epsshape = tf.multiply(tf.shape(Eps2_mean), [n_samples, 1])
		Eps2 = tf.random_normal(epsshape, 0, 1, dtype=self.args.dtype)
		Eps2 = tf.tile(Eps2_mean, [n_samples, 1]) + \
			   tf.multiply(tf.tile( tf.sqrt(Eps2_sigmasq), [n_samples, 1]), Eps2) #todo simplify with broadcasting

		z1, z2 = self.gen(Phi, Eps1, Eps2, n_samples=n_samples)
		return z1, z2, Eps1, Eps2, Phi

	def gen(self, Phi, Eps1, Eps2, n_samples=1):
		z1 = tf.tile(self.W1, [n_samples, 1]) * self.pad2z1(Phi) + Eps1
		z2 = tf.tile(self.W2, [n_samples, 1]) * self.pad2z2(Phi) + Eps2
		return z1, z2


class model_VAE():
	def __init__(self, args, ae, name='model'):

		self.name = name
		self.args = args
		self.hp = args.hp
		self.loss_rec = ae.loss_rec
		self.var_rec = ae.var_rec
		self.n_samples = args.n_samples

		self.hidden_dim = ae.shape_enc[-1] #args.hidden_dim
		self.gate_enc = ae.gate_enc
		self.splits_enc = [self.hidden_dim if gt else 0 for gt in ae.postenc_gate]
		self.shape_enc = 1*ae.shape_enc
		self.shape_enc[-1] = sum(self.splits_enc)
		self.gate_postenc = [gate_dict[pst_gt] for pst_gt in ae.postenc_gate]

		self.shape_dec = 1*ae.shape_dec
		self.gate_dec = ae.gate_dec

		if self.loss_rec == 'WLS':
			self.shape_dec[-1] = 2 * self.shape_dec[-1]
		elif self.loss_rec == 'LS_isotropic':
			self.shape_dec[-1] = 1 + self.shape_dec[-1]
		else:
			self.shape_dec[-1] = 1 * self.shape_dec[-1]

		self.gate_dec_mean = gate_dict[ae.postdec_gate[0]]
		self.gate_dec_sigsq = gate_dict[ae.postdec_gate[1]]
		self.nll_logscale = True if self.gate_dec_sigsq == 'linear' else False

		self.var_prior = tf.constant(ae.var_prior, dtype=args.dtype) #tf.constant([ae.var_prior] * self.hidden_dim, dtype=args.dtype)

		self.init_net_F = [args.init_nets for i in range(len(self.gate_enc))]
		self.init_net_G = [args.init_nets for i in range(len(self.gate_dec))]

		self.regularizer = tf.keras.regularizers.l2(1.)
		self.regularizer_bias = tf.keras.regularizers.l2(1.) if args.hp.reg_bias else None
		# self.normalize = normalizaion(self.hp, name=name + 'norm')

	def NLL(self, x, mu, log_sigsq): #todo: check with my NLL (BPD)
		d_in = x.get_shape().as_list()[1]
		
		if self.loss_rec == 'CE_logit':
			# Cross entropy loss given logit .
			reconstr_loss = tf.reduce_sum(
				tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=mu), 1)
		elif self.loss_rec == 'WCE_logit':
			# Cross entropy loss given logit.
			# https://discuss.pytorch.org/t/weights-in-bcewithlogitsloss/27452/11
			# https://discuss.pytorch.org/t/how-to-apply-weighted-loss-to-a-binary-segmentation-problem/35317/5
			# reconstr_loss = (1./self.var_rec) * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=mu), 1)
			reconstr_loss = tf.reduce_sum(
				tf.nn.weighted_cross_entropy_with_logits(labels=x, logits=mu, pos_weight=(1./self.var_rec)), 1)
		elif self.loss_rec == 'CE':
			# Cross entropy loss.
			reconstr_loss = - tf.reduce_sum(
				x * tf.log(1e-6 + mu) + (1 - x) * tf.log(1e-6 + 1 - mu), 1)
		elif self.loss_rec == 'WCE':
			# Cross entropy loss.
			reconstr_loss = - (1./self.var_rec) * tf.reduce_sum(
				x * tf.log(1e-6 + mu) + (1 - x) * tf.log(1e-6 + 1 - mu), 1)
		elif self.loss_rec in ['WLS', 'LS_isotropic']:
			# wheighted Least squares loss, with learned var.
			if self.nll_logscale:
				reconstr_loss = 0.5 * tf.reduce_sum(tf.div(tf.square(x - mu), 1e-6 + tf.exp(log_sigsq)), 1) + \
								0.5 * tf.reduce_sum(log_sigsq, 1) + 0.5 * np.log(2 * np.pi) * d_in
			else:
				sigsq = log_sigsq
				reconstr_loss = 0.5 * tf.reduce_sum(tf.div(tf.square(x - mu), 1e-6 + sigsq), 1) + \
								0.5 * tf.reduce_sum(tf.log(sigsq), 1) + 0.5 * np.log(2 * np.pi) * d_in
		elif self.loss_rec == 'LS':
			# Least squares loss, with fixed std.
			reconstr_loss = 0.5 * tf.reduce_sum(tf.square(x - mu) / self.var_rec, 1) + \
							0.5 * np.log(2 * np.pi * self.var_rec) * d_in
		else:
			raise ValueError('not a valid reconstruction loss')

		# Average over the minibatch.
		loss = tf.reduce_mean(reconstr_loss)
		return loss

	def KL_gaussian(self, mu, sigsq, logscale=False):
		if logscale:
			log_sigsq = sigsq
			return - 0.5 * tf.reduce_sum(1 + log_sigsq - tf.log(self.var_prior)
										 - tf.square(mu)/self.var_prior - tf.exp(log_sigsq)/self.var_prior, 1)
		else:
			return - 0.5 * tf.reduce_sum(1 + tf.log(sigsq) - tf.log(self.var_prior)
										 - tf.square(mu)/self.var_prior - sigsq/self.var_prior, 1)

	def encoder(self, x, keepprob, reuse=True): #todo
		with tf.variable_scope(self.name, reuse=reuse):
			y = encoder_net(x, self.shape_enc, self.gate_enc, 'encoder', self.init_net_F,
							regularizer=self.regularizer, regularizer_bias=self.regularizer_bias, keepprob=keepprob)
			y_splits = tf.split(y, self.splits_enc, axis=-1)
			return [self.gate_postenc[i](y_splt_) for i, y_splt_, in enumerate(y_splits)]
			# mean, sigmasq_ = tf.split(y, 2, axis=-1)
			# return self.gate_enc_mean(mean), self.gate_enc_sigsq(sigmasq_)


	def decoder(self, y, keepprob, reuse=True): #todo
		with tf.variable_scope(self.name, reuse=reuse):
			x = encoder_net(y, self.shape_dec, self.gate_dec, 'decoder', self.init_net_G,
							regularizer=self.regularizer, regularizer_bias=self.regularizer_bias, keepprob=keepprob)
			if self.loss_rec == 'WLS':
				mean, log_sigmasq = tf.split(x, 2, axis=-1)
				return self.gate_dec_mean(mean), self.gate_dec_sigsq(log_sigmasq)
			elif self.loss_rec == 'LS_isotropic':
				d_out_ = x.get_shape().as_list()[-1] - 1
				mean, log_sigmasq = tf.split(x, [d_out_, 1], axis=-1)
				log_sigmasq = tf.tile(log_sigmasq,
									  (len(mean.get_shape().as_list()) - 1) * [1] + [d_out_])
				return self.gate_dec_mean(mean), self.gate_dec_sigsq(log_sigmasq)
			else:
				return self.gate_dec_mean(x), None


def one_layer_ff(x, dim_in, dim_out, activation, name, keepprob, initial_wgt = None,
				 regularizer=tf.keras.regularizers.l2(1.), regularizer_bias=tf.keras.regularizers.l2(1.)):
	with tf.variable_scope(name):
		x = tf.nn.dropout(x, keepprob)
		#define variable
		if initial_wgt in [None, 'none', 'None']:
			if activation == 'relu':
				# initial_W = tf.initializers.truncated_normal( stddev=1.0/np.sqrt(dim_in+1), seed=SEED)
				initial_W = tf.keras.initializers.he_normal(seed=SEED_op) # (Xavier normal initializer) samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out))
				initial_b = tf.initializers.constant(0.1, dtype=tf.float32)
			else:
				initial_W = tf.keras.initializers.glorot_uniform(seed=SEED_op)
				initial_b = tf.keras.initializers.glorot_uniform(seed=SEED_op)
				# initial_W = tf.initializers.random_normal( stddev=1.0/np.sqrt(dim_in+1), seed=SEED, dtype=tf.float32)
				# initial_b = tf.initializers.random_normal(stddev=1.0/np.sqrt(dim_in+1), seed=SEED, dtype=tf.float32)
		elif initial_wgt == 'unif':
			initial_W = tf.random_uniform_initializer(-0.05, 0.05)
			initial_b = tf.random_uniform_initializer(-0.05, 0.05)
		else:
			initial_W = tf.initializers.constant(initial_wgt[:-1,:], dtype=tf.float32)
			initial_b = tf.initializers.constant(initial_wgt[-1,:], dtype=tf.float32)
		W = tf.get_variable("W", shape = [dim_in, dim_out],
							dtype=tf.float32, initializer=initial_W, regularizer=regularizer)
		b = tf.get_variable("b", shape=[dim_out],
							dtype=tf.float32, initializer=initial_b, regularizer=regularizer_bias)

		# if activation == 'relu':
		# 	activation_fcn = tf.nn.relu
		# elif activation == 'sigmoid':
		# 	activation_fcn = tf.sigmoid
		# elif activation == 'tanh':
		# 	activation_fcn = tf.tanh
		# elif activation == 'linear':
		# 	activation_fcn = tf.identity
		# elif activation == 'sigm_linear':
		# 	raise ValueError('half sigm_linear not implemented yet')
		activation_fcn = gate_dict[activation]
		z = tf.nn.bias_add(tf.matmul(x,W), b)
		y = activation_fcn(z)
	return y, W, b

def encoder_net(x, enc_shape, activations, name, initial_wgts, regularizer, regularizer_bias, keepprob):
	with tf.variable_scope(name):
		h = x
		for l in range(len(enc_shape)-1):
			init_wgt = None if initial_wgts == None else initial_wgts[l]
			h, _, _ = one_layer_ff(h, enc_shape[l], enc_shape[l+1], activations[l],
								   "ff%d" %l, keepprob, init_wgt,
								   regularizer=regularizer, regularizer_bias=regularizer_bias)

	return h

