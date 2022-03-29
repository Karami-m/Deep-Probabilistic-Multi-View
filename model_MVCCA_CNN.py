from __future__ import division, print_function
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
			self.AE = []
			for i in range(self.modalities):
				self.AE += [model_VAE(args, args.ae_list[i], name='VAE%d' % i)]

			self.CA = model_MCCA(args, name ='MCCA')

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
				self.x, self.z_mean, self.z_sigmasq, ae_raw_outs = [], [], [], []
				for i in range(self.modalities):
					x_ = tf.placeholder(args.inp_dtype, args.inp_shapes[i], name='x%d'%i)

					# inference model all views
					z_mean_, z_sigmasq_, ae_out_ = self.AE[i].encoder(x_, keepprob=self.keepprob, reuse=None)
					# self.z1_sigmasq = tf.exp(self.z1_log_sigmasq)
					self.x += [x_]
					self.z_mean, self.z_sigmasq = self.z_mean+[z_mean_], self.z_sigmasq+[z_sigmasq_]
					ae_raw_outs += [ae_out_]

				# inference model for shared factor
				if self.hp.shrd_on_all == 'none':
					x_in_shrd = self.x[0]
				elif self.hp.shrd_on_all in ['x1', 'x2', 'x3', 'x4', 'x5']:
					x_in_shrd = self.x[int(self.hp.shrd_on_all[1])-1]
				elif self.hp.shrd_on_all in ['early', 'xall']:
					x_in_shrd = tf.concat(self.x, axis=-1)
				elif self.hp.shrd_on_all == 'late':
					x_in_shrd = tf.concat(ae_raw_outs, axis=-1)
					if not(self.hp.shrd_est in ['z', 'nn']):
						ae_raw_outs_temp = [ae_raw_outs[i] for i in range(self.modalities) if (str(i + 1) in self.hp.shrd_est)]
						x_in_shrd = tf.concat(ae_raw_outs_temp, axis=-1)
					x_in_shrd = tf.expand_dims(tf.expand_dims(x_in_shrd, axis=1),axis=1)


				Phi_mean_, self.p_corr, _ = self.AE0.encoder(x_in_shrd, keepprob=self.keepprob, reuse=None)
				# self.p_corr = clipped_sigmoid(p_corr_)

				self.Eps_mean, self.Eps_sigmasq, self.Phi_mean, self.Phi_sigmasq, self.Phi_mean_v1= \
					self.CA.fit_MCCA(self.z_mean, self.z_sigmasq, Phi_mean_, self.p_corr)

				# Calculate KL divergence for shared variable Phi and all Eps's.
				self.KL_Phi = self.AE0.KL_gaussian(mu = self.Phi_mean, sigsq= self.Phi_sigmasq, logscale = False)
				self.KL_Eps = []
				for i in range(self.modalities):
					self.KL_Eps += [self.AE[i].KL_gaussian(mu=self.Eps_mean[i], sigsq=self.Eps_sigmasq[i], logscale=False)]

				self.latent_KL_loss = tf.reduce_mean(self.KL_Phi + tf.add_n(self.KL_Eps))

				# Draw L samples of views based on the generative CCA model
				self.Z, _, _ = self.CA.draw_samples(Phi_mean=self.Phi_mean, Phi_sigmasq=self.Phi_sigmasq,
													Eps_mean=self.Eps_mean, Eps_sigmasq=self.Eps_sigmasq)

				# generator network for all views
				self.xhat_mean, self.xhat_sigmasq, self.NLL = [], [], []
				for i in range(self.modalities):
					xhat_mean_, xhat_sigmasq_ = self.AE[i].decoder(self.Z[i], keepprob=self.keepprob, reuse=None)
					# self.xhat1_sigmasq = tf.exp(xhat1_log_sigmasq)
					self.xhat_mean, self.xhat_sigmasq = self.xhat_mean+[xhat_mean_], self.xhat_sigmasq+[xhat_sigmasq_]

					# Compute negative log-likelihood (NLL) for both input views.
					NLL_ = self.AE[i].NLL(tf.tile(self.x[i], [args.n_samples] + [1]*self.AE[i].image_rank),
										  mu=self.xhat_mean[i], log_sigsq=self.xhat_sigmasq[i])
					self.NLL += [NLL_]

				# regularizer on the variables
				weigth_reg = self.hp.reg_var_wgt * tf.reduce_sum(
					tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
				# weigth_reg = self.hp.reg_var_wgt * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

				# ELBO is indeed negative elbo that we are going to minimize
				self.ELBO = tf.add_n(self.NLL) + self.beta_plh * self.latent_KL_loss + weigth_reg

				if args.optimizer in ['AdamaxOptimizer']:
					# todo: not working
					with tf.GradientTape() as tape:
						loss_ = lambda: self.ELBO
					self.train_op = optimizer.minimize(loss_, var_list= tf.compat.v1.trainable_variables())
					global_step = optimizer.iterations
				else:
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
	def train(self, beta, x_values):
		self.sess.run(self.set_isTraining)

		dict_tr = lambda x: merge_two_dicts(
			{self.keepprob: 1. - self.args.p_drop,
			 self.beta_plh: beta},
			x)

		feed_dict = {self.x[i]: x_values[i] for i in range(len(self.x))}
		return self.sess.run([self.train_op, self.ELBO]+self.NLL, feed_dict=dict_tr(feed_dict))[1:]

	def train_epoch(self, epoch, data ):
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
		elbo_tr = []
		NLL_tr = [[] for i in range(self.args.modalities)]
		for it in range(data.n_batches):

			batch_ = data.train_iterator()
			_x_ba, _batch_ind = batch_[0:self.args.modalities], batch_[self.args.modalities]
			# Run a training step.
			out_ba = self.train(beta, x_values=_x_ba)
			elbo_tr.append(out_ba[0])
			for i in range(self.args.modalities):
				NLL_tr[i].append(out_ba[i + 1])

			global_step_val = self.global_step_val()
			if (it % 200) == 0:
				print('  ------- step: %d, time = %ds, training OBJ=%g' %
					  (global_step_val, time.time() - t, elbo_tr[-1]))

		elbo_tr = np.mean(np.asarray(elbo_tr), axis=0)
		NLL_tr = [np.mean(np.asarray(nll_tr), axis=0) for nll_tr in NLL_tr]

		dtrain = time.time() - t

		str_NLL = ""
		str_NLL = str_NLL.join([',  NLL%d ' % i for i in range(1, self.args.modalities + 1)])
		if epoch % 10 == 1:
			self.args._print("EPOCH, TIME| TRAIN: obj%s || LR,  beta" % str_NLL, 'log')
		str_ = "%05d: %5d|      %.4g" + "   %.4g" * self.args.modalities + "|| %.4g  %.2g "
		self.args._print(str_ % ((epoch, dtrain, elbo_tr) + tuple(NLL_tr) + (self.lr_val(), beta)), 'log')

		results_dict = dict([['obj_tr', elbo_tr]] + [['NLL%d_tr' % i, NLL_tr[i]] for i in range(self.args.modalities)])
		return results_dict

	def run_batched(self, output_list, x_values, bs=None):
		dict_te = lambda x: merge_two_dicts({self.keepprob: 1., self.beta_plh: 1.}, x)

		n = x_values[0].shape[0]
		bs = self.args.batch_size_valid if bs is None else bs
		bs = n if bs == -1 else bs
		n_iter = int(np.ceil( float(n)/float(bs) ))
		n_out = len(output_list)

		out_val_list = [[] for i in range(n_out)]
		for i in range(n_iter):
			_from = i * bs
			_to   = min((i+1) * bs , n)
			if _to == _from: continue
			feed_dict = {self.x[i]: x_values[i][_from:_to] for i in range(len(self.x))}
			out_val = self.sess.run(output_list, feed_dict=dict_te(feed_dict))

			for i in range(n_out):
				# to check whether out_val_ is np.array and convert it if not so.
				out_val_ = np.array(out_val[i])
				out_val_= np.expand_dims(out_val_,axis=0) if out_val_.ndim==0 else out_val_
				out_val_list[i].append(out_val_)

		return [np.concatenate(out_val_list[i], axis=0) for i in range(n_out)]

	#### Testing
	def test(self, x_values, bs=None):
		self.sess.run(self.clear_isTraining)

		outputs_tf = [self.ELBO] + self.NLL
		outputs_val = self.run_batched(output_list= outputs_tf, x_values=x_values, bs=bs)
		return [np.mean(out_val, axis=0) for out_val in outputs_val]

class model_MCCA():
	def __init__(self, args, name='model_CCA'): # todo
		self.name = name
		self.args = args
		self.hp = args.hp
		self.modalities = args.modalities

		self.n_basis = args.hidden_dim_shrd
		self.hidden_dims = [args.hidden_dim for i in range(self.modalities)]
		self.n_samples = args.n_samples
		self.alpha0 = 1./args.ae_shrd.var_prior
		if  self.hp.shrd_est in ['z', 'nn', 'z1']:
			self.a = [1./args.ae_list[i].var_prior for i in range(self.modalities)]
		elif self.hp.shrd_est in ['z234', 'z345', 'z2345', 'z2', 'z3', 'z4', 'z5' ]:
			self.a = [1. / args.ae_list[i].var_prior if (str(i+1) in self.hp.shrd_est) else 0.
					  for i in range(self.modalities) ]


	def load_var(self, var, val):
		return var.load(val, self.sess)

	def pad2z(self, x, view_i):
		dim_in = x.get_shape().as_list()[1]
		return tf.pad(x, paddings=tf.constant([[0, 0], [0, self.hidden_dims[view_i] - dim_in]]), mode='CONSTANT')

	def fit_MCCA(self, z_mean, z_sigmasq, Phi_mean_in, p_corr): # todo

		self.M = tf.sqrt(p_corr)
		self.W, Eps_sigmasq = [], []
		T_num, T_den = 0., 0.
		for i in range(self.modalities):
			self.W.append( self.pad2z(tf.sqrt(z_sigmasq[i])[:, 0:self.n_basis] * self.M, i) )
			Eps_sigmasq.append(z_sigmasq[i] - tf.square(self.W[i]))

			T_num += self.a[i] * self.W[i] * z_mean[i] if self.a[i] != 0. else 0.
			T_den += self.a[i] * tf.square(self.W[i]) if self.a[i] != 0. else 0.
		Phi_mean = tf.reciprocal(self.alpha0 + T_den[:, 0:self.n_basis] + 1e-6) * T_num[:, 0:self.n_basis]

		T_num0 = self.a[0] * self.W[0] * z_mean[0]
		T_den0 = self.a[0] * tf.square(self.W[0])
		Phi_mean_v1 = tf.reciprocal(self.alpha0 + T_den0[:, 0:self.n_basis] + 1e-6) * T_num0[:, 0:self.n_basis]

		if self.hp.shrd_est == 'nn':
			Phi_mean = Phi_mean_in
		elif self.hp.shrd_est == 'z1':
			Phi_mean = Phi_mean_v1

		# Phi_mean_v1 = tf.reciprocal((self.alpha0 + self.a[0] * tf.square(self.W[0]))[:, 0:self.n_basis] + 1e-6) * \
		# 		   (self.a[0] * self.W[0] * z_mean[0])[:, 0:self.n_basis]

		Eps_mean = []
		for i in range(self.modalities):
			Eps_mean.append( z_mean[i] - self.W[i] * self.pad2z(Phi_mean, i) )

		Phi_sigmasq = tf.ones_like(Phi_mean) #tf.ones(shape=[1, self.n_basis])
		return Eps_mean, Eps_sigmasq, Phi_mean, Phi_sigmasq, Phi_mean_v1

	def draw_samples(self, Phi_mean, Phi_sigmasq, Eps_mean, Eps_sigmasq, n_samples=None):
		if n_samples is None:
			n_samples = self.n_samples

		# Draw samples of phi.
		phishape = tf.multiply(tf.shape(Phi_mean), [n_samples, 1])
		Phi = tf.random_normal(phishape, 0, 1, dtype=self.args.dtype)
		Phi = tf.tile(Phi_mean, [n_samples, 1]) +\
			  tf.multiply(tf.tile( tf.sqrt(Phi_sigmasq), [n_samples, 1]), Phi) #todo simplify with broadcasting

		# Draw samples of each view.
		Eps = []
		for i in range(self.modalities):
			epsshape = tf.multiply(tf.shape(Eps_mean[i]), [n_samples, 1])
			Eps_ = tf.random_normal(epsshape, 0, 1, dtype=self.args.dtype)
			Eps_ = tf.tile(Eps_mean[i], [n_samples, 1]) + \
				   tf.multiply(tf.tile( tf.sqrt(Eps_sigmasq[i]), [n_samples, 1]), Eps_) #todo simplify with broadcasting
			Eps.append(Eps_)

		z = self.gen(Phi, Eps, n_samples=n_samples)
		return z, Eps, Phi

	def gen(self, Phi, Eps, n_samples=1):
		z = []
		for i in range(self.modalities):
			z.append( tf.tile(self.W[i], [n_samples, 1]) * self.pad2z(Phi, i) + Eps[i] )
		return z


class model_VAE():
	def __init__(self, args, ae, name='model'):

		self.name = name
		self.args = args
		self.hp = args.hp
		self.loss_rec = ae.loss_rec
		self.var_rec = ae.var_rec
		self.n_samples = args.n_samples

		self.NN_type = 'CNN'
		self.encoder = self.encoder_CNN
		self.decoder = self.decoder_CNN
		if hasattr(args, 'NN_type'):
			if args.NN_type == 'FF':
				self.NN_type = 'FF'
				self.encoder = self.encoder_FF
				self.decoder = self.decoder_FF


		# self.hidden_dim = ae.shape_enc[-1] #args.hidden_dim
		# self.enc_gates = ae.enc_gates
		self.enc_spec = 1*ae.enc_spec
		self.postenc_gate = [gate_dict[pst_gt] for pst_gt in ae.postenc_gate]
		self.postenc_dims = ae.postenc_dims

		self.predec_gate = 'linear' #['relu']
		self.dec_spec = 1*ae.dec_spec
		if self.loss_rec == 'WLS':
			self.dec_spec[0]['dim_out'] = 2 * self.dec_spec[0]['dim_out']
		elif self.loss_rec == 'LS_isotropic':
			self.dec_spec[0]['dim_out'] = 1 + self.dec_spec[0]['dim_out']
		else:
			self.dec_spec[0]['dim_out'] = 1 * self.dec_spec[0]['dim_out']
		# self.dec_gates = ae.dec_gates

		self.gate_dec_mean = gate_dict[ae.postdec_gate[0]]
		self.gate_dec_sigsq = gate_dict[ae.postdec_gate[1]]
		self.nll_logscale = True if self.gate_dec_sigsq == 'linear' else False

		self.var_prior = tf.constant(ae.var_prior, dtype=args.dtype) #tf.constant([ae.var_prior] * self.hidden_dim, dtype=args.dtype)
		self.regularizer = tf.keras.regularizers.l2(1.)

		if args.init_nets in [None, 'none', 'None']:
			# self.initial_W = tf.keras.initializers.he_normal(seed=SEED_op) # (Xavier normal initializer) samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out))
			self.initial_W = tf.keras.initializers.glorot_uniform(seed=SEED_op)
		elif args.init_nets == 'unif':
			self.initial_W = tf.random_uniform_initializer(-0.05, 0.05)
		# self.normalize = normalizaion(self.hp, name=name + 'norm')


	def NLL(self, x, mu, log_sigsq):
		d_in = np.prod(np.array(x.get_shape().as_list()[1:]))
		sum_axis = list(range(1, (self.image_rank+1)))
		
		if self.loss_rec == 'CE':
			# Cross entropy loss.
			reconstr_loss = - tf.reduce_sum(
				x * tf.log(1e-6 + mu) + (1 - x) * tf.log(1e-6 + 1 - mu), sum_axis)
		elif self.loss_rec == 'WCE':
			# Cross entropy loss.
			reconstr_loss = - (1./self.var_rec) *tf.reduce_sum(
				x * tf.log(1e-6 + mu) + (1 - x) * tf.log(1e-6 + 1 - mu), sum_axis)
		elif self.loss_rec in ['WLS', 'LS_isotropic']:
			# wheighted Least squares loss, with learned var.
			if self.nll_logscale:
				reconstr_loss = 0.5 * tf.reduce_sum(tf.div(tf.square(x - mu), 1e-6 + tf.exp(log_sigsq)), sum_axis) + \
								0.5 * tf.reduce_sum(log_sigsq, sum_axis) + 0.5 * np.log(2 * np.pi) * d_in
			else:
				sigsq = log_sigsq
				reconstr_loss = 0.5 * tf.reduce_sum(tf.div(tf.square(x - mu), 1e-6 + sigsq), sum_axis) + \
								0.5 * tf.reduce_sum(tf.log(sigsq), sum_axis) + 0.5 * np.log(2 * np.pi) * d_in
		elif self.loss_rec == 'LS':
			# Least squares loss, with fixed std.
			reconstr_loss = 0.5 * tf.reduce_sum(tf.square(x - mu) / self.var_rec, sum_axis) + \
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


	def encoder_CNN(self, x, keepprob, reuse=True):
		with tf.variable_scope(self.name, reuse=reuse):
			self.image_dims = x._shape_as_list()[1:]
			self.image_rank = len(self.image_dims)
			with tf.variable_scope('encoder'):
				y = x
				self.latent_shapes = []
				for l, enc_spec in enumerate(self.enc_spec):
					self.latent_shapes += [[-1] + y._shape_as_list()[1:]] #[tf.shape(y)]  #
					y = conv2d(y,
							   dim_out=enc_spec['dim_out'],
							   filter_size=enc_spec['filter_size'],
							   stride=enc_spec['stride'],
							   pad=enc_spec['pad'],
							   activation=enc_spec['gate'],
							   name="cnn%d" % l,
							   keepprob=keepprob,
							   initial_wgt=self.initial_W,
							   regularizer=self.regularizer)

				self.latent_shapes += [[-1] + y._shape_as_list()[1:]] # [tf.shape(y)] #
				# self.shape_latent2D = y.get_shape().as_list()[1:]
				y = tf.keras.layers.Flatten()(y)
				# y = tf.contrib.layers.flatten(y)

				# to perform post encoder fully connected filters
				dim_in = y.get_shape().as_list()[-1]
				outputs = []
				for i in range(len(self.postenc_dims)):
					W = tf.get_variable("W_postenc%d" % i, shape=[dim_in, self.postenc_dims[i]],
										dtype=tf.float32, initializer=self.initial_W, regularizer=self.regularizer)
					b = tf.get_variable("b_postenc%d" % i, shape=[self.postenc_dims[i]],
										dtype=tf.float32, initializer=tf.zeros_initializer(), regularizer=self.regularizer)
					outputs += [self.postenc_gate[i](tf.nn.bias_add(tf.matmul(y, W), b)) if self.postenc_dims[i] > 0
								else None]
			return outputs+[y]

	def decoder_CNN(self, y, keepprob, reuse=True):
		with tf.variable_scope(self.name, reuse=reuse):
			with tf.variable_scope('decoder'):
				dim_in_ = y.get_shape().as_list()[-1]
				if self.hp.predec == 'fc':
					dim_out_ = np.prod(np.array(self.latent_shapes[-1][1:]))
					W = tf.get_variable("W_predec", shape=[dim_in_, dim_out_],
										dtype=tf.float32, initializer=self.initial_W, regularizer=self.regularizer)
					b = tf.get_variable("b_predec", shape=[dim_out_],
										dtype=tf.float32, initializer=tf.zeros_initializer(), regularizer=self.regularizer)
					y = gate_dict[self.predec_gate](tf.nn.bias_add(tf.matmul(y, W), b))
					y = tf.reshape(y, shape=self.latent_shapes[-1]) #shape=[-1]+self.shape_latent2D)

				elif self.hp.predec in ['cnnT', 'convT']:
					y = tf.reshape(y, [-1, 1, 1, dim_in_])
					y = conv2d_transpose(y,
										 dim_out=self.latent_shapes[-1][-1],
										 output_shape=self.latent_shapes[-1],
										 filter_size=self.latent_shapes[-1][1:3],
										 stride=[1, 1],
										 pad='valid',
										 activation=self.predec_gate,
										 name="cnn_trans_pre",
										 keepprob=keepprob,
										 initial_wgt=self.initial_W,
										 regularizer=self.regularizer)
				else:
					raise ValueError('Not a valid predec type')

				for l in reversed(range(len(self.dec_spec))):
					dec_spec = self.dec_spec[l]
					# output_shape_ = self.latent_shapes[l] if self.loss_rec != 'WLS' else \
					# 	tf.stack([self.latent_shapes[l][0], self.latent_shapes[l][1], self.latent_shapes[l][2], dim_out_])
					y = conv2d_transpose(y,
										 dim_out= dec_spec['dim_out'],
										 output_shape = self.latent_shapes[l],
										 filter_size=dec_spec['filter_size'],
										 stride=dec_spec['stride'],
										 pad=dec_spec['pad'],
										 activation=dec_spec['gate'],
										 name="cnn_trans%d" % l,
										 keepprob=keepprob,
										 initial_wgt=self.initial_W,
										 regularizer=self.regularizer)

				if self.loss_rec == 'WLS':
					mean, log_sigmasq = tf.split(y, 2, axis=-1)
					return self.gate_dec_mean(mean), self.gate_dec_sigsq(log_sigmasq)
				else:
					return self.gate_dec_mean(y), None

	def encoder_FF(self, x, keepprob, reuse=True):
		with tf.variable_scope(self.name, reuse=reuse):
			self.image_dims = x._shape_as_list()[1:]
			self.image_rank = len(self.image_dims)
			with tf.variable_scope('encoder'):
				y = x
				self.latent_shapes = []
				for l, enc_spec in enumerate(self.enc_spec):
					self.latent_shapes += [[-1] + y._shape_as_list()[1:]]  # [tf.shape(y)]  #
					y = one_layer_ff(y,
									 dim_out=enc_spec['dim_out'],
									 activation=gate_dict[enc_spec['gate']],
									 name="ff%d" % l,
									 keepprob=keepprob,
									 initial_W=self.initial_W,
									 initial_b=self.initial_W,
									 regularizer=self.regularizer)

				self.latent_shapes += [[-1] + y._shape_as_list()[1:]]  # [tf.shape(y)] #

				# to perform post encoder fully connected filters
				dim_in = y.get_shape().as_list()[-1]
				outputs = []
				for i in range(len(self.postenc_dims)):
					outputs += [one_layer_ff(y,
											 dim_out=self.postenc_dims[i],
											 activation=self.postenc_gate[i],
											 name="postenc%d"%i,
											 keepprob=keepprob,
											 initial_W=self.initial_W,
											 initial_b=self.initial_W, #tf.zeros_initializer(),
											 regularizer=self.regularizer)
								if self.postenc_dims[i] > 0 else None]

		return outputs + [y]

	def decoder_FF(self, y, keepprob, reuse=True):
		with tf.variable_scope(self.name, reuse=reuse):
			with tf.variable_scope('decoder'):
				for l in reversed(range(len(self.dec_spec))):
					dec_spec = self.dec_spec[l]
					y = one_layer_ff(y,
									 dim_out=dec_spec['dim_out'],
									 activation=gate_dict[dec_spec['gate']],
									 name="ff%d" % l,
									 keepprob=keepprob,
									 initial_W=self.initial_W,
									 initial_b=self.initial_W,
									 regularizer=self.regularizer)

				if self.loss_rec == 'WLS':
					mean, log_sigmasq = tf.split(y, 2, axis=-1)
					return self.gate_dec_mean(mean), self.gate_dec_sigsq(log_sigmasq)
				elif self.loss_rec == 'LS_isotropic':
					d_out_ = y.get_shape().as_list()[-1] - 1
					mean, log_sigmasq = tf.split(y, [d_out_, 1], axis=-1)
					log_sigmasq = tf.tile(log_sigmasq,
										  (len(mean.get_shape().as_list())-1)* [1] +[d_out_])
					return self.gate_dec_mean(mean), self.gate_dec_sigsq(log_sigmasq)
				else:
					return self.gate_dec_mean(y), None


def conv2d(x, dim_out, activation, name, keepprob,
		   filter_size=[3, 3], stride=[1, 1], pad="SAME",
		   initial_wgt = tf.random_uniform_initializer(-0.05, 0.05),
		   regularizer=tf.keras.regularizers.l2(1.) ): # todo
	with tf.variable_scope(name):
		x = tf.nn.dropout(x, keepprob)

		z = tf.keras.layers.Convolution2D(
			filters=dim_out, kernel_size= filter_size, strides= stride, padding=pad,
			kernel_initializer = initial_wgt, kernel_regularizer= regularizer,
			use_bias= True, bias_initializer=tf.zeros_initializer(), bias_regularizer=regularizer
		)(x)
		activation_fcn = gate_dict[activation]
		y = activation_fcn(z)
	return y


def conv2d_transpose(x, dim_out, output_shape, activation, name, keepprob,
					 filter_size=[3, 3], stride=[1, 1], pad="SAME",
					 initial_wgt = tf.random_uniform_initializer(-0.05, 0.05),
					 regularizer=tf.keras.regularizers.l2(1.) ): # todo
	with tf.variable_scope(name):
		x = tf.nn.dropout(x, keepprob)

		z = tf.keras.layers.Conv2DTranspose(
			filters=dim_out, kernel_size= filter_size, strides= stride, padding=pad,
			kernel_initializer = initial_wgt, kernel_regularizer= regularizer,
			use_bias= True, bias_initializer=tf.zeros_initializer(), bias_regularizer=regularizer
		)(x)
		activation_fcn = gate_dict[activation]
		y = activation_fcn(z)
	return y

def conv2d_old(x, dim_out, activation, name, keepprob,
		   filter_size=[3, 3], stride=[1, 1], pad="SAME",
		   initial_wgt = tf.random_uniform_initializer(-0.05, 0.05),
		   regularizer=tf.keras.regularizers.l2(1.) ): # todo
	with tf.variable_scope(name):
		x = tf.nn.dropout(x, keepprob)

		dim_in = int(x.get_shape()[3])
		stride_shape = [1] + stride + [1]
		filter_shape = filter_size + [dim_in, dim_out]
		#define variable
		W = tf.get_variable("W", shape = filter_shape,
							dtype=tf.float32, initializer=initial_wgt, regularizer=regularizer)
		b = tf.get_variable("b", shape=[dim_out],
							dtype=tf.float32, initializer=tf.zeros_initializer(), regularizer=regularizer)

		z = tf.nn.bias_add(tf.nn.conv2d(x, W, strides=stride_shape, padding=pad, data_format='NHWC'), b)
		activation_fcn = gate_dict[activation]
		y = activation_fcn(z)
	return y

# def encoder_CNN(x, enc_spec, activations, name, initial_wgt, regularizer, keepprob):
# 	with tf.variable_scope(name):
# 		h = x
# 		for l in range(len(enc_spec)):
# 			h = conv2d(h,
# 					   dim_out=enc_spec['dim_out'],
# 					   filter_size=enc_spec['filter_size'],
# 					   stride=enc_spec['stride'],
# 					   pad=enc_spec['pad'],
# 					   activation=activations[l],
# 					   name="cnn%d" %l,
# 					   keepprob=keepprob,
# 					   initial_wgt=initial_wgt,
# 					   regularizer=regularizer)
# 	return h

def one_layer_ff(x, dim_out, activation, name, keepprob, initial_W, initial_b,
				 regularizer=tf.keras.regularizers.l2(1.)):
	with tf.variable_scope(name):
		dim_in = x.get_shape().as_list()[-1]
		x = tf.nn.dropout(x, keepprob)
		#define variable
		W = tf.get_variable("W", shape = [dim_in, dim_out],
							dtype=tf.float32, initializer=initial_W, regularizer=regularizer)
		b = tf.get_variable("b", shape=[dim_out],
							dtype=tf.float32, initializer=initial_b, regularizer=regularizer)
		z = tf.nn.bias_add(tf.matmul(x,W), b)
		# activation_fcn = gate_dict[activation]
		y = activation(z)
	return y

