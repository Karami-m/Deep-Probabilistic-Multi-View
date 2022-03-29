"""
The downstream tasks that we used in our experiments.
To provide a fair comparison, the downstream tasks that were used for 2view noisy MNIST datasets are based
on the codes of the paper: Weiran Wang, et al. "Deep Variational Canonical Correlation Analysis",
which can be found (http://ttic.uchicago.edu/~wwang5/)
"""

from __future__ import division, print_function
import os, time
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn import cluster, metrics
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from munkres import Munkres
import tensorflow as tf
import tensorflow_probability as tfp


debug_mode = (os.environ['debugmode'] == 'True')


def linear_svm_classify(x_tr, label_tr, x_tu, label_tu, x_te, label_te,
						standardization=False, alg='SVC', cache_size=1000, C_list=[0.1, 1.0, 10.0]):
	"""TIPS on Practical use of SVM:
	https://scikit-learn.org/stable/modules/svm.html
	"""
	global debug_mode
	if debug_mode:
		return 1., 1.

	if standardization:
		scaler = preprocessing.StandardScaler().fit(x_tr)
		x_tr = scaler.transform(x_tr)
		x_tu = scaler.transform(x_tu)
		x_te = scaler.transform(x_te)

	print("\n Running linear SVM!")
	best_error_tu = 1.0 + 1e-4
	for c in C_list: #[0.02, 0.1, 1.0, 10.0]
		if alg == 'SVC':
			lin_clf = svm.SVC(C=c, kernel="linear", cache_size=cache_size)
		elif alg == 'LinearSVC':
			lin_clf = svm.LinearSVC(C=c)
		# training
		lin_clf.fit(x_tr, label_tr)

		pred_tr = lin_clf.predict(x_tr)
		error_tr = np.mean(pred_tr != label_tr)
		print("c=%f, train error %f" % (c, error_tr))

		# validation
		pred = lin_clf.predict(x_tu)
		error_tu = np.mean(pred != label_tu)
		print("c=%f, tune error %f" % (c, error_tu))
		if error_tu < best_error_tu:
			best_error_tu = error_tu
			bestsvm = lin_clf
	# test
	pred = bestsvm.predict(x_te)
	error_te = np.mean(pred != label_te)
	return best_error_tu, error_te

def linear_svm_classify2(x_tr, label_tr, x_tu, label_tu, x_te, label_te,
						standardization=False, alg='SVC', cache_size=1000, C_list=[0.1, 1.0, 10.0]):
	"""TIPS on Practical use of SVM:
	https://scikit-learn.org/stable/modules/svm.html
	"""
	global debug_mode
	if debug_mode:
		return 1., 1.

	if standardization:
		scaler = preprocessing.StandardScaler().fit(x_tr)
		x_tr = scaler.transform(x_tr)
		x_tu = scaler.transform(x_tu)
		x_te = scaler.transform(x_te)

	print("\n Running linear SVM!")

	cv_update_ratio = 5.
	(min_reg, max_reg) = (min(C_list), max(C_list)) if len(C_list) > 1 else (-np.inf, np.inf)
	reg_coef_set = set(C_list)
	best_error_tu = 1.0 + 1e-4
	while len(reg_coef_set) > 0:
		c = reg_coef_set.pop() # this is INVERSE of reg_coef in SVM
		if alg == 'SVC':
			lin_clf = svm.SVC(C=c, kernel="linear", cache_size=cache_size)
		elif alg == 'LinearSVC':
			lin_clf = svm.LinearSVC(C=c)
		# training
		lin_clf.fit(x_tr, label_tr)

		pred_tr = lin_clf.predict(x_tr)
		error_tr = np.mean(pred_tr != label_tr)
		print("c=%f, train error %f" % (c, error_tr))

		# validation
		pred = lin_clf.predict(x_tu)
		error_tu = np.mean(pred != label_tu)
		print("c=%f, tune error %f" % (c, error_tu))
		if error_tu < best_error_tu:
			best_error_tu = error_tu
			bestsvm = lin_clf
			best_reg_coef = c

		# if the best element is the last element of the set, expand the set from one side
		if len(reg_coef_set) == 0 and best_reg_coef == min_reg:
			reg_coef_set.add(min_reg / cv_update_ratio)
			min_reg /= cv_update_ratio
		if len(reg_coef_set) == 0 and best_reg_coef == max_reg:
			reg_coef_set.add(max_reg * cv_update_ratio)
			max_reg *= cv_update_ratio
	# test
	pred = bestsvm.predict(x_te)
	error_te = np.mean(pred != label_te)
	return best_error_tu, error_te



def purity_score(y_true, y_pred):
	"""
	https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
	"""
	# compute contingency matrix (also called confusion matrix)
	contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
	# return purity
	return float(np.sum(np.amax(contingency_matrix, axis=0))) / np.sum(contingency_matrix)


def best_map(L1, L2):
	# L1 should be the groundtruth labels and L2 should be the clustering labels we got
	Label1 = np.unique(L1)
	nClass1 = len(Label1)
	Label2 = np.unique(L2)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1, nClass2)
	G = np.zeros((nClass, nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i, j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:, 1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[c[i]]
	return newL2

def clustering_scores(label, pred,
					  average_method = 'warn', #''max'  #'arithmetic'
					  ACC_alg='main'):
	c_x = best_map(label, pred)
	# c_x = s
	nmi = normalized_mutual_info_score(label[:], c_x[:], average_method=average_method)
	ari = adjusted_rand_score(label[:], c_x[:])
	if ACC_alg == 'main':
		acc = purity_score(label[:], c_x[:])
	else:
		err_x = np.sum(label[:] != c_x[:])
		missrate = err_x.astype(float) / (label.shape[0])
		acc = 1. - missrate
	return acc ,nmi, ari


def spectral_clustering(x_tr, label_tr, x_tu, label_tu, x_te, label_te,
						n_clusters=10, standardization=False, compute_knn_graph = False):
	"""TIPS on Practical use of spectral_clustering:
	https://scikit-learn.org/stable/modules/clustering.html
	https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering
	https://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html#sphx-glr-auto-examples-cluster-plot-segmentation-toy-py
	https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering
	"""
	global debug_mode
	if debug_mode:
		return 0., 0., 0., 0.

	from sklearn.neighbors import kneighbors_graph
	def affinity_matrix(x, k):
		if compute_knn_graph:
			return kneighbors_graph(x, n_neighbors=k, mode='connectivity', include_self=True)
		else:
			return x

	if standardization:
		scaler = preprocessing.StandardScaler().fit(x_tr)
		x_tr = scaler.transform(x_tr)
		x_tu = scaler.transform(x_tu)
		x_te = scaler.transform(x_te)

	print("\n Running Spectral Clustering!")
	average_method = 'max' #'arithmetic'
	best_NMI_tu, best_ACC_tu = .0, .0
	for k in [5, 10, 20, 30, 50]:
		affinity_ = 'precomputed' if compute_knn_graph else 'nearest_neighbors'
		cluster_alg = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',
												 n_init=20, assign_labels='kmeans',
												 affinity = affinity_, n_neighbors=k)

		# # training
		# clustering = cluster_alg.fit(x_tr)
		#
		# pred_tr = clustering.labels_
		# NMI_tr = metrics.normalized_mutual_info_score(label_tr, pred_tr, average_method=average_method)
		# ACC_tr = purity_score(label_tr, pred_tr)
		# print("k=%f, train clustering NMI: %f, purity: :%f" % (k, NMI_tr, ACC_tr))

		# validation
		pred = cluster_alg.fit_predict(affinity_matrix(x_tu, k))
		NMI_tu = metrics.normalized_mutual_info_score(label_tu, pred, average_method=average_method)
		ACC_tu = purity_score(label_tu, pred)
		print("k=%f, tune clustering NMI: %f, purity: :%f" % (k, NMI_tu, ACC_tu))

		if NMI_tu > best_NMI_tu:
			best_NMI_tu = NMI_tu
			bestAlg_NMI = cluster_alg

		if ACC_tu > best_ACC_tu:
			best_ACC_tu = ACC_tu
			bestAlg_ACC = cluster_alg

	# test
	pred = bestAlg_NMI.fit_predict(affinity_matrix(x_te, bestAlg_NMI.n_neighbors))
	NMI_te = metrics.normalized_mutual_info_score(label_te, pred, average_method=average_method)

	pred = bestAlg_ACC.fit_predict(affinity_matrix(x_te, bestAlg_ACC.n_neighbors))
	ACC_te = purity_score(label_te, pred)
	return best_NMI_tu, NMI_te, best_ACC_tu, ACC_te


def spectral_clustering2(x_tr, label_tr, x_te, label_te, k_knn = [5, 10, 20, 30, 50],
						 n_clusters=10, assign_labels='kmeans', affinity = 'nearest_neighbors', degree_poly=8, gamma=None,
						 average_method = 'warn',  # ''max'  #'arithmetic'
						 ACC_alg = 'main'):
	"""
	assign_labels={"kmeans", "discretize"}
	TIPS on Practical use of spectral_clustering:
	https://scikit-learn.org/stable/modules/clustering.html
	https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering
	https://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html#sphx-glr-auto-examples-cluster-plot-segmentation-toy-py
	https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering
	"""
	global debug_mode
	if debug_mode:
		return 0., 0., 0., 0., 0., 0.

	compute_knn_graph = True if affinity[0:4]=='pre_' else False
	affinity_ = 'precomputed' if compute_knn_graph else affinity

	from sklearn.neighbors import kneighbors_graph
	def affinity_matrix(x, k):
		if compute_knn_graph:
			return kneighbors_graph(x, n_neighbors=k, mode='connectivity', include_self=True)
		else:
			return x

	print("\n Running Spectral Clustering!")
	best_NMI_tu, best_ACC_tu, best_ARI_tu = .0, .0, 0.
	for k in k_knn:
		cluster_alg = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',
												 n_init=20, assign_labels=assign_labels,
												 affinity = affinity_, n_neighbors=k,
												 gamma=gamma, degree=degree_poly)

		# # training
		# clustering = cluster_alg.fit(x_tr)
		#
		# pred_tr = clustering.labels_
		# NMI_tr = metrics.normalized_mutual_info_score(label_tr, pred_tr, average_method=average_method)
		# ACC_tr = purity_score(label_tr, pred_tr)
		# print("k=%f, train clustering NMI: %f, purity: :%f" % (k, NMI_tr, ACC_tr))

		# validation
		pred = cluster_alg.fit_predict(affinity_matrix(x_tr, k))
		ACC_tu, NMI_tu, ARI_tu = clustering_scores(label_tr,pred, average_method=average_method, ACC_alg= ACC_alg)
		# NMI_tu = metrics.normalized_mutual_info_score(label_tr, pred, average_method=average_method)
		# ACC_tu = purity_score(label_tr, pred)
		# ARI_tu = adjusted_rand_score(label_tr, pred)
		print("k=%f, tune clustering NMI: %f, purity: :%f, ARI: :%f" % (k, NMI_tu, ACC_tu, ARI_tu))

		if NMI_tu > best_NMI_tu:
			best_NMI_tu = NMI_tu
			bestAlg_NMI = cluster_alg

		if ACC_tu > best_ACC_tu:
			best_ACC_tu = ACC_tu
			bestAlg_ACC = cluster_alg

		if ARI_tu > best_ARI_tu:
			best_ARI_tu = ARI_tu
			bestAlg_ARI = cluster_alg

	# test
	pred = bestAlg_NMI.fit_predict(affinity_matrix(x_te, bestAlg_NMI.n_neighbors))
	_, NMI_te, _ = clustering_scores(label_te, pred, average_method=average_method, ACC_alg=ACC_alg)
	# NMI_te = metrics.normalized_mutual_info_score(label_te, pred, average_method=average_method)

	pred = bestAlg_ACC.fit_predict(affinity_matrix(x_te, bestAlg_ACC.n_neighbors))
	ACC_te, _, _ = clustering_scores(label_te, pred, average_method=average_method, ACC_alg=ACC_alg)
	# ACC_te = purity_score(label_te, pred)

	pred = bestAlg_ARI.fit_predict(affinity_matrix(x_te, bestAlg_ARI.n_neighbors))
	_, _, ARI_te = clustering_scores(label_te, pred, average_method=average_method, ACC_alg=ACC_alg)
	# ARI_te = adjusted_rand_score(label_te, pred)
	return best_NMI_tu, NMI_te, best_ACC_tu, ACC_te, best_ARI_tu, ARI_te


def spectral_clustering3(x_tr, label_tr, n_clusters, alpha=8):
	# C: coefficient matrix, K: number of clusters, d: dimension of each subspace
	# C = 0.5*(C + C.T)
	# r = d*K + 1
	# U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
	# U = U[:,::-1]
	# S = np.sqrt(S[::-1])
	# S = np.diag(S)
	# U = U.dot(S)
	U = normalize(x_tr, norm='l2', axis = 1)
	Z = U.dot(U.T)
	Z = Z * (Z>0)
	L = np.abs(Z ** alpha)
	L = L/L.max()
	L = 0.5 * (L + L.T)
	spectral = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize')
	spectral.fit(L)
	pred = spectral.fit_predict(L) + 1
	ACC, NMI, ARI = clustering_scores(label_tr, pred)
	return NMI, ACC, ARI


def kmeans_clustering(x_tr, label_tr, x_tu, label_tu, x_te, label_te,
					n_clusters=10, standardization=False):

	if standardization:
		scaler = preprocessing.StandardScaler().fit(x_tr)
		x_tr = scaler.transform(x_tr)
		x_tu = scaler.transform(x_tu)
		x_te = scaler.transform(x_te)

	print("\n Running Spectral Clustering!")
	average_method = 'max' #'arithmetic'
	best_NMI_tu, best_ACC_tu = .0, .0
	cluster_alg = cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=20)

	# # training
	# clustering = cluster_alg.fit(x_tr)
	#
	# pred_tr = clustering.labels_
	# NMI_tr = metrics.normalized_mutual_info_score(label_tr, pred_tr, average_method=average_method)
	# ACC_tr = purity_score(label_tr, pred_tr)
	# print("k=%f, train clustering NMI: %f, purity: :%f" % (k, NMI_tr, ACC_tr))

	# validation
	pred = cluster_alg.fit_predict(x_tu)
	NMI_tu = metrics.normalized_mutual_info_score(label_tu, pred, average_method=average_method)
	ACC_tu = purity_score(label_tu, pred)
	print("tune clustering NMI: %f, purity: :%f" % (NMI_tu, ACC_tu))

	if NMI_tu > best_NMI_tu:
		best_NMI_tu = NMI_tu
		bestAlg_NMI = cluster_alg

	if ACC_tu > best_ACC_tu:
		best_ACC_tu = ACC_tu
		bestAlg_ACC = cluster_alg

	# test
	pred = bestAlg_NMI.fit_predict(x_te)
	NMI_te = metrics.normalized_mutual_info_score(label_te, pred, average_method=average_method)

	pred = bestAlg_ACC.fit_predict(x_te)
	ACC_te = purity_score(label_te, pred)
	return best_NMI_tu, NMI_te, best_ACC_tu, ACC_te


def compute_ap_sub(recall, prcs):
	recall = np.concatenate([np.zeros(1), recall, np.ones(1)])
	prcs = np.concatenate([np.zeros(1), prcs, np.zeros(1)])

	n_dim = prcs.shape[0]
	#     prcs_ = [max(prcs[i], prcs[i+1]) for i in range(n_dim-1)]
	#     prcs = np.array(prcs_ + [0.])
	for i in range(n_dim - 2, -1, -1):
		prcs[i] = max(prcs[i], prcs[i + 1])
	idx_ = np.where(recall[1:] != recall[0:-1])[0] + 1
	return np.sum((recall[idx_] - recall[idx_ - 1]) * prcs[idx_])

def compute_ap(y_true, y_score):
	y_true = 2. * y_true - 1.

	idx_ = np.argsort(-y_score)
	tp = y_true[idx_] >= 0
	fp = y_true[idx_] < 0

	fp = np.cumsum(fp)
	tp = np.cumsum(tp)

	eps = 1e-8
	recall = tp / (np.sum(y_true > 0) + eps)
	prcs = tp / (fp + tp + eps)
	return compute_ap_sub(recall, prcs)

def compute_mAP(y_true, y_score):
	n_dim = y_true.shape[1]
	ap_list = [compute_ap(y_true=y_true[:, i], y_score=y_score[:, i]) for i in range(n_dim)]
	return np.mean(np.array(ap_list))


""" ----------multiclass logistic regression ------------"""
from sklearn.multiclass import OneVsRestClassifier
def multiclass_logistic_regression0(x_tr, label_tr, x_tu, label_tu, x_te, label_te,
								   args):
	"""TIPS on multiclass_logistic_regression:
		https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
		TIPS on mean Average Precision (mAP):
		https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

		MULTI-Label classification:
		https://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification-format
		https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier

		other possible available codes:
		https://github.com/benhamner/Metrics
		pip install ml_metrics
		https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
	"""
	global debug_mode
	if debug_mode:
		return 0., 0.

	print("\n Running multiclass logistic regression!")
	if args.reg_coef is None:
		args.reg_coef = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10.]

	best_mAP_tu = 0.0
	for reg_coef in args.reg_coef:
		mclr = OneVsRestClassifier(
			LogisticRegression(penalty= args.penalty, C=1./reg_coef,
							   solver=args.solver, max_iter=args.max_iter, tol=1e-8,
							   multi_class=args.multi_class, verbose=0),
								n_jobs=-1)

		# training
		mclr.fit(x_tr, label_tr)

		pred_tr = mclr.predict(x_tr)
		mAP_tr = average_precision_score(y_true=label_tr, y_score=pred_tr)
		print("regularizer coeficient=%f, train mAP %f" % (reg_coef, mAP_tr))

		# validation
		pred = mclr.predict(x_tu)
		mAP_tu = average_precision_score(y_true=label_tu, y_score=pred)
		print("regularizer coeficient=%f, tune mAP %f" % (reg_coef, mAP_tu))
		if mAP_tu > best_mAP_tu:
			best_mAP_tu = mAP_tu
			best_mclr = mclr
	# test
	pred = mclr.predict(x_te)
	mAP_te = average_precision_score(y_true=label_te, y_score=pred)
	return best_mAP_tu, mAP_te


def function_factory(x, loss, shape):
	fn = lambda x: loss(tf.reshape(x, shape))
	loss, grads = tfp.math.value_and_gradient(fn, x)
	#     print(grads)
	grads = tf.reshape(grads, shape=[-1])
	return loss, grads

class ml_logistic_reg_tf(object):
	def __init__(self, n_feat, n_label, reg_coef, max_iter, sess, name):
		self.max_iter = max_iter
		self.reg_coef = reg_coef
		self.eps = 1e-8 # was 1e-6
		self.dtype = tf.float64
		self.n_feat = n_feat
		self.n_label = n_label
		self.shape_W = [n_feat+1, n_label]
		self.ndims = np.prod(np.array(self.shape_W))
		self.sess = sess
		# self.graph = tf.get_default_graph()
		# with self.graph.as_default():
		# with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			# self.W = tf.get_variable(name='W_mclr', shape = self.shape_W,
			# 						 dtype=self.dtype,)
			# self.features = tf.get_variable(name='features', shape=[1, self.n_feat+1], validate_shape=False,
			# 								trainable=False, dtype=self.dtype,)
			# self.labels = tf.get_variable(name='labels', shape=[1, self.n_label], validate_shape=False,
			# 							  trainable=False, dtype=self.dtype,)
			#
			# self.sess.run(tf.global_variables_initializer())
			# feat_plhr = tf.placeholder(self.dtype, shape=[None, self.n_feat])
			# feat_asgn = tf.assign(self.features, value=feat_plhr, validate_shape=False)
			# self.load_feat = lambda val: self.sess.run(feat_asgn, feed_dict={feat_plhr:val})
			# self.load_labels = lambda val: self.sess.run(tf.assign(self.labels, value=val, validate_shape=False))
			# self.load_feat = lambda val: self.features.load(val, self.sess)
			# self.load_labels = lambda val: self.labels.load(val, self.sess)
			# self.assign_parameters = lambda param: self.W.load(param, self.sess)


	def fit(self, X, y):
		"""
		how to use L-BFGS in tensorflow 2:
		https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993
		"""
		# self.load_feat(X)
		# self.load_labels(y)
		self.loss_fn = lambda W: self.loss(X=X, y=y, W=W)
		func = lambda x: function_factory(x, loss=self.loss_fn, shape = self.shape_W)
		init = tf.ones(self.ndims, dtype=self.dtype)
		optim_results = tfp.optimizer.lbfgs_minimize(
			func,
			initial_position=init,
			max_iterations=self.max_iter,
		)

		# after training, assign the final optimized parameters
		self.W = tf.reshape(optim_results.position, shape=self.shape_W)
		# self.assign_parameters(optim_results.position)
		return
	# def fit(self, X, y):
	# 	"""
	# 	how to use L-BFGS in tensorflow 2:
	# 	https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993
	# 	"""
	# 	self.load_feat(X)
	# 	self.load_labels(y)
	# 	self.loss_fn = lambda W: self.loss(X=self.features, y=self.labels, W=W)
	# 	func = lambda x: function_factory(x, loss=self.loss_fn, shape = self.shape_W)
	# 	init = tf.ones(self.ndims, dtype=self.dtype)
	# 	optim_results = tfp.optimizer.lbfgs_minimize(
	# 		func,
	# 		initial_position=init,
	# 		max_iterations=self.max_iter,
	# 	)
	#
	# 	# after training, assign the final optimized parameters
	# 	self.assign_parameters(optim_results.position)
	# 	return

	def loss(self, X, y, W=None):
		# compute regularized Cross-Entropy as loss
		if W is None:
			W = self.W

		return self.CE(tf.constant(y, self.dtype), self.predict_prob(X, W=W)) + \
			   self.reg_coef * tf.nn.l2_loss(W)

	def CE(self, y, z):
		return - tf.reduce_sum(
			y * tf.log(self.eps + z) + (1 - y) * tf.log(self.eps + 1 - z), axis=[0, 1])

	def predict(self, X, W=None):
		if W is None:
			W = self.W
		# with self.graph.as_default():
		# self.load_feat(X)
		w_, b_ = tf.split(W, [self.n_feat, 1], axis=0)
		return tf.matmul(tf.constant(X, self.dtype), w_) + b_

	def predict_labeles(self, X, W=None):
		return

	def predict_prob(self, X, W=None):
		# with self.graph.as_default():
		z = self.predict(X, W=W)
		return tf.sigmoid(z)


def multiclass_logistic_regression1(x_tr, label_tr, x_tu, label_tu, x_te, label_te,
								   args):
	"""TIPS on multiclass_logistic_regression:
		https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
		TIPS on mean Average Precision (mAP):
		https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
		other possible available codes:
		https://github.com/benhamner/Metrics
		pip install ml_metrics
		https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
	"""
	# global debug_mode
	# if debug_mode:
	# 	return 0., 0.

	print("\n Running multiclass logistic regression!")
	if args.reg_coef is None:
		args.reg_coef = [0., 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10.]

	g = tf.Graph()
	sess = tf.InteractiveSession(graph=g)
	args.sess = sess
	with g.as_default():

		best_mAP_tu = 0.0
		for reg_coef in args.reg_coef:
			mclr = ml_logistic_reg_tf(n_feat=x_tr.shape[1], n_label=label_tr.shape[1],
									  reg_coef=reg_coef, max_iter=args.max_iter, sess=args.sess, name='mclr%f'%reg_coef)

			# training
			mclr.fit(x_tr, label_tr)

			pred_tr = sess.run(mclr.predict(x_tr))
			mAP_tr = average_precision_score(y_true=label_tr, y_score=pred_tr)
			print("regularizer coeficient=%f, train mAP %f" % (reg_coef, mAP_tr))

			# validation
			pred = sess.run(mclr.predict(x_tu))
			mAP_tu = average_precision_score(y_true=label_tu, y_score=pred)
			print("regularizer coeficient=%f, tune mAP %f" % (reg_coef, mAP_tu))
			if mAP_tu > best_mAP_tu:
				best_mAP_tu = mAP_tu
				best_mclr = mclr
		# test
		pred = sess.run(best_mclr.predict(x_te))
		mAP_te = average_precision_score(y_true=label_te, y_score=pred)
		print("regularizer coeficient=%f, test mAP %f" % (reg_coef, mAP_te))
	sess.close()
	return best_mAP_tu, mAP_te

import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Activation

class ml_logistic_reg_keras(object):
	def __init__(self, n_feat, n_label, reg_coef, args):
		self.dtype = np.float64 # np.float32 #
		keras.backend.set_floatx("float64") #("float32")
		self.args = args
		self.reg_coef = reg_coef # /(38*10000)
		# self.eps = 1e-8 # was 1e-6
		self.n_feat = n_feat
		self.n_label = n_label
		self.batch_size = args.batch_size
		self.max_iter = args.max_iter
		self.verbose = self.args.verbose if hasattr(self.args, 'verbose') else 0

		self.model = Sequential()
		self.model.add(Dense(n_label, input_dim=n_feat, activation='sigmoid',
							 bias_regularizer=None, # keras.regularizers.l2(self.reg_coef), # VERY IMPORTANT
							 kernel_regularizer=keras.regularizers.l2(self.reg_coef)))

		if hasattr(args, 'opt'):
			if args.opt is not None:
				self.optimizer = args.opt
		else:
			self.optimizer = keras.optimizers.Adam(0.001)
			# self.optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.99, nesterov=False)


		def CE(y, z):
			eps = 1e-6
			return - self.n_label * tf.reduce_mean(y * tf.log(eps + z) + (1 - y) * tf.log(eps + 1 - z))

		self.model.compile(optimizer=self.optimizer,
						   loss=CE, #'binary_crossentropy',  #
						   metrics=None)
		# https://stackoverflow.com/questions/42081257/why-binary-crossentropy-and-categorical-crossentropy-give-different-performances/45435674#45435674

	def fit(self, X, y):
		return self.model.fit(x=X.astype(self.dtype), y=y.astype(self.dtype),
							  batch_size=self.batch_size, epochs=self.max_iter,
							  verbose=self.verbose,
							  callbacks=[
								  keras.callbacks.callbacks.EarlyStopping(
									  monitor='loss', min_delta=1e-10,
									  patience=int(self.max_iter/5.), verbose=self.verbose, mode='min')]
							  )

	def predict(self, X):
		return self.model.predict(x=X.astype(self.dtype))

def multiclass_logistic_regression(x_tr, label_tr, x_tu, label_tu, x_te, label_te,
								   args):
	""""""
	# global debug_mode
	# if debug_mode:
	# 	return 0., 0.

	print("\n Running multiclass logistic regression!")
	if args.reg_coef is None:
		args.reg_coef = [0., 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10.]

	# g = tf.Graph()
	# sess = tf.InteractiveSession(graph=g)
	# args.sess = sess
	# with g.as_default():

	best_mAP_tu = 0.0
	for reg_coef in args.reg_coef:
		mclr = ml_logistic_reg_keras(n_feat=x_tr.shape[1], n_label=label_tr.shape[1],
									 reg_coef=reg_coef, max_iter=args.max_iter)

		# training
		mclr.fit(x_tr, label_tr)

		pred_tr = mclr.predict(x_tr)
		mAP_tr = average_precision_score(y_true=label_tr, y_score=pred_tr)
		print("regularizer coeficient=%f, train mAP %f" % (reg_coef, mAP_tr))

		# validation
		pred = mclr.predict(x_tu)
		mAP_tu = average_precision_score(y_true=label_tu, y_score=pred)
		print("regularizer coeficient=%f, tune mAP %f" % (reg_coef, mAP_tu))
		if mAP_tu > best_mAP_tu:
			best_mAP_tu = mAP_tu
			best_mclr = mclr
	# test
	pred = best_mclr.predict(x_te)
	mAP_te = average_precision_score(y_true=label_te, y_score=pred)
	print("regularizer coeficient=%f, test mAP %f" % (reg_coef, mAP_te))
	return best_mAP_tu, mAP_te


# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# class mclr_torch(nn.Module):
# 	def __init__(self, n_feat, n_label, reg_coef, args):
# 		super(mclr_torch, self).__init__()
#
# 		self.device = args.device
#
# 		torch.set_default_dtype(torch.float64)
# 		self.dtype = np.float64 #np.float64
#
# 		self.args = args
# 		self.reg_coef = reg_coef  #/n_label todo: to normalize the coef
# 		self.n_feat = n_feat
# 		self.n_label = n_label
# 		self.verbose = self.args.verbose if hasattr(self.args, 'verbose') else 0
#
# 		self.layers = nn.Sequential(
# 			nn.Linear(n_feat, n_label),
# 			nn.Sigmoid()).to(device=self.device)
# 		torch.nn.init.normal_(self.layers._modules['0'].weight, std=.001)
# 		# torch.nn.init.constant_(self.layers._modules['0'].weight, .001) #temp
# 		torch.nn.init.constant_(self.layers._modules['0'].bias, 0.)
#
# 		self.criterion = nn.BCELoss(reduction='sum')
#
# 		if args.opt_ == "lbfgs_torch":
# 			self.optimizer = optim.LBFGS(
# 				self.parameters(),
# 				max_iter=args.max_iter,
# 				max_eval=None,
# 				tolerance_grad=1e-09,
# 				tolerance_change=1e-09,
# 				history_size=100) # default is 100
# 		else:
# 			self.optimizer = torch.optim.Adamax(self.parameters(), lr=0.002) #args.opt(self.parameters())
#
#
# 	def forward(self, x):
# 		outputs = self.layers(x)
# 		return outputs
#
# 	def fit(self, X, y):
# 		if self.args.opt_ == "lbfgs_torch":
# 			self.fit_lbfgs(X,y)
# 		else:
# 			self.fit_optim(X,y)
#
# 	def fit_optim(self, X, y):
# 		X = torch.from_numpy(X.astype(self.dtype)).to(self.device)
# 		y = torch.from_numpy(y.astype(self.dtype)).to(self.device)
#
# 		t0 = time.time()
# 		for epoch in range(self.args.max_iter):
#
# 			self.optimizer.zero_grad()
# 			out = self.forward(X)
# 			# reg_loss = 0
# 			# for param in self.parameters():
# 			# 	reg_loss += 0.5 * torch.sum(param ** 2)
# 			reg_loss = 0.5 * torch.sum(self.layers._modules['0'].weight ** 2) # removong the regularizer from the bias had huge improvement
# 			loss = self.criterion(out, y) + self.reg_coef * X.shape[0] * reg_loss
# 			if self.verbose > 0:
# 				print('loss:', loss.item())
# 			loss.backward()
# 			self.optimizer.step()
#
# 		if self.verbose > 0:
# 			print('Training time:', time.time() - t0)
#
# 	def fit_lbfgs(self, X, y):
# 		X = torch.from_numpy(X.astype(self.dtype)).to(self.device)
# 		y = torch.from_numpy(y.astype(self.dtype)).to(self.device)
# 		# X = Variable(X.view(-1, 28 * 28))
# 		# y = Variable(y)
# 		t0 = time.time()
# 		def closure():
# 			self.optimizer.zero_grad()
# 			out = self(X)
# 			# reg_loss = 0
# 			# for param in self.parameters():
# 			# 	reg_loss += 0.5 * torch.sum(param ** 2)
# 			reg_loss = 0.5 * torch.sum(self.layers._modules['0'].weight ** 2)
# 			loss = self.criterion(out, y) + self.reg_coef * X.shape[0]*reg_loss
# 			if self.verbose >0:
# 				print('loss:', loss.item())
# 			loss.backward()
# 			return loss
#
# 		self.optimizer.step(closure)
# 		if self.verbose > 0:
# 			print('Training time:', time.time() - t0)
#
# 	def predict(self, X):
# 		X = torch.as_tensor(X.astype(self.dtype), device=self.device)
# 		return self.layers._modules['0'](X).cpu().data.numpy() # todo: to match the Matlab implementation, Indeed it doesn't matter
# 		# return self.forward(X).cpu().data.numpy()  #.squeeze()


def logistic_regression_allfolds(inp_tr, inp_tu, inp_te, data, args):

	cv_update_ratio = 10.

	if args.opt_ == 'lbfgs_torch' or args.opt_[0:5] == "torch":
		logistic_regression_alg = mclr_torch
	else:
		logistic_regression_alg = ml_logistic_reg_keras


	reg_coef_list = args.reg_coef
	mAP_list_tu = []
	for fold in args.folds_list:
		args._print("FOLD = %d" % fold)
		pair_tr_, pair_tu_, pair_te_ = data.permute_folds(inp_tr=[inp_tr, data.label_trlb],
														  inp_tu=[inp_tu, data.label_tu],
														  inp_te=[inp_te, data.label_te],
														  fold=fold, n_folds=len(args.folds_list))

		if args.evalmode:
			x_tr, label_tr = pair_tr_[0], pair_tr_[1]
			x_tu, label_tu = pair_tu_[0], pair_tu_[1]
			x_te, label_te = pair_te_[0], pair_te_[1]
		else:
			x_tr 		= np.concatenate([pair_tr_[0], pair_tu_[0]], axis=0)
			label_tr 	= np.concatenate([pair_tr_[1], pair_tu_[1]], axis=0)
			x_tu, label_tu = pair_te_[0], pair_te_[1]
			x_te, label_te = pair_te_[0], pair_te_[1]

		(min_reg, max_reg) = (min(reg_coef_list), max(reg_coef_list)) if len(reg_coef_list)>1 else (-np.inf, np.inf)
		reg_coef_set = set(reg_coef_list)
		best_mAP_tu = 0.0
		while len(reg_coef_set)>0:
			reg_coef = reg_coef_set.pop()
			mclr = logistic_regression_alg(n_feat=inp_tr.shape[1], n_label=data.label_trlb.shape[1],
										 reg_coef=reg_coef, args=args)
			if args.opt_ == 'lbfgs_torch':
				mclr.to(device=args.device)

			mclr.fit(x_tr, label_tr)

			pred_tr = mclr.predict(x_tr)
			mAP_tr_ = compute_mAP(y_true=label_tr, y_score=pred_tr)

			# validation
			pred_tu_ = mclr.predict(x_tu)
			mAP_tu_ = compute_mAP(y_true=label_tu, y_score=pred_tu_)
			args._print("regularizer coef=%f, train mAP %f, tune mAP %f" % (reg_coef, mAP_tr_, mAP_tu_))

			# mAP_tr___ = average_precision_score(y_true=label_tr, y_score=pred_tr)
			# args._print("SKlearn mAP: regularizer coeficient=%f, train mAP %f" % (reg_coef, mAP_tr___))
			# mAP_tu___ = average_precision_score(y_true=label_tu, y_score=pred_tu_)
			# args._print("SKlearn mAP: regularizer coeficient=%f, tune mAP %f" % (reg_coef, mAP_tu___))

			if mAP_tu_ > best_mAP_tu:
				best_mAP_tu = mAP_tu_
				best_mclr = mclr
				best_reg_coef = reg_coef

			# if the best element is the last element of the set, expand the set from one side
			if len(reg_coef_set) == 0 and best_reg_coef == min_reg:
				reg_coef_set.add(min_reg / cv_update_ratio)
				min_reg /= cv_update_ratio
			if len(reg_coef_set) == 0 and best_reg_coef == max_reg:
				reg_coef_set.add(max_reg * cv_update_ratio)
				max_reg *= cv_update_ratio

		reg_coef_list = [best_reg_coef] # so it means that the best reg coef is found from first fold and will be used for the following folds
		mAP_list_tu.append(best_mAP_tu)

	mAP_tu_avg_allfolds = np.mean(np.array(mAP_list_tu))

	best_mclr = logistic_regression_alg(n_feat=inp_tr.shape[1], n_label=data.label_trlb.shape[1],
							reg_coef=best_reg_coef, args=args)
	if args.opt_ == 'lbfgs_torch':
		best_mclr.to(device=mclr.device)

	best_mclr.fit(np.concatenate([inp_tr, inp_tu], axis=0),
				  np.concatenate([data.label_trlb, data.label_tu], axis=0))
	pred_te = best_mclr.predict(inp_te)
	mAP_te = compute_mAP(y_true=data.label_te, y_score=pred_te)
	args._print("\n FINALLY: regularizer coeficient=%f, avg-validation mAP %f, test mAP %f"
				% (best_reg_coef, mAP_tu_avg_allfolds, mAP_te))

	# if args.opt_ == 'lbfgs_torch':
	# 	torch.cuda.empty_cache()
	return mAP_tu_avg_allfolds, mAP_te


""" ----------TSNE ------------"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

COLORS={
1:[1.0000, 0,      0     ],
2:[0,      1.0000, 0     ],
3:[0,      0,      1.0000],
4:[1.0000, 0,      1.0000],
5:[0.9569, 0.6431, 0.3765],
6:[0.4000, 0.8039, 0.6667],
7:[0.5529, 0.7137, 0.8039],
8:[0.8039, 0.5882, 0.8039],
9:[0.7412, 0.7176, 0.4196],
10:[0,      0,      0    ],
0:[0,      0,      0     ]}

LABELS={
	0:'$0$',
	1:'$1$',
	2:'$2$',
	3:'$3$',
	4:'$4$',
	5:'$5$',
	6:'$6$',
	7:'$7$',
	8:'$8$',
	9:'$9$',
	10:'$0$',}

def tsne_embeding(x, labels, savefile=None):
	t0 = time.time()
	color = labels
	tsne=TSNE(perplexity=20, n_components=2, init="pca", n_iter=3000)
	x_tsne=tsne.fit_transform(np.asfarray(x, dtype="float"))
	print("time of TSNE: " + str(time.time() - t0) )
	if savefile:
		plt.figure(figsize=(10, 10))
		for i in range(min(labels), max(labels) + 1):
			idx = np.argwhere(labels == i)
			plt.plot(x_tsne[idx, 0], x_tsne[idx, 1], "o", c=COLORS[i], marker=LABELS[i], markersize=8.0, label=LABELS[i], alpha=.5)
			# plt.scatter(x_tsne[idx, 0], x_tsne[idx, 1], c=COLORS[i], cmap=plt.cm.Spectral, marker=LABELS[i], alpha=.5)
		plt.legend(loc='best', fontsize='xx-large', markerscale=1.5, shadow=True)
		plt.tight_layout()
		plt.axis('off')
		plt.savefig(savefile) #+'.eps')
		plt.close()
	return x_tsne


""" ----------Orthogonality Measures ------------"""

def orth_measure(X, Y, X_mean=None, Y_mean=None, X_sigmasq=None, Y_sigmasq=None, n_samples=1):
	n = X.shape[0]
	if n % n_samples != 0:
		raise ValueError('shape of input tensors are not multiple of n_samples')
	n //= n_samples

	def corr(x, y, mu_x=None, mu_y=None, sigsq_x=None, sigsq_y=None):
		mu_x = np.mean(x, 0) if mu_x is None else mu_x
		mu_y = np.mean(y, 0) if mu_y is None else mu_y
		sigsq_x = np.matmul((x - mu_x).transpose(), (x - mu_x)) / len(x) if sigsq_x is None else sigsq_x
		sigsq_y = np.matmul((y - mu_y).transpose(), (y - mu_y)) / len(y) if sigsq_y is None else sigsq_y
		denom = np.sqrt(np.sqrt(np.sum(sigsq_x**2)) * np.sqrt(np.sum(sigsq_y**2)) + 1e-7)
		return np.linalg.norm(1 / len(x) * np.matmul((x - mu_x).transpose(), (y - mu_y)), ord='fro') / denom

	return np.mean([corr(x=X[i::n], y=Y[i::n],
						 mu_x=X_mean[i] if not(X_mean is None) else None, mu_y=Y_mean[i] if not(Y_mean is None) else None,
						 sigsq_x=X_sigmasq[i] if not(X_sigmasq is None) else None, sigsq_y=Y_sigmasq[i] if not(Y_sigmasq is None) else None)
					for i in range(n)])

def orth_measure2(X, Y):
	return np.linalg.norm(np.matmul(X.transpose(), Y), ord='fro')/ \
		   np.sqrt(np.linalg.norm(np.matmul(X.transpose(), X), ord='fro') * np.linalg.norm(np.matmul(Y.transpose(), Y), ord='fro') + 1e-7)

	# In VCCA_Wang it is (not true as orth(X,X)!=0):
	# 	return (np.linalg.norm(np.matmul(X.transpose(), Y), ord='fro')/(np.linalg.norm(X, ord='fro') * np.linalg.norm(Y, ord='fro') + 0.)) ** 2

from sklearn.metrics.pairwise import cosine_similarity
def orth_measure3(X, Y):
	return np.mean(np.abs(cosine_similarity(X.transpose(), Y.transpose()) ))
	# sig_X = np.sqrt(np.mean(X**2., 0))
	# sig_Y = np.sqrt(np.mean(Y**2., 0))
	# X /= sig_X
	# Y /= sig_Y
	# return np.mean(np.abs(np.matmul(X.transpose(), Y) / X.shape[0]))
	# return np.linalg.norm(cosine_similarity(X.transpose(), Y.transpose()), ord='fro')

