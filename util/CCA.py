
import numpy as np
import tensorflow as tf

eps_eig = 1e-6

def linCCA(H1, H2, dim, rcov1, rcov2):

	N, d1 = H1.shape
	_, d2 = H2.shape

	# Remove mean.
	m1 = np.mean(H1, axis=0, keepdims=True)
	H1 = H1 - np.tile(m1, [N,1])

	m2 = np.mean(H2, axis=0, keepdims=True)
	H2 = H2 - np.tile(m2, [N,1])

	S11 = np.matmul(H1.transpose(), H1) / (N-1) + rcov1 * np.eye(d1)
	S22 = np.matmul(H2.transpose(), H2) / (N-1) + rcov2 * np.eye(d2)
	S12 = np.matmul(H1.transpose(), H2) / (N-1)

	E1, V1 = np.linalg.eig(S11)
	E2, V2 = np.linalg.eig(S22)

	# For numerical stability.
	idx1 = np.where(E1>eps_eig)[0]
	E1 = E1[idx1]
	V1 = V1[:, idx1]

	idx2 = np.where(E2>eps_eig)[0]
	E2 = E2[idx2]
	V2 = V2[:, idx2]

	K11 = np.matmul( np.matmul(V1, np.diag(np.reciprocal(np.sqrt(E1)))), V1.transpose())
	K22 = np.matmul( np.matmul(V2, np.diag(np.reciprocal(np.sqrt(E2)))), V2.transpose())
	T = np.matmul( np.matmul(K11, S12), K22)
	# print(T)
	U, E, V = np.linalg.svd(T, full_matrices=False)
	V = V.transpose()

	A = np.matmul(K11, U[:, 0:dim])
	B = np.matmul(K22, V[:, 0:dim])
	E = E[0:dim]

	return A, B, m1, m2, E


def CanonCorr(H1, H2, N, d1, d2, dim, rcov1, rcov2):

	# Remove mean.
	m1 = tf.reduce_mean(H1, axis=0, keep_dims=True)
	H1 = tf.subtract(H1, m1)

	m2 = tf.reduce_mean(H2, axis=0, keep_dims=True)
	H2 = tf.subtract(H2, m2)

	S11 = tf.matmul(tf.transpose(H1), H1) / (N-1) + rcov1 * tf.eye(d1)
	S22 = tf.matmul(tf.transpose(H2), H2) / (N-1) + rcov2 * tf.eye(d2)
	S12 = tf.matmul(tf.transpose(H1), H2) / (N-1)

	E1, V1 = tf.self_adjoint_eig(S11)
	E2, V2 = tf.self_adjoint_eig(S22)

	# For numerical stability.
	idx1 = tf.where(E1>eps_eig)[:,0]
	E1 = tf.gather(E1, idx1)
	V1 = tf.gather(V1, idx1, axis=1)

	idx2 = tf.where(E2>eps_eig)[:,0]
	E2 = tf.gather(E2, idx2)
	V2 = tf.gather(V2, idx2, axis=1)

	K11 = tf.matmul(tf.matmul(V1, tf.diag(tf.reciprocal(tf.sqrt(E1)))), tf.transpose(V1))
	K22 = tf.matmul(tf.matmul(V2, tf.diag(tf.reciprocal(tf.sqrt(E2)))), tf.transpose(V2))
	T = tf.matmul(tf.matmul(K11, S12), K22)

	# Eigenvalues are sorted in increasing order.
	E3, U = tf.self_adjoint_eig(tf.matmul(T, tf.transpose(T)))
	idx3 = tf.where(E3 > eps_eig)[:, 0]
	# This is the thresholded rank.
	dim_svd = tf.cond(tf.size(idx3) < dim, lambda: tf.size(idx3), lambda: dim)

	return tf.reduce_sum(tf.sqrt(E3[-dim_svd:])), E3, dim_svd


def inf_CCA(self, y1, y2, mode, est_CCA=False, dim = -1):
	global debug_mode
	debug_mode = self.selfargs.debug_mode

	if est_CCA:
		self.A_cca, self.B_cca, self.m1_cca, self.m2_cca, _= linCCA(y1, y2, dim, rcov1= 1e-4, rcov2=1e-4)

	Phi_cca1 = np.matmul(y1 - self.m1_cca, self.A_cca)
	Phi_cca2 = np.matmul(y2 - self.m2_cca, self.B_cca)

	if mode == 'tr':
		self.Phi_tr_cca1 = Phi_cca1
		self.Phi_tr_cca2 = Phi_cca2
	elif mode == 'tu':
		self.Phi_tu_cca1 = Phi_cca1
		self.Phi_tu_cca2 = Phi_cca2
	elif mode == 'te':
		self.Phi_te_cca1 = Phi_cca1
		self.Phi_te_cca2 = Phi_cca2
	else:
		raise ValueError('not a valid inference mode in inf_cca()')
	return Phi_cca1, Phi_cca2


def PCCA(H1, H2, dim, rcov1, rcov2):

	N, d1 = H1.shape
	_, d2 = H2.shape

	# Remove mean.
	mu1 = np.mean(H1, axis=0, keepdims=True)
	H1 = H1 - np.tile(mu1, [N,1])

	mu2 = np.mean(H2, axis=0, keepdims=True)
	H2 = H2 - np.tile(mu2, [N,1])

	S11 = np.matmul(H1.transpose(), H1) / (N-1) + rcov1 * np.eye(d1)
	S22 = np.matmul(H2.transpose(), H2) / (N-1) + rcov2 * np.eye(d2)
	S12 = np.matmul(H1.transpose(), H2) / (N-1)

	E1, V1 = np.linalg.eig(S11)
	E2, V2 = np.linalg.eig(S22)

	# For numerical stability.
	idx1 = np.where(E1>eps_eig)[0]
	E1 = E1[idx1]
	V1 = V1[:, idx1]

	idx2 = np.where(E2>eps_eig)[0]
	E2 = E2[idx2]
	V2 = V2[:, idx2]

	K11 = np.matmul( np.matmul(V1, np.diag(np.reciprocal(np.sqrt(E1)))), V1.transpose())
	K22 = np.matmul( np.matmul(V2, np.diag(np.reciprocal(np.sqrt(E2)))), V2.transpose())
	T = np.matmul( np.matmul(K11, S12), K22)
	# print(T)
	U, E, V = np.linalg.svd(T, full_matrices=False)
	V = V.transpose()

	A = np.matmul(K11, U[:, 0:dim])
	B = np.matmul(K22, V[:, 0:dim])
	E = E[0:dim]

	M1 = M2 = np.diag(np.sqrt(E))
	W1 = np.matmul(S11, np.matmul(A, M1))
	W2 = np.matmul(S22, np.matmul(B, M2))

	Psi1 = S11 - np.matmul(W1, W1.transpose())
	Psi1 = .5 * (Psi1 + Psi1.transpose())
	Psi2 = S22 - np.matmul(W2, W2.transpose())
	Psi2 = .5 * (Psi2 + Psi2.transpose())

	return W1.transpose(), W2.transpose(), mu1, mu2, Psi1, Psi2, E

