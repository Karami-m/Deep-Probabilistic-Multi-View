"""
Functions for defining some neural network layers in tensorflow
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

__all__ = ['tensorflow_session', 'lr_scheduler', 'optimizer_variables_initializer',
           'beta_fn', 'clipped_softplus', 'clipped_sigmoid']


def tensorflow_session(hp):
	# allow_soft_placement must be set to True to build towers on GPU, as some of the ops do not have GPU implementations.
	config = tf.ConfigProto(
		allow_soft_placement=True,
		log_device_placement = False)
	config.gpu_options.allow_growth = hp.grow_GPU_mem
	sess = tf.Session(config=config)
	return sess

def lr_scheduler(lr, global_step, global_step0, lr_decay, lr_decaymin, step_size, epochs_warmup,
				 n_batches, lr_scheduler = 'exp', epoch_delay=0):
	lr1 = lr * tf.cast((global_step - global_step0), dtype=tf.float32) / float(n_batches * epochs_warmup + 1)
	if lr_scheduler == 'exp':
		lr2 = tf.train.exponential_decay(learning_rate=lr,
										  global_step= global_step - epoch_delay*n_batches,
										  decay_steps= n_batches * step_size,
										  decay_rate= lr_decay,
										  staircase=True)
		lr2 = tf.maximum(lr2, lr_decaymin * lr)
	elif lr_scheduler == 'cosine':
		lr2 = tf.train.cosine_decay_restarts(learning_rate=lr,
											 global_step=global_step - epoch_delay*n_batches,
											 first_decay_steps=n_batches * step_size,
											 t_mul=2., m_mul=lr_decay, alpha=lr_decaymin)
	elif lr_scheduler in ['none', None]:
		lr2 = lr

	return tf.minimum(lr1, lr2)

def optimizer_variables_initializer(optimizer, var_list):
	opt_vars = [optimizer.get_slot(var, name)
				 for name in optimizer.get_slot_names()
				 for var in var_list if var is not None]
	if isinstance(optimizer, tf.train.AdamOptimizer):
		opt_vars.extend(list(optimizer._get_beta_accumulators()))
	return tf.variables_initializer(opt_vars)

def beta_fn(epoch, args):
	return min([max([(epoch - args.warmupDelay + 1), 0]) * (args.max_beta - args.min_beta) / max(
		[args.warmup, 1.]) + args.min_beta, args.max_beta])


epsilon_num = 1e-5
def clipped_sigmoid(x):
	return tf.sigmoid(x)*(1.-2.*epsilon_num) + epsilon_num

def clipped_softplus(x):
	return tf.nn.softplus(x)/np.log(2.) + epsilon_num


def fully_connected(name, label, var_in, dim_in, dim_out,
                    initializer, transfer, reuse=False):
  """Standard fully connected layer"""
  with variable_scope.variable_scope(name, reuse=reuse):
    with variable_scope.variable_scope(label, reuse=reuse):
      if reuse:
        W = variable_scope.get_variable("W", [dim_in, dim_out])
        b = variable_scope.get_variable("b", [dim_out])
      else: # new
        W = variable_scope.get_variable("W", [dim_in, dim_out],
                                        initializer=initializer)
        b = variable_scope.get_variable("b", [dim_out],
                                        initializer=initializer)
  z_hat = math_ops.matmul(var_in, W)
  z_hat = nn_ops.bias_add(z_hat, b)
  y_hat = transfer(z_hat)
  return W, b, z_hat, y_hat


def convolution_2d(name, label, var_in, f, dim_in, dim_out,
                   initializer, transfer, reuse=False):
  """Standard convolutional layer"""
  with variable_scope.variable_scope(name, reuse=reuse):
    with variable_scope.variable_scope(label, reuse=reuse):
      if reuse:
        W = variable_scope.get_variable("W", [f, f, dim_in, dim_out])
        b = variable_scope.get_variable("b", [dim_out])
      else: # new
        W = variable_scope.get_variable("W", [f, f, dim_in, dim_out],
                                        initializer=initializer)
        b = variable_scope.get_variable("b", [dim_out],
                                        initializer=initializer)
  z_hat = nn_ops.conv2d(var_in, W, strides=[1,1,1,1], padding="SAME")
  z_hat = nn_ops.bias_add(z_hat, b)
  y_hat = transfer(z_hat)
  return W, b, z_hat, y_hat

