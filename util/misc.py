from __future__ import division, print_function
import os
import numpy as np
import json
import tensorflow as tf

# server_device = os.environ['paramserver']
#
# def variable_on_cpu(name, shape=None, initializer=None, trainable=True, dtype=None):
# 	"""Helper to create a Variable stored on CPU memory.
#
# 	Args:
# 			name: name of the variable
# 			shape: list of ints
# 			initializer: initializer for Variable
# 			trainable: boolean defining if the variable is for training
# 	Returns:
# 			Variable Tensor
# 	"""
# 	# global server_device
# 	with tf.device(server_device):
# 		var = tf.get_variable(
# 			name, shape, dtype=dtype, initializer=initializer, trainable=trainable)
# 	return var


class ResultLogger(object):
	def __init__(self, path, *args, **kwargs):
		self.f_log = open(path, 'a')
		self.f_log.write(json.dumps(kwargs) + '\n')

	def log(self, **kwargs):
		self.f_log.write(json.dumps(kwargs) + '\n')
		self.f_log.flush()

	def close(self):
		self.f_log.close()


"""
code is from: 
https://github.com/tensorflow/models/blob/master/research/real_nvp/real_nvp_multiscale_dataset.py"""

"""call vars(), which returm to get the dictionary of HParams """

class HParams_new(object):
	"""Dictionary of hyperparameters."""
	def __init__(self, **kwargs):
		self.dict_ = kwargs
		self.__dict__.update(self.dict_)

	def update_config(self, in_string):
		"""Update the dictionary with a comma separated list."""
		pairs = in_string.split(",")
		pairs = [pair.split("=") for pair in pairs]
		for key, val in pairs:
			self.dict_[key] = type(self.dict_[key])(val)
		self.__dict__.update(self.dict_)
		return self

	def __getitem__(self, key):
		return self.dict_[key]

	def __setitem__(self, key, val):
		self.dict_[key] = val
		self.__dict__.update(self.dict_)

class HParams(object):
	"""Dictionary of hyperparameters."""
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)


class struct():
	def __init__(self):
		pass


def myprint_file(log_file, verbose = 'no_log', *args):
	print(*args)
	if verbose == 'log':
		with open(log_file, 'a') as f1:
			print(*args, file=f1)


def merge_two_dicts(x, y):
	z = x.copy()   # start with x's keys and values
	z.update(y)    # modifies z with y's keys and values & returns None
	return z

def list_None2negative1(ls):
	return [-1 if x is None else x for x in ls]
