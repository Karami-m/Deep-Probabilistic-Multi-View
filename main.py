# from __future__ import division, print_function

# # TODO:

import sys, os, time, copy

try:
	from urllib.request import urlretrieve  # Python 3
except ImportError:
	from urllib import urlretrieve  # Python 2
homepath = os.getcwd() #+ "/.."
sys.path.append(homepath)
sys.path.append(homepath+'/util')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action='store_true', help="Verbose mode")
parser.add_argument("--restore_path", type=str, default='default',
					help="Location of checkpoint to restore")
parser.add_argument("--mode", type=str, default='train', choices=['train', 'cross-validate', 'inference'],
					help="Mode of execution. Must be 'train', 'cross-validate' or 'inference'.")
parser.add_argument("--logdir", type=str,
					default='./logs/test', help="Location to save logs")
parser.add_argument("--test_logfile", type=str,
					default='test_log.txt', help="log file of test")
parser.add_argument("--tf_tensorboard", type=str, default='none', choices=['standard', 'detailed', 'none'],
					help=" 'none' 'standard' or 'detailed'. detailed will save histograms and images and projector")
parser.add_argument("--tf_save", type = str, default="save", choices=['save', 'save_graph', 'ignore'],
					help="To 'save' or 'ignore' the checkpoint and the TensorBoard, save_graph:"
						 "to save the graph")
parser.add_argument("--grow_GPU_mem", action="store_true",
					help="Allowing GPU memory growth by only allocating a subset of the available memory")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--n_gpu", type=int, default=1, help="number of GPUs")
parser.add_argument("--gpu_server", action="store_true",
					help="to use GPU as the param server or CPU.")

# method hyperparameters
parser.add_argument("--hidden_dim", type=int, default=80, help="size of total hidden dimension")
parser.add_argument("--hidden_dim_shrd", type=int, default=50, help="size of shared hidden dimension")
parser.add_argument("--n_samples", type=int, default=5, help="number of samples for Monte Carlo averaging")
parser.add_argument("--init_CCA", type = str, default="VCCA_wang", choices=['VCCA_wang', 'random', 'infere'],
					help="infere: to perform inference for multi-view model in the initialization.")
parser.add_argument('--inf_on_CPU', action='store_true', default=False,
					help='to move fx1 and fx2 to CPU for 2v-inference and SVM classification')
parser.add_argument('--fast_SVM', action='store_true', default=False,
					help='to use svm.LinearSVC instead of svm.SVC')

parser.add_argument('--warmup', type=int, default=1, metavar='N',
					help='number of epochs for warm-up. Set to 0 to turn warmup off.')
parser.add_argument('--max_beta', type=float, default=1., metavar='MB',
					help='max beta for warm-up')  # set max_beta > 2 for damped sinusuidal warmup
parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB',
					help='min beta for warm-up')
parser.add_argument('--warmupDelay', type=int, default=0, metavar='N',
					help='number of epochs for warm-up. Set to 0 to turn warmup off.')
parser.add_argument('--to_reset_optimizer', action='store_true', default=False,
					help='to reset optimizer right after the warmup is complete (only once)')
parser.add_argument("--eval_methods", type = str, default="class_NMI_ACC",
					help="evaluation downstream tasks such as for classification.")
parser.add_argument("--init_nets", type = str, default="unif", choices=['none', 'unif'],
					help="initializer for the network parameters")
parser.add_argument("--reconst", type=str, default='none',
					help="one2all: to opt for the reconstruction of all views based on first view(main view)")

# Dataset hyperparams:
parser.add_argument("--dataset", type=str, default='n-mnist',
					help="dataset (digits OR digits_fc OR ....")
parser.add_argument("--data_dir", type=str, default='',
					help="Location of data")
parser.add_argument("--image_size", type=int, default=0, help="size of images to be padded to, 0 means no extra padding")


# Optimization hyperparams:
parser.add_argument("--n_train", type=int,
					default=-1, help="Train epoch size")
parser.add_argument("--n_test", type=int, default=-1, help="Test epoch size")
parser.add_argument("--n_valid", type=int, default=-1, help="Valid epoch size")
parser.add_argument("--batch_size", type=int,
					default=100, help="Minibatch size")
parser.add_argument("--batch_size_test", type=int,
					default=-1, help="Minibatch size")
parser.add_argument("--batch_size_valid", type=int,
					default=-1, help="Minibatch size")
parser.add_argument('--n_epoch', type=int, default=100000, metavar='EPOCHS',
					help='number of epochs to train (default: 1000)')
parser.add_argument("--optimizer", type=str,
					default="AdamOptimizer", help="adam or adamax or ....")
parser.add_argument("--lr", type=float, default=0.0001,
					help="Base learning rate")
parser.add_argument("--lr_scheduler", type=str,
					default="exp", help="exp or cosine or ....")
parser.add_argument("--lr_decay", type=float, default=1.,
					help="decay rate every lr_stpsize(10) epochs")
parser.add_argument("--lr_decaymin", type=float, default=0.1,
					help="minimum decay of learning rate ")
parser.add_argument("--lr_stpsize", type=int, default=10,
					help="step size in epochs for exponential annealing OR periods size in update step for cosine restart")
parser.add_argument('--warmup_lr', type=int, default=0, metavar='WLR',
					help='number of epoches for annealing warm-up')
parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
parser.add_argument("--beta2", type=float, default=.999, help="Adam beta2")
parser.add_argument("--eps_opt", type=float, default=1e-7, help="Adam epsilon")
parser.add_argument("--momentum", type=float, default=.97, help="Momentum for MomentumOptimizer")
# parser.add_argument("--polyak_epochs", type=float, default=1,
# 					help="Nr of averaging epochs for Polyak and beta2")
parser.add_argument("--weight_decay", type=float, default=0.,
					help="Weight decay. Switched off by default.")
parser.add_argument("--eval_interval", type=int,
					default=5, help="Epochs between valid")
parser.add_argument("--save_interval", type=int,
					default=5, help="Epochs between save")
parser.add_argument('--compute_corr', action='store_true', default=False,
					help='to compute the correlation (disentangelment) after each Epoch')

parser.add_argument("--fold", type=int, default=0, help="fold index of cross validation")
parser.add_argument("--n_folds", type=int, default=6, help="number of folds for cross validation")

# Synthesis/Sampling hyperparameters:
parser.add_argument("--n_sample", type=int, default=1,
					help="minibatch size for sample")
parser.add_argument("--epochs_full_sample", type=int,
					default=50, help="Epochs between full scale sample")

# conv architecture
parser.add_argument("--hpconfig", type=str,
					default="shrd_est=z,p_drop=0.1,var_prior1=1.",
					help="A comma separated list of hyperparameters for the conv flow network. Format is "
						 "hp1=value1,hp2=value2,etc. The model will be trained with the specified hyperparameters, filling in missing hyperparameters "
						 "from the default_values in get_default_arch().")

args = parser.parse_args()

args.debug_mode = False
""" ----------temp for debug ------------"""
# args.debug_mode = True
# args.dataset= 'n-mnist'
# args.compute_corr = True
# args.eval_methods =  "class_NMI_ACC_classv1"
# args.hpconfig = "shrd_est=z12,p_drop=0.1,var_prior1=1.,var_prior2_ratio=1.,postenc1=sp,postenc2=sp"
# args.n_samples = 1
# args.logdir = "./logs/n-mnist/temp"
# args.tf_save = 'ignore'
# args.batch_size_valid = -1
# args.seed = 100
# args.restore_path = "./logs/temp/model_best_loss.ckpt"
#
# args.debug_mode = False
# args.dataset= 'n-mnist'
# args.mode = 'inference_corr' #'inference'
# args.hidden_dim, args.hidden_dim_shrd = 60, 30
# args.eval_methods =  "class_NMI_ACC_classv1" #"class_NMI_ACC"
# args.hpconfig = \
# 	"shrd_est=z12,gate=relu,p_drop=0.2,var_prior0=500.,var_prior1=4.,postenc1=sp,postenc2=sp,postdec2=sp,reg_var_wgt=0.0001"
# 	# "shrd_est=z1,gate=relu,p_drop=0.2,var_prior0=500.,var_prior1=2.,postenc1=sp,postenc2=sp,postdec2=sp,reg_var_wgt=0.0001"
# args.logdir = \
# 	"./logs_VCCA/n-mnist/K60_Ksh30/shrd_est=z12,gate=relu,p_drop=0.2,var_prior0=500.,var_prior1=4.,postenc1=sp,postenc2=sp,postdec2=sp,reg_var_wgt=0.0001/LR0002_BS200_fastSVM_LRdecay0_exp8"
# 	# "./logs_VCCA/n-mnist/K60_Ksh30/shrd_est=z1,gate=relu,p_drop=0.2,var_prior0=500.,var_prior1=2.,postenc1=sp,postenc2=sp,postdec2=sp,reg_var_wgt=0.0001/LR0002_BS200_fastSVM_exp1"
# args.restore_path = \
# 	"./logs_VCCA/n-mnist/K60_Ksh30/shrd_est=z12,gate=relu,p_drop=0.2,var_prior0=500.,var_prior1=4.,postenc1=sp,postenc2=sp,postdec2=sp,reg_var_wgt=0.0001/LR0002_BS200_fastSVM_LRdecay0_exp8/model_last.ckpt"
# 	# "./logs_VCCA/n-mnist/K60_Ksh30/shrd_est=z1,gate=relu,p_drop=0.2,var_prior0=500.,var_prior1=2.,postenc1=sp,postenc2=sp,postdec2=sp,reg_var_wgt=0.0001/LR0002_BS200_fastSVM_exp1/model_last.ckpt"
# args.tf_save = 'ignore'
# args.batch_size_valid = -1
# args.fast_SVM = True


""" ----------END temp for debug ------------"""

# global server_device or parameter server device
if args.gpu_server or (args.n_gpu == 1):
	args.server_device = '/gpu:0'
else:
	args.server_device = '/cpu:0'
print('\n parameter server is: ' + args.server_device)
os.environ['paramserver'] = args.server_device
os.environ['debugmode'] = str(args.debug_mode)

import runner_PVCCA

if __name__ == "__main__":
	runner = runner_PVCCA.Runner(args)

	if args.mode == 'train':
		# Perform training
		runner.train()
	elif args.mode == "cross-validate":
		runner.train()
	elif args.mode == 'inference':
		runner.infer()
	elif args.mode == 'inference_corr':
		runner.infer_correlation()
	else:
		raise Exception('Not a valid mode')
