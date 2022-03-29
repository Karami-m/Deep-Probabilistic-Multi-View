from __future__ import division, print_function
import numpy as np
# from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Plot image examples.
def plot_img(imgs, title=None, savefig=False, results_subfldr='./', filenames = 'images' ):
	for img, filename in zip(imgs, filenames):
		cmap = None
		if len(img.shape) == 2:
			cmap = 'gray'
		elif (len(img.shape) == 3) and (img.shape[-1] == 1):
			cmap = 'gray'
			img = img[:,:,0]

		plt.figure()
		plt.imshow(img, interpolation='nearest', cmap = cmap)
		if title is not None:
			plt.title(title)
		plt.axis('off')
		plt.tight_layout()
		if savefig:
			plt.savefig(results_subfldr + filename, bbox_inches='tight')
			plt.close()


def border_images(images, border=1, border_color=1.):
	return [[np.pad(image, ((border, border), (border, border)), 'constant', constant_values=border_color)
			 for image in imgs] for imgs in images]

def tile_imgs(imgs_list, aspect_ratio=1.0, tile_shape=None, border=1,
			 border_color=0, stretch=False):
	imgs_list = border_images(imgs_list, border=border, border_color=border_color)
	tiled_imgs = np.concatenate(np.concatenate(imgs_list, axis=2), axis=2)
	return tiled_imgs

def tile_imgs2(imgs_list, aspect_ratio=1.0, tile_shape=None, border=1,
			 border_color=1., stretch=False):
	imgs_list = border_images(imgs_list, border=border, border_color=border_color)
	tiled_imgs = np.concatenate(np.concatenate(imgs_list, axis=1), axis=1)
	return tiled_imgs

def plot_tile_image(imgs_list, title=None, savefig=False, results_subfldr='./', filenames = 'images',
					border = 1, border_color = 1.):
	img_tiled = tile_imgs2(imgs_list, border=border, border_color=border_color)
	plot_img([img_tiled], savefig=savefig, results_subfldr=results_subfldr, filenames=[filenames], title=title)
	return





