#Jacob Gildenblat, 2015
#Implementation of edge preserving smoothing by minimizing with the Ambrosio-Tortorelli appoach
#AM scheme, using conjugate gradients
import os

import cv2, scipy
import numpy as np
import sys
import scipy
from scipy.sparse.linalg import LinearOperator
from scipy.io import savemat
import matplotlib.pyplot as plt
	
class AmbrosioTortorelliMinimizer():
	def __init__(self, img, iterations = 1, solver_maxiterations = 10, tol = 0.1, alpha = 1000, beta = 0.01, epsilon = 0.01):
		self.iterations = iterations
		self.tol = tol
		self.g = np.float64(img) / np.max(img)
		self.f = self.g
		self.edges = np.zeros(img.shape)
		self.update_gradients()		
		self.alpha, self.beta, self.epsilon = alpha, beta, epsilon
		self.add_const = self.beta / (4 * self.epsilon)
		self.multiply_const = self.epsilon * self.beta
		self.maxiter = solver_maxiterations

	def update_gradients(self):
		self.grad_x, self.grad_y = self.gradients(self.f)
		self.gradient_mag = np.power(self.grad_x, 2) + np.power(self.grad_y, 2)

	def edge_linear_operator(self, input):
		v = input.reshape(*self.g.shape)
		result = np.multiply(v, self.gradient_mag * self.alpha + self.add_const) \
				- self.multiply_const* cv2.Laplacian(v, cv2.CV_64F)

		return result.reshape(*input.shape)

	def image_linear_operator(self, input):
		f = input.reshape(*self.g.shape)

		x, y = self.gradients(f)

		result = f - 2*self.alpha * (self.calc_grad_x(np.multiply(self.edges, x)) + self.calc_grad_y(np.multiply(self.edges, y)) )
		return result.reshape(*input.shape)

	def solve_edges(self):
		size = self.g.shape[0]* self.g.shape[1]
		A = LinearOperator( (size, size), matvec = self.edge_linear_operator, dtype = np.float64)
		b = np.ones(size) * self.beta / (4 * self.epsilon)

		self.edges, info_ = scipy.sparse.linalg.cg(A, b, tol = self.tol, maxiter = self.maxiter)
		# if info_:
		# 	print(info_)
		self.edges = np.power(self.edges.reshape(*self.g.shape), 2)
		return self.edges

	def solve_image(self):
		size = self.g.shape[0]* self.g.shape[1]
		A = LinearOperator( (size, size), matvec = self.image_linear_operator, dtype = np.float64)
		b = self.g.reshape(size)

		self.f, info_ = scipy.sparse.linalg.cg(A, b, tol = self.tol, maxiter = self.maxiter)
		# if info_:
		# 	print(info_)
		self.f = self.f.reshape(*self.g.shape)
		self.update_gradients()
		return self.f

	def minimize(self):
		for i in range(0, self.iterations):
			self.solve_edges()
			self.solve_image()

		self.edges = np.power(self.edges, 0.5)
		cv2.normalize(self.f, self.f ,0,255,cv2.NORM_MINMAX)
		cv2.normalize(self.edges, self.edges ,0,255,cv2.NORM_MINMAX)
		self.f = np.uint8(self.f)
		self.edges = 255 - np.uint8(self.edges)

		return self.f, self.edges

	def calc_grad_x(self, img):
		return cv2.filter2D(img, cv2.CV_64F, np.array([[-1, 0, 1]]))

	def calc_grad_y(self, img):
		return cv2.filter2D(img, cv2.CV_64F, np.array([[-1, 0, 1]]).T)

	def gradients(self, img):
		return self.calc_grad_x(img), self.calc_grad_y(img)


def show_image(image, name):
	img = image * 1
	cv2.normalize(img, img,0,255,cv2.NORM_MINMAX)
	img = np.uint8(img)
	cv2.imshow(name, img)


def segment_images(main_ims_folder, out_dir):
	ims = [x for x in os.listdir(main_ims_folder) if x.endswith('.jpg')]
	for k_im,im in enumerate(ims):
		print(f'{k_im} im')
		# img = cv2.imread(sys.argv[1], 1)
		img = cv2.imread(os.path.join(main_ims_folder, im))
		result, edges = [], []
		for channel in cv2.split(img):
			solver = AmbrosioTortorelliMinimizer(channel, iterations = 3, tol = 0.1, solver_maxiterations = 5)
			u, v = solver.minimize()
			result.append(u)
			edges.append(v)
		lap = cv2.Laplacian(img, ddepth=cv2.CV_8UC3, ksize=3)
		u = cv2.merge(result)
		u = u[:, :, [2, 1, 0]].astype(float)

		v = np.maximum(*edges)
		# plt.imsave((rf"D:\Study\DOCS\Thesis\Images\AT\u_{k_im}.png"), u.astype(np.uint8))
		# plt.imsave((rf"D:\Study\DOCS\Thesis\Images\AT\v_{k_im}.png"), np.dstack([v,v,v]))
		# plt.imsave((rf"D:\Study\DOCS\Thesis\Images\AT\img_{k_im}.png"), img[:,:,[2,1,0]])

		v = (v-v.min())/(v.max()-v.min()+1e-14)
		v = (v > 0.5).astype(float)
		for k_ch in range(3):
			u[:, :, k_ch] += (v * lap[:, :, k_ch])
			u[:, :, k_ch] = 255 * (u[:, :, k_ch] - u[:, :, k_ch].min()) / (u[:, :, k_ch].max() - u[:, :, k_ch].min())

		# show_image(v, "edges")
		# show_image(u, "image")
		# show_image(img, "original")
		# cv2.waitKey(-1)
		# plt.figure()
		# plt.imshow(v)
		# plt.title("edges")
		# plt.set_cmap('gray')
		# plt.figure()
		# plt.imshow(u[:, :, [2, 1, 0]])
		# plt.title("images")
		# plt.figure()
		# plt.imshow(img[:, :, [2, 1, 0]])
		# plt.title("original")
		# plt.show()
		savemat(os.path.join(out_dir, f'{im[:-4]}.mat'), {'interp_sp_image':u, 'edges':v})


def main_():
	decimation = None#4
	if decimation:
		num_dec_ims = decimation ** 2

		for k_dec_im in range(num_dec_ims):
			decimation_suffix = f'_{decimation}_{k_dec_im + 1:02d}'
			ds_type = 'bsd300'
			if os.environ["COMPUTERNAME"] == 'BENNYK':
				if ds_type == 'bsd300':
					base_ims_dir = fr'C:\Study\Datasets\BSD\300\BSDS300{decimation_suffix}\images\features'
				elif ds_type == 'bsd500':
					base_ims_dir = fr'C:\Study\Datasets\BSD\500\BSDS500{decimation_suffix}\data\images\features'
			else:
				if ds_type == 'bsd300':
					main_ims_folder = fr"D:\DataSet\BSD\300\BSDS300{decimation_suffix}\images"
					base_ims_dir = fr'D:\DataSet\BSD\300\BSDS300{decimation_suffix}\images\features'
				elif ds_type == 'bsd500':
					main_ims_folder = fr"D:\DataSet\BSD\500\BSDS500{decimation_suffix}\data\images"
					base_ims_dir = fr'D:\DataSet\BSD\500\BSDS500{decimation_suffix}\data\images\features'

			for split_type in ['train', 'test']:
				out_dir = os.path.join(base_ims_dir, f'{split_type}_AT_with_sharpening{decimation_suffix}')
				if not os.path.isdir(out_dir):
					os.makedirs(out_dir)
				segment_images(os.path.join(main_ims_folder, split_type), out_dir)

	else:

		ds_type = 'bsd300'
		if os.environ["COMPUTERNAME"] == 'BENNYK':
			if ds_type == 'bsd300':
				base_ims_dir = fr'C:\Study\Datasets\BSD\300\BSDS300\images\features'
			elif ds_type == 'bsd500':
				base_ims_dir = fr'C:\Study\Datasets\BSD\500\BSDS500\data\images\features'
		else:
			if ds_type == 'bsd300':
				main_ims_folder = r"D:\DataSet\BSD\300\BSDS300\images"
				base_ims_dir = fr'D:\DataSet\BSD\300\BSDS300\images\features'
			elif ds_type == 'bsd500':
				main_ims_folder = r"D:\DataSet\BSD\500\BSDS500\data\images"
				base_ims_dir = fr'D:\DataSet\BSD\500\BSDS500\data\images\features'

		for split_type in ['train', 'test']:
			out_dir = os.path.join(base_ims_dir,f'{split_type}_AT')
			if not os.path.isdir(out_dir):
				os.makedirs(out_dir)
			segment_images(os.path.join(main_ims_folder, split_type), out_dir)


if __name__ == "__main__":
	main_()
