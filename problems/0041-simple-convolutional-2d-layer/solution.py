import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	# Your code here
	padded = np.pad(input_matrix, padding)
	
	windows = np.lib.stride_tricks.sliding_window_view(
		padded,
		(kernel_height, kernel_width)
	)[::stride, ::stride]

	return np.einsum('ijmn,mn->ij', windows, kernel)
