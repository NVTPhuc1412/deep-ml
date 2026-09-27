import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	# Your code here
	tmp = np.zeros(
		(input_height + 2*padding, 
		input_width + 2*padding),
		dtype=float
		)
	tmp[padding:padding+input_height,padding:padding+input_width] = input_matrix
	out_height = int(np.floor((input_height + 2*padding - kernel_height)/stride)) + 1
	out_width = int(np.floor((input_width + 2*padding - kernel_width)/stride)) + 1
	output_matrix = np.zeros((out_height, out_width), dtype=float)
	for i in range(out_height):
		for j in range(out_width):
			output_matrix[i,j] = (
				tmp[i*stride : i*stride + kernel_height,
					j*stride : j*stride + kernel_width]
				* kernel).sum()
	return output_matrix
