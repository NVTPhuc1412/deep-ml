import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	new_row = new_shape[0]
	new_col = new_shape[1]
	if new_row*new_col != len(a)*len(a[0]):
		return []
	flatten = [item for row in a for item in row]
	reshaped_matrix = [] 
	for i in range(new_row):
		reshaped_matrix.append(
			flatten[i*new_col:i*new_col+new_col]
			)
	return reshaped_matrix