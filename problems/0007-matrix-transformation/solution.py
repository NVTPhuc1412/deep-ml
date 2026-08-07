import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	T_np = np.array(T)
	S_np = np.array(S)
	if T_np.shape[0] != T_np.shape[1]:
		return -1 
	if np.linalg.det(T_np) == 0:
		return -1
	if S_np.shape[0] != S_np.shape[1]:
		return -1 
	if np.linalg.det(S_np) == 0:
		return -1
	
	transformed_matrix = np.linalg.inv(T_np) @ np.array(A) @ np.array(S)

	return transformed_matrix