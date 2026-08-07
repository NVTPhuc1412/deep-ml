def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	trace = matrix[0][0] + matrix[1][1] 
	det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
	delta = trace**2 - 4*det
	if delta < 0:
		return []
	eigenvalues = [
		(trace + delta**0.5)/2,
		(trace - delta**0.5)/2,
	]
	return eigenvalues