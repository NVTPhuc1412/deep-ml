def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if mode == "row":
		col = len(matrix[0])
		means = [
			sum(row)/col for row in matrix
		]
	elif mode == "column":
		row = len(matrix)
		means = [
			sum(col)/row for col in zip(*matrix)
		]

	return means