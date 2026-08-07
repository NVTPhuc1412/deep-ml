def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	return [
		[scalar*element for element in row]
		for row in matrix
	]