def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	n = len(vectors[0])
	m = len(vectors)
	means = [sum(vec)/len(vec) for vec in vectors]
	cov_mat = [
		[
			(sum(x*y for x,y in zip(vectors[i], vectors[j]))
			-n*means[i]*means[j])/(n-1)
			for j in range(m)
		]
		for i in range(m)
	]
	
	return cov_mat