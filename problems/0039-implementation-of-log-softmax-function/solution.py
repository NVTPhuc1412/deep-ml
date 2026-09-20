import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	np_scores = np.asarray(scores)
	scores_max = np_scores.max()
	log_sum = np.log(np.exp(np_scores).sum())
	return np_scores - log_sum
	