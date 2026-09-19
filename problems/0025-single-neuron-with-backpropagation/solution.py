import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	updated_weights = initial_weights
	updated_bias = initial_bias
	mse_values = []
	sigmoid = lambda x: 1/(1+np.exp(-x))
	for epoch in range(epochs):
		z = np.dot(features, updated_weights) + updated_bias
		pred = sigmoid(z)
		error = (pred - labels)
		de_dpred = pred * (1-pred)
		dw = learning_rate * 2 * np.mean(de_dpred * error * features.T, axis=1) 
		db = learning_rate * 2 * np.mean(de_dpred * error)
		updated_weights = updated_weights - dw
		updated_bias = updated_bias - db
		mse_values.append(np.mean(error**2)) 

	return updated_weights, updated_bias, mse_values