import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	logits = [0] * len(features)
	for i in range(len(features)):
		for x, w in zip(features[i], weights):
			logits[i] += x*w
		logits[i] += bias
	
	probabilities = [round(1/(1+math.exp(-logit)), 4) for logit in logits]
	mse = sum(
		(label-prob)**2
		for label, prob
		in zip(labels, probabilities)
	) / len(labels)
	
	return probabilities, mse