import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	input_sequence = np.asarray(input_sequence, dtype=float)
	Wx = np.asarray(Wx, dtype=float)
	Wh = np.asarray(Wh, dtype=float)
	b = np.asarray(b, dtype=float)

	curr_h = np.asarray(initial_hidden_state, dtype=float)
	for i in range(input_sequence.shape[0]):
		curr_h = np.tanh(
			Wx @ input_sequence[i]
			+ Wh @ curr_h
			+ b
		)
	return curr_h