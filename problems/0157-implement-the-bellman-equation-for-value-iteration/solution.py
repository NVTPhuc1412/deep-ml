import numpy as np

def bellman_update(V, transitions, gamma):
    """
    Perform one step of value iteration using the Bellman equation.
    Args:
      V: np.ndarray, state values, shape (n_states,)
      transitions: list of dicts. transitions[s][a] is a list of (prob, next_state, reward, done)
      gamma: float, discount factor
    Returns:
      np.ndarray, updated state values
    """
    # TODO: Implement Bellman update
    
    new_V = np.zeros(V.shape)

    for s in range(len(transitions)):
      action_values = []
      
      for a, outcomes in transitions[s].items():
        value = 0.0
        for prob, next_state, reward, done in outcomes:
          value += prob * (
            reward + gamma * V[next_state] * (not done)
            )
        
        action_values.append(value)
      new_V[s] = max(action_values)

    return new_V