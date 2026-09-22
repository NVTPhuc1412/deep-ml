def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:
    """
    Evaluate state-value function for a policy on a 5x5 gridworld.
    
    Args:
        policy: dict mapping (row, col) to action probability dicts
        gamma: discount factor
        threshold: convergence threshold
    Returns:
        5x5 list of floats
    """
    # Your code here
    values = [[0.0 for _ in range(5)] for _ in range(5)]
    terminal_states = [
        (0, 0), (0, 4),
        (4, 0), (4, 4)
        ]
    actions_changes = {
        'up': (0, -1),
        'down': (0, 1),
        'left': (-1, 0),
        'right': (1, 0)
        }
    while True:
        new_values = [[0.0 for _ in range(5)] for _ in range(5)]
        max_change = 0

        for row in range(5):
            for col in range(5):
                if (row, col) in terminal_states:
                    new_values[row][col] = 0.0
                    continue
                
                state_value = 0.0

                for action, prob in policy[(row, col)].items():
                    dr, dc = actions_changes[action]
                    next_row = row + dr
                    next_col = col + dc
                    if not (
                        0 <= next_row < 5
                        and
                        0 <= next_col < 5
                    ):
                        next_row = row
                        next_col = col
                    
                    reward = -1
                    state_value += prob * (
                        reward + 
                        gamma * values[next_row][next_col]
                        )
                
                new_values[row][col] = state_value
                
                change = abs(values[row][col]-state_value)
                max_change = max(change, max_change)

        values = new_values

        if max_change < threshold:
            break
    
    return values
                