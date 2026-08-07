def compute_pruning_alphas(tree: dict) -> list:
    """
    Computes effective alpha values for cost-complexity pruning.
    
    Args:
        tree: Dictionary representing a decision tree node with keys:
              - 'samples': number of samples reaching this node
              - 'errors': misclassification count if node becomes a leaf
              - 'left': left child subtree (dict) or None
              - 'right': right child subtree (dict) or None
        
    Returns:
        List of effective alpha values for internal nodes, sorted ascending.
    """
    numerators = []
    leaf_counts = []
    curr_nodes = []
    stack = [(tree, False)]
    while stack:
        node, visited = stack.pop()

        is_leaf = (
            node['left'] is None
            and node['right'] is None
        )
        if is_leaf:
            for index in curr_nodes:  
                numerators[index] -= node['errors']
                leaf_counts[index] += 1
            continue
        
        if not visited:
            numerators.append(node['errors'])
            leaf_counts.append(-1)
            curr_nodes.append(len(numerators)-1)

            stack.append((node, True))
            if node['right'] is not None:
                stack.append((node['right'], False))
            if node['left'] is not None:
                stack.append((node['left'], False))
        if visited:
            curr_nodes.pop()

    return sorted(
        numerator/leaf_count 
        for numerator,leaf_count 
        in zip(numerators, leaf_counts)
        )