import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    max_score = max(scores)
    scores_exp = [math.exp(score-max_score) for score in scores]
    denom = sum(scores_exp)
    return [score/denom for score in scores_exp]