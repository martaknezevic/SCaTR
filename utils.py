import numpy as np
from typing import Any, Tuple, Optional
from openai.types.chat.chat_completion import Choice
class CustomChoice(Choice):
    def __init__(self, rollout_idx=None, problem_id=None, **kwargs):
        super().__init__(**kwargs)
        self.rollout_idx = rollout_idx
        self.problem_id = problem_id
    
    @classmethod
    def from_dict(cls, data: dict):
        """Automatically reconstruct from dictionary"""
        return cls(**data)

# Helper function for creating distribution of logprobs
def _get_topk_probs(token_data: Any, k: Optional[int] = None) -> np.ndarray:
    """Helper to convert top_logprobs (up to top_k) into probabilities."""
    if not token_data.top_logprobs:
        # Fallback if top_logprobs is not available (only main logprob is used)
        return np.array([np.exp(token_data.logprob)])

    # 1. Get the logprobs
    logprobs = np.array([lp.logprob for lp in token_data.top_logprobs])
    if k is not None:
        logprobs = logprobs[:k]
    
    # 2. Convert to probabilities (softmax operation)
    
    # Exponentiate
    probs = np.exp(logprobs)
    
    # Normalize (ensure the local distribution sums to 1)
    sum_probs = np.sum(probs)
    if sum_probs > 0:
        probs /= sum_probs
    
    return probs


def _get_topk_probs_np(token_data: Any, k: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Helper to convert top_logprobs into normalized probabilities P and return K."""
    if not token_data.top_logprobs:
        return np.array([1.0]), 1

    logprobs = np.array([lp.logprob for lp in token_data.top_logprobs])
    K = len(logprobs)
    
    unnormalized_probs = np.exp(logprobs)
    sum_probs = np.sum(unnormalized_probs)
    
    if sum_probs > 1e-12:
        P = unnormalized_probs / sum_probs
    else:
        # Fallback to uniform if probabilities are all near-zero
        P = np.full(K, 1.0/K)
    
    return P, K