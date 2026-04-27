"""
Confidence metrics computation module.
Easy to extend with new metrics.
"""
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from types import SimpleNamespace

# Helper function for creating distribution of logprobs
def _get_topk_probs(token_data: Any, k: Optional[int] = None) -> np.ndarray:
    """Helper to convert top_logprobs (up to top_k) into probabilities."""
    if not token_data.top_logprobs:
        # Fallback if top_logprobs is not available (only main logprob is used)
        return np.array([np.exp(token_data.logprob)])

    # 1. Get the logprobs
    logprobs = np.array(sorted([lp.logprob for lp in token_data.top_logprobs]))
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

@dataclass
class MetricConfig:
    """Configuration for a confidence metric"""
    name: str
    compute_fn: Callable[[List[Any]], List[float]]
    description: str = ""
    uses_k: bool = False


class ConfidenceMetrics:
    """Computes various confidence metrics from logprobs"""
    
    def __init__(self):
        self.metrics = self._register_metrics()
    
    def _register_metrics(self) -> Dict[str, MetricConfig]:
        """Register all available metrics"""
        return {
            'mean': MetricConfig(
                name='mean',
                compute_fn=self._compute_mean_confidence,
                description='Negative mean of top-k logprobs',
                uses_k=True
            ),
            'median': MetricConfig(
                name='median',
                compute_fn=self._compute_median_confidence,
                description='Negative median of top-k logprobs',
                uses_k=True
            ),
            'variance': MetricConfig(
                name='variance',
                compute_fn=self._compute_variance_confidence,
                description='Variance of top-k logprobs',
                uses_k=True
            ),
            'gap': MetricConfig(
                name='gap',
                compute_fn=self._compute_gap_confidence,
                description='Gap between top-1 and top-2 logprobs',
                uses_k=False
            ),
            'top_prob': MetricConfig(
                name='top_prob',
                compute_fn=self._compute_top_prob_confidence,
                description='Top-1 probability',
                uses_k=False
            ),
            'entropy': MetricConfig(
                name='entropy',
                compute_fn=self._compute_entropy_uncertainty,
                description='Normalized Entropy of top-k distribution (Higher = Higher Uncertainty)',
                uses_k=True
            ),
            'gap_probs': MetricConfig(
                name='gap_probs',
                compute_fn=self._compute_gap_probs_confidence,
                description='Gap between top-1 and top-2 probabilities (Higher = Higher Confidence)',
                uses_k=False
            ),
        }
    
    def add_metric(self, metric_config: MetricConfig):
        """Add a new metric dynamically"""
        self.metrics[metric_config.name] = metric_config
    
    def compute_all_metrics(self, choice: Any, k: Optional[int] = None) -> Dict[str, List[float]]:
        
        # handle both chat completions and completions format
        if hasattr(choice, "assistant_tokens"):
            # /v1/completions format — build token_data_list from attached attributes
            token_data_list = []
            for token, logprob, top_lps in zip(
                choice.assistant_tokens,
                choice.assistant_logprobs,
                choice.assistant_top_logprobs,
            ):
                # build an object that mimics chat completions token data
                token_data_list.append(SimpleNamespace(
                    token=token,
                    logprob=logprob,
                    top_logprobs=[
                        SimpleNamespace(token=t, logprob=lp)
                        for t, lp in (top_lps or {}).items()
                    ],
                ))
        elif choice.logprobs and choice.logprobs.content:
            # /v1/chat/completions format — use as-is
            token_data_list = choice.logprobs.content
        else:
            return {name: [] for name in self.metrics.keys()}

        results = {}
        for metric_name, metric_config in self.metrics.items():
            if metric_config.uses_k:
                results[metric_name] = metric_config.compute_fn(token_data_list, k)
            else:
                results[metric_name] = metric_config.compute_fn(token_data_list)

        return results
    
    # Individual metric computation functions
    def _compute_mean_confidence(self, token_data_list: List[Any], k: Optional[int] = None) -> List[float]:
        """Mean confidence = negative average of top logprobs"""
        confidences = []
        for token_data in token_data_list:
            if token_data.top_logprobs:
                logprobs = sorted([lp.logprob for lp in token_data.top_logprobs], reverse=True)
                if k is not None:
                    logprobs = logprobs[:k]
                confidences.append(-np.mean(logprobs))
            else:
                confidences.append(-token_data.logprob)
        return confidences
    
    def _compute_median_confidence(self, token_data_list: List[Any], k: Optional[int] = None) -> List[float]:
        """Median confidence = negative median of top logprobs"""
        confidences = []
        for token_data in token_data_list:
            if token_data.top_logprobs:
                logprobs = sorted([lp.logprob for lp in token_data.top_logprobs], reverse=True)
                if k is not None:
                    logprobs = logprobs[:k]
                confidences.append(-np.median(logprobs))
            else:
                confidences.append(-token_data.logprob)
        return confidences
    
    def _compute_variance_confidence(self, token_data_list: List[Any], k: Optional[int] = None) -> List[float]:
        """Variance of top-k logprobs"""
        confidences = []
        for token_data in token_data_list:
            if token_data.top_logprobs:
                logprobs = sorted([lp.logprob for lp in token_data.top_logprobs], reverse=True)
                if k is not None:
                    logprobs = logprobs[:k]
                confidences.append(np.var(logprobs))
            else:
                confidences.append(0.0)
        return confidences
    
    def _compute_gap_confidence(self, token_data_list: List[Any]) -> List[float]:
        """Gap between top-1 and top-2 logprobs"""
        confidences = []
        for token_data in token_data_list:
            if token_data.top_logprobs and len(token_data.top_logprobs) >= 2:
                logprobs = sorted([lp.logprob for lp in token_data.top_logprobs], reverse=True)
                confidences.append(logprobs[0] - logprobs[1])
            else:
                confidences.append(0.0)
        return confidences
    
    def _compute_top_prob_confidence(self, token_data_list: List[Any]) -> List[float]:
        """Top-1 probability (The probability of the chosen token)"""
        confidences = []
        for token_data in token_data_list:
            confidences.append(np.exp(token_data.logprob))
        return confidences
    
    def _compute_entropy_uncertainty(self, token_data_list: List[Any], k: Optional[int] = None) -> List[float]:
        """
        Normalized Entropy of the top-k probability distribution 
        (Higher value = Higher Uncertainty).
        """
        uncertainties = []
        for token_data in token_data_list:
            if not token_data.top_logprobs:
                uncertainties.append(0.0)
                continue
                
            probs = _get_topk_probs(token_data, k)
            
            # Shannon Entropy H = - sum(p * log(p))
            # Exclude p=0 to avoid log(0) error
            non_zero_probs = probs[probs > 1e-12] 
            entropy = -np.sum(non_zero_probs * np.log(non_zero_probs))
            
            uncertainties.append(entropy)
            
        return uncertainties
    
    def _compute_gap_probs_confidence(self, token_data_list: List[Any]) -> List[float]:
        """Gap between top-1 and top-2 logprobs (Higher value = Higher Confidence)."""
        confidences = []
        for token_data in token_data_list:
            if token_data.top_logprobs and len(token_data.top_logprobs) >= 2:
                logprobs = sorted([lp.logprob for lp in token_data.top_logprobs], reverse=True)
                # Gap is log(P1) - log(P2)
                probs = np.exp(logprobs)
                confidences.append(probs[0] - probs[1])
            else:
                # Cannot calculate gap. A value of 0.0 is used for a neutral or
                # impossible gap calculation.
                confidences.append(0.0)
        return confidences

class AggregationStrategy:
    """Handles aggregation of confidence sequences"""
    
    @staticmethod
    def aggregate_full(sequence: List[float]) -> float:
        """Aggregate over full sequence"""
        return float(np.mean(sequence)) if sequence else 0.0
    
    @staticmethod
    def aggregate_tail(sequence: List[float], tail_n: int) -> float:
        """Aggregate over last N tokens"""
        if not sequence:
            return 0.0
        tail_seq = sequence[-tail_n:] if len(sequence) >= tail_n else sequence
        return float(np.mean(tail_seq))
    
    @staticmethod
    def aggregate_rolling_groups(sequence: List[float], group_size: int = 1024) -> List[float]:
        """
        Compute rolling window group confidences efficiently (O(n)).
        Each token's group confidence = average of [token_idx - group_size + 1 : token_idx]
        """
        if not sequence:
            return []

        arr = np.array(sequence, dtype=float)
        cumsum = np.cumsum(np.insert(arr, 0, 0))  # prefix sums, shifted by 1

        n = len(arr)
        # For index i, sum = cumsum[i+1] - cumsum[max(0, i+1-group_size)]
        starts = np.maximum(np.arange(n + 1) - group_size, 0)
        sums = cumsum[1:] - cumsum[starts[1:]]
        counts = np.minimum(np.arange(1, n + 1), group_size)
        rolling_means = sums / counts

        return rolling_means.tolist()
    
    @staticmethod
    def aggregate_groups_stats(group_confidences: List[float], bottom_percent: float = 0.1) -> Dict[str, float]:
        """Compute statistics over group confidences"""
        if not group_confidences:
            return {
                'lowest': 0.0,
                'bottom_pct': 0.0,
                'mean': 0.0,
                'top_pct': 0.0,
                'highest': 0.0
            }
        
        k = max(1, int(len(group_confidences) * bottom_percent))
        sorted_groups = np.sort(group_confidences)
        
        return {
            'lowest': float(np.min(group_confidences)),
            'highest': float(np.max(group_confidences)),
            'bottom_pct': float(np.mean(sorted_groups[:k])),
            'top_pct': float(np.mean(sorted_groups[-k:])),
            'mean': float(np.mean(group_confidences))
        }


def aggregate_metrics(
    metric_sequences: Dict[str, List[float]], 
    tail_n: int = 512,
    group_size: int = 1024,
    bottom_percent: float = 0.1
) -> Dict[str, Any]:
    """
    Aggregate all metrics with full, tail, and group-based aggregations.
    
    Returns a flat dictionary with keys like:
        - {metric_name}_full
        - {metric_name}_tail
        - {metric_name}_group_lowest
        - {metric_name}_group_bottom_pct
    """
    results = {}
    
    for metric_name, sequence in metric_sequences.items():
        if not sequence:
            results[f'{metric_name}_full'] = 0.0
            results[f'{metric_name}_tail'] = 0.0
            results[f'{metric_name}_group_lowest'] = 0.0
            results[f'{metric_name}_group_bottom_pct'] = 0.0
            results[f'{metric_name}_group_highest'] = 0.0
            results[f'{metric_name}_group_top_pct'] = 0.0
            
            continue
        
        # Full sequence aggregation
        results[f'{metric_name}_full'] = AggregationStrategy.aggregate_full(sequence)
        
        # Tail aggregation
        results[f'{metric_name}_tail'] = AggregationStrategy.aggregate_tail(sequence, tail_n)
        
        # Group-based aggregation
        group_confs = AggregationStrategy.aggregate_rolling_groups(sequence, group_size)
        group_stats = AggregationStrategy.aggregate_groups_stats(group_confs, bottom_percent)
        
        results[f'{metric_name}_group_lowest'] = group_stats['lowest']
        results[f'{metric_name}_group_bottom_pct'] = group_stats['bottom_pct']
        results[f'{metric_name}_group_highest'] = group_stats['highest']
        results[f'{metric_name}_group_top_pct'] = group_stats['top_pct']
    
    return results