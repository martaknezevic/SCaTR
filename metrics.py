"""
Confidence metrics computation module.
Easy to extend with new metrics.
"""
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from utils import _get_topk_probs, _get_topk_probs_np
import math
from types import SimpleNamespace

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
            #'exp_mean': MetricConfig(
            #    name='exp_mean',
            #    compute_fn=self._compute_exp_mean_confidence,
            #    description='Exponential of mean (geometric mean in prob space)'
            #    uses_k=True
            #),
            #'distinctiveness': MetricConfig(
            #    name='distinctiveness',
            #    compute_fn=self._compute_distinctiveness_confidence,
            #    description='Z-score normalized difference between top-1 and mean of others',
            #    uses_k=True
            #),
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
            #'perplexity': MetricConfig(
            #    name='perplexity',
            #    compute_fn=self._compute_perplexity_uncertainty,
            #    description='Perplexity of top-k distribution (Higher = Higher Uncertainty)',
            #    uses_k=True
            #),
            #'kl_divergence': MetricConfig(
            #    name='kl_divergence',
            #    compute_fn=self._compute_kl_divergence_confidence,
            #    description='KL Divergence from top-k distribution to uniform (Higher = Higher Confidence)',
            #    uses_k=True
            #),
            #'renyi_divergence': MetricConfig(
            #    name='renyi_divergence',
            #    compute_fn=self._compute_renyi_divergence_uncertainty,
            #    description='Rényi Divergence (alpha=2) from top-k distribution to uniform (Higher = Higher Uncertainty)',
            #    uses_k=True
            #),
            #'fisher_rao_distance': MetricConfig(
            #    name='fisher_rao_distance',
            #    compute_fn=self._compute_fisher_rao_distance_uncertainty,
            #    description='Fisher-Rao Distance from top-k distribution to uniform (Higher = Higher Uncertainty)',
            #    uses_k=True
            #),
            #'inverse_probability': MetricConfig(
            #    name='inverse_probability',
            #    compute_fn=self.compute_inverse_probability_uncertainty,
            #    description='Inverse Probability using sampled token logprobs (Higher = Higher Uncertainty)',
            #    uses_k=False
            #),
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
    
    def _compute_exp_mean_confidence(self, token_data_list: List[Any], k: Optional[int] = None) -> List[float]:
        """Exponential of mean (geometric mean in probability space)"""
        confidences = []
        for token_data in token_data_list:
            if token_data.top_logprobs:
                logprobs = sorted([lp.logprob for lp in token_data.top_logprobs], reverse=True)
                if k is not None:
                    logprobs = logprobs[:k]
                confidences.append(np.exp(-np.mean(logprobs)))
            else:
                confidences.append(np.exp(-token_data.logprob))
        return confidences
    
    def _compute_distinctiveness_confidence(self, token_data_list: List[Any], k: Optional[int] = None) -> List[float]:
        """Z-score normalized difference between top-1 and mean of others"""
        confidences = []
        for token_data in token_data_list:
            if token_data.top_logprobs and len(token_data.top_logprobs) > 1:
                logprobs = sorted([lp.logprob for lp in token_data.top_logprobs], reverse=True)
                if k is not None:
                    logprobs = logprobs[:k]
                top1 = logprobs[0]
                others = logprobs[1:]
                mean_others = np.mean(others)
                std_all = np.std(logprobs) if np.std(logprobs) > 0 else 1e-9
                distinctiveness = (top1 - mean_others) / std_all
                confidences.append(np.clip(distinctiveness, -10, 10))
            else:
                confidences.append(0.0)
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
    
    def _compute_perplexity_uncertainty(self, token_data_list: List[Any], k: Optional[int] = None) -> List[float]:
        """
        Perplexity (2^Entropy) of the top-k distribution 
        (Higher value = Higher Uncertainty).
        Perplexity roughly estimates the effective number of choices the model has.
        """
        uncertainties = []
        # Reuse the entropy calculation
        entropy_values = self._compute_entropy_uncertainty(token_data_list, k)
        
        for entropy in entropy_values:
            # Perplexity is 2^H
            uncertainties.append(2**entropy)
            
        return uncertainties
    
    def _compute_kl_divergence_confidence(self, token_data_list: List[Any], k: Optional[int] = None) -> List[float]:
        """
        Computes the Kullback-Leibler (KL) Divergence from the token's top-K 
        normalized distribution (P) to a uniform distribution (Q).
        
        Formula: D_KL(P || Q) = log(K) - H(P)
        (Higher value = Higher Confidence/Peakedness)
        
        Args:
            token_data_list: A list of token logprob objects for a single generation.
            
        Returns:
            A list of float values representing the KL divergence for each token.
        """
        confidences = []

        for token_data in token_data_list:
            if not token_data.top_logprobs:
                # If only the main logprob is available, the distribution is singular (P=1)
                # H(P) = 0, D_KL = 0 (since K=1, log(K)=0)
                confidences.append(0.0)
                continue
                
            # 1. Prepare P (Normalized Probabilities)
            
            # Get the logprobs and K (number of top choices)
            logprobs = np.array(sorted([lp.logprob for lp in token_data.top_logprobs], reverse=True))
            if k is not None:
                logprobs = logprobs[:k]
            K = len(logprobs)
            
            # Convert log-probabilities to a normalized probability distribution P (local softmax)
            # Exponentiate
            unnormalized_probs = np.exp(logprobs) 
            
            # Normalize (ensure P sums to 1 over the top-K set)
            sum_probs = np.sum(unnormalized_probs)
            if sum_probs > 1e-12:  # Avoid division by zero/near-zero sum
                P = unnormalized_probs / sum_probs
            else:
                confidences.append(0.0)
                continue

            # 2. Compute Shannon Entropy H(P)
            # H(P) = -sum(P * log(P))
            # Use np.log for natural log, and filter out near-zero probabilities
            non_zero_P = P[P > 1e-12]
            H_P = -np.sum(non_zero_P * np.log(non_zero_P))

            # 3. Compute KL Divergence D_KL(P || Q)
            # D_KL(P || Q) = log(K) - H(P)
            log_K = np.log(K) # Natural log of K
            
            kl_divergence = log_K - H_P
            
            confidences.append(kl_divergence)
            
        return confidences

    def compute_inverse_probability_uncertainty(self, token_data_list: List[Any]) -> List[float]:
        """
        Computes the Inverse Probability of the sampled token (1/P_l) for each token 
        in the sequence.
        
        Metric calculated per token: V_l = exp(-log(P(y_l | y_<l)))
        (Higher value = Higher Uncertainty / Lower Confidence)

        Args:
            token_data_list: A list of token logprob objects for a single generation.
                            (Assumed to be the 'content' field from logprobs)
            
        Returns:
            A list of float values representing the Inverse Probability for each token.
        """
        
        if not token_data_list:
            return []
            
        # Extract the log probability of the sampled token for each step
        try:
            # log_probs[l] = log(P(y_l | y_<l))
            log_probs = np.array([token_data.logprob for token_data in token_data_list])
        except AttributeError:
            print("Error: Input list does not contain objects with a 'logprob' attribute.")
            return [0.0] * len(token_data_list)

        # 1. Calculate the Inverse Probability for each token: exp(-log P) = 1/P
        # This results in a NumPy array of size L (sequence length)
        inverse_probs_per_token = np.exp(-log_probs)
        
        # Return as a standard Python list of floats
        return inverse_probs_per_token.tolist()
        

    def _compute_renyi_divergence_uncertainty(self, token_data_list: List[Any], k: Optional[int] = None, alpha: float = 2.0) -> List[float]:
        """
        Computes the per-token Rényi Divergence (alpha=2.0 by default) 
        from the normalized P distribution to a uniform Q distribution over K.
        (Higher value = Higher Uncertainty)
        """
        if alpha <= 0.0 or alpha == 1.0:
            raise ValueError("Alpha must be > 0 and not equal to 1 for Rényi Divergence.")

        uncertainties = []
        for token_data in token_data_list:
            P, K = _get_topk_probs_np(token_data, k)
            
            # If K=1, the distribution is singular, divergence is 0.
            if K <= 1:
                uncertainties.append(0.0)
                continue
                
            # Calculate the sum term: sum(P_i^alpha)
            sum_p_alpha = np.sum(P**alpha)
            
            # URD_l = log(K) + (1 / (alpha - 1)) * log(sum(P_i^alpha))
            term1 = math.log(K)
            term2 = (1.0 / (alpha - 1.0)) * math.log(sum_p_alpha)
            
            uncertainties.append(term1 + term2)
            
        return uncertainties

# ----------------------------------------------------------------------------

    def _compute_fisher_rao_distance_uncertainty(self, token_data_list: List[Any], k: Optional[int] = None) -> List[float]:
        """
        Computes the per-token Fisher-Rao based distance (as defined in the query)
        from the normalized P distribution to a uniform Q distribution over K.
        (Higher value = Higher Distance/Uncertainty)
        """
        uncertainties = []
        for token_data in token_data_list:
            P, K = _get_topk_probs_np(token_data, k)
            
            # If K=1, the distance is 0.
            if K <= 1:
                uncertainties.append(0.0)
                continue
                
            # 1. Calculate the cosine similarity term: sum(sqrt(P_i) * sqrt(Q_i))
            # Since sqrt(Q_i) = 1/sqrt(K)
            sum_sqrt_p = np.sum(np.sqrt(P))
            cosine_term = (1.0 / math.sqrt(K)) * sum_sqrt_p
            
            # Ensure argument to arccos is within [-1, 1] due to floating point error
            cosine_term = np.clip(cosine_term, -1.0, 1.0)
            
            # 2. Calculate UFR_l = (2 / pi) * arccos(cosine_term)
            distance = (2.0 / math.pi) * math.acos(cosine_term)
            
            uncertainties.append(distance)
            
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