"""
Profiler utilities for training with accurate LoRA FLOPs estimation.

LoRA training FLOPs formula:
- Forward: forward_flops (measured via FlopCounterMode)
- Backward activation grads: forward_flops (same compute as forward)  
- Backward weight grads: forward_flops * trainable_ratio
- Total: forward_flops * (2 + trainable_ratio)
"""

import json
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def count_parameters(model) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_forward_flops(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> int:
    """Measure forward FLOPs using FlopCounterMode."""
    model.eval()
    with torch.no_grad():
        with FlopCounterMode(display=False) as fcm:
            if attention_mask is not None:
                model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                model(input_ids=input_ids)
    model.train()
    return fcm.get_total_flops()


class FlopsTimingCallback(TrainerCallback):
    """
    Callback that measures FLOPs via FlopCounterMode and tracks timing.
    
    For LoRA: training_flops = forward_flops * (2 + trainable_ratio)
    For full: training_flops = forward_flops * 3
    """
    
    def __init__(self, model, output_dir: str, profile_step: int = 5):
        self.model = model
        self.output_dir = Path(output_dir)
        self.profile_step = profile_step
        
        # Parameter counts
        self.total_params, self.trainable_params = count_parameters(model)
        self.trainable_ratio = self.trainable_params / self.total_params
        self.is_lora = self.trainable_ratio < 0.5
        
        # FLOPs (set via measure_with_sample)
        self.forward_flops_per_token: Optional[float] = None
        self.training_flops_per_token: Optional[float] = None
        self.measured_seq_len: Optional[int] = None
        
        # Timing
        self.step_start_time: Optional[float] = None
        self.train_start_time: Optional[float] = None
        self.step_times: list[float] = []
        self.step_tokens: list[int] = []
        self.cumulative_flops: int = 0
        
        print(f"\n{'─' * 60}")
        print(f"FlopsTimingCallback: {'LoRA' if self.is_lora else 'Full'} training")
        print(f"  Total params:     {self.total_params:,} ({self.total_params / 1e9:.2f}B)")
        print(f"  Trainable params: {self.trainable_params:,} ({self.trainable_params / 1e6:.2f}M)")
        print(f"  Trainable ratio:  {100 * self.trainable_ratio:.4f}%")
        print(f"{'─' * 60}\n")
    
    def measure_with_sample(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Measure FLOPs using a sample input. Call before training."""
        batch_size, seq_len = input_ids.shape
        self.measured_seq_len = seq_len
        
        forward_flops = measure_forward_flops(self.model, input_ids, attention_mask)
        total_tokens = batch_size * seq_len
        
        self.forward_flops_per_token = forward_flops / total_tokens
        # LoRA: fwd + bwd_activation + bwd_weight = forward * (2 + trainable_ratio)
        multiplier = (2 + self.trainable_ratio) if self.is_lora else 3
        self.training_flops_per_token = self.forward_flops_per_token * multiplier
        
        print(f"FLOPs measured (batch {batch_size} × {seq_len}):")
        print(f"  Forward FLOPs:   {forward_flops / 1e12:.4f} TFLOPs")
        print(f"  Training/token:  {self.training_flops_per_token:,.0f}")
        if self.is_lora:
            savings = 100 * (1 - multiplier / 3)
            print(f"  Savings vs full: {savings:.1f}%")
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.train_start_time = time.perf_counter()
        self.step_times = []
        self.step_tokens = []
        self.cumulative_flops = 0
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.step_start_time = time.perf_counter()
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.step_start_time is None:
            return
        
        step_time = time.perf_counter() - self.step_start_time
        self.step_times.append(step_time)
        
        # Tokens this step
        seq_len = self.measured_seq_len or 2048
        tokens = args.per_device_train_batch_size * args.gradient_accumulation_steps * seq_len
        self.step_tokens.append(tokens)
        
        # FLOPs this step
        if self.training_flops_per_token is not None:
            step_flops = int(tokens * self.training_flops_per_token)
        else:
            # Fallback estimate
            multiplier = (2 + self.trainable_ratio) if self.is_lora else 3
            step_flops = int(tokens * 2 * self.total_params * multiplier)
        self.cumulative_flops += step_flops
        
        if state.global_step == self.profile_step:
            tflops = step_flops / step_time / 1e12
            print(f"\nStep {state.global_step}: {step_time:.3f}s, {tokens:,} tokens, {tflops:.2f} TFLOPs/s")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.train_start_time:
            return
        
        total_time = time.perf_counter() - self.train_start_time
        total_tokens = sum(self.step_tokens)
        avg_tflops = self.cumulative_flops / total_time / 1e12 if total_time > 0 else 0
        
        results = {
            "training_type": "LoRA" if self.is_lora else "Full",
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "trainable_ratio": self.trainable_ratio,
            "forward_flops_per_token": self.forward_flops_per_token,
            "training_flops_per_token": self.training_flops_per_token,
            "total_steps": state.global_step,
            "total_time_seconds": total_time,
            "total_tokens": total_tokens,
            "cumulative_flops": self.cumulative_flops,
            "cumulative_pflops": self.cumulative_flops / 1e15,
            "avg_tflops": avg_tflops,
        }
        
        print(f"\n{'═' * 60}")
        print(f"Training Complete: {state.global_step} steps, {total_time:.1f}s")
        print(f"  Total FLOPs: {self.cumulative_flops / 1e15:.4f} PFLOPs")
        print(f"  Throughput:  {avg_tflops:.2f} TFLOPs/s")
        if self.is_lora and self.forward_flops_per_token:
            full_flops = self.forward_flops_per_token * 3 * total_tokens
            savings = 100 * (1 - self.cumulative_flops / full_flops)
            print(f"  Savings:     {savings:.1f}% vs full fine-tuning")
            results["flops_savings_percent"] = savings
        print(f"{'═' * 60}\n")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "flops_profile.json", "w") as f:
            json.dump(results, f, indent=2)
