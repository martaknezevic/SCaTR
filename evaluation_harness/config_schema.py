"""
Configuration schema for evaluation harness.

Defines the structure of config files and provides validation.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import yaml
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig


@dataclass
class DataConfig:
    """Configuration for data loading"""
    train_data_path: Optional[str] = None  # Path to training dataset or dataset name
    external_test_data_paths: Union[str, List[str]] = ""  # Path(s) to external test dataset(s)
    
    # Data loading parameters
    num_problems: Optional[int] = None  # Limit number of problems to load
    num_workers: int = 4  # Number of parallel workers for loading
    num_choice_workers: int = 4  # Workers for processing choices
    use_cache: bool = True  # Use cached data if available
    cache_dir: Optional[str] = "cache"  # Cache directory
    subsample_size: Optional[int] = None  # Subsample data after loading
    
    # Train/val/test split parameters (only used if train_data_path is provided)
    num_train_problems: Optional[int] = None  # Number of problems for training
    num_val_problems: Optional[int] = None  # Number of problems for validation
    num_test_problems: Optional[int] = None  # Number of problems for internal test
    random_seed: int = 42  # Random seed for splitting
    
    # Feature extraction parameters
    tail_tokens: int = 256  # Number of tail tokens for features


@dataclass
class MethodConfig:
    """Configuration for a single method"""
    name: str  # Method name (must be registered in registry)
    params: Dict[str, Any] = field(default_factory=dict)  # Method-specific parameters
    
    def get_method_id(self) -> str:
        """Generate unique ID for this method instance"""
        # Include key params in ID for uniqueness
        param_str = "_".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        if param_str:
            return f"{self.name}_{param_str}"
        return self.name


@dataclass
class OutputConfig:
    """Configuration for output handling"""
    base_dir: str = "outputs"  # Base output directory
    save_predictions: bool = True  # Save predictions to JSONL
    save_models: bool = True  # Save trained models
    save_metrics: bool = True  # Save evaluation metrics


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases logging"""
    enabled: bool = False  # Enable wandb logging
    project: Optional[str] = None  # WandB project name
    entity: Optional[str] = None  # WandB entity/team name
    name: Optional[str] = None  # Run name (auto-generated if None)
    tags: List[str] = field(default_factory=list)  # Tags for the run


@dataclass
class EvaluationConfig:
    """Main configuration for evaluation harness"""
    # Core configurations
    data: DataConfig
    methods: List[MethodConfig]
    output: OutputConfig
    wandb: WandBConfig
    
    # Evaluation settings
    evaluation: str = "math"  # Type of evaluation: math, code
    device: str = "cuda"  # Device for training: cuda, cpu
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'EvaluationConfig':
        """
        Load configuration from YAML file
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            EvaluationConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EvaluationConfig':
        """
        Create config from dictionary
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            EvaluationConfig instance
        """
        # Parse data config
        data_config = DataConfig(**config_dict.get('data', {}))
        
        # Parse method configs
        methods = []
        for method_dict in config_dict.get('methods', []):
            method_config = MethodConfig(
                name=method_dict['name'],
                params=method_dict.get('params', {})
            )
            methods.append(method_config)
        
        # Parse output config
        output_config = OutputConfig(**config_dict.get('output', {}))
        
        # Parse wandb config
        wandb_dict = config_dict.get('wandb', {})
        wandb_config = WandBConfig(**wandb_dict) if wandb_dict else WandBConfig()
        
        # Create main config
        return cls(
            data=data_config,
            methods=methods,
            output=output_config,
            wandb=wandb_config,
            evaluation=config_dict.get('evaluation', 'math'),
            device=config_dict.get('device', 'cuda')
        )
    
    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> 'EvaluationConfig':
        """
        Create config from OmegaConf DictConfig (for Hydra compatibility)
        
        Args:
            cfg: OmegaConf DictConfig
            
        Returns:
            EvaluationConfig instance
        """
        # Convert to regular dict
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary
        
        Returns:
            Configuration dictionary
        """
        return {
            'data': {
                'train_data_path': self.data.train_data_path,
                'external_test_data_paths': self.data.external_test_data_paths,
                'num_problems': self.data.num_problems,
                'num_workers': self.data.num_workers,
                'num_choice_workers': self.data.num_choice_workers,
                'use_cache': self.data.use_cache,
                'cache_dir': self.data.cache_dir,
                'subsample_size': self.data.subsample_size,
                'num_train_problems': self.data.num_train_problems,
                'num_val_problems': self.data.num_val_problems,
                'num_test_problems': self.data.num_test_problems,
                'random_seed': self.data.random_seed,
                'tail_tokens': self.data.tail_tokens,
            },
            'methods': [
                {
                    'name': m.name,
                    'params': m.params
                }
                for m in self.methods
            ],
            'output': {
                'base_dir': self.output.base_dir,
                'save_predictions': self.output.save_predictions,
                'save_models': self.output.save_models,
                'save_metrics': self.output.save_metrics,
            },
            'evaluation': self.evaluation,
            'device': self.device,
        }
    
    def to_yaml(self, yaml_path: Path):
        """
        Save configuration to YAML file
        
        Args:
            yaml_path: Path to save YAML file
        """
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def validate(self):
        """Validate configuration"""
        # Check at least one method is specified
        if not self.methods:
            raise ValueError("At least one method must be specified")
        
        # Check evaluation type is valid
        if self.evaluation not in ['math', 'code']:
            raise ValueError(f"Invalid evaluation type: {self.evaluation}. Must be 'math' or 'code'")
        
        # Validate train/val/test split if train data provided
        if self.data.train_data_path:
            if self.data.num_train_problems is None or self.data.num_val_problems is None or self.data.num_test_problems is None:
                raise ValueError("num_train_problems, num_val_problems, and num_test_problems must be specified when train_data_path is provided")
