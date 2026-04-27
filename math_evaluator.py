"""
Modular Math Evaluation Framework

Supports AIME, MATH, and custom datasets through pluggable components.

Key Features:
- Abstract base classes for dataset loaders, answer extractors, and graders
- Factory pattern for component instantiation from config
- Configuration-driven (YAML) with all parameters externalized

Usage:
    python math_evaluator.py --config configs/config_aime.yaml
    python math_evaluator.py --config configs/config_math.yaml --n_gen 20
"""

import time
import openai
import numpy as np
import asyncio
import json
import random
import re
import yaml
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
import argparse

# Import existing components
from metrics import ConfidenceMetrics, aggregate_metrics
from storage import ChoiceStorage, MetricsStorage, ResultsAggregator

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

try:
    from grading.grader import grade_answer
    GRADING_AVAILABLE = True
except ImportError:
    GRADING_AVAILABLE = False
    print("Warning: grading module not available.")


# ============================================================================
# Helper Functions
# ============================================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content from \boxed{...} with proper nested brace handling.
    
    Args:
        text: Text containing \boxed{...} notation
        
    Returns:
        The content inside the last \boxed{...}, or None if not found
    """
    # Find the last occurrence of \boxed{
    last_boxed_start = text.rfind('\\boxed{')
    if last_boxed_start == -1:
        return None
    
    # Start after '\boxed{'
    start_pos = last_boxed_start + 7
    brace_count = 1
    pos = start_pos
    
    # Track brace depth to find matching closing brace
    while pos < len(text) and brace_count > 0:
        if text[pos] == '{' and (pos == 0 or text[pos-1] != '\\'):
            brace_count += 1
        elif text[pos] == '}' and (pos == 0 or text[pos-1] != '\\'):
            brace_count -= 1
        pos += 1
    
    if brace_count == 0:
        # Found matching closing brace
        return text[start_pos:pos-1].strip()
    
    return None


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class MathProblem:
    """Universal structure for math problems"""
    problem_id: str
    problem_text: str
    answer: Union[str, int, float]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ResponseResult:
    """Structure for response evaluation"""
    response_text: str
    extracted_answer: Union[str, int, float, None]
    is_correct: bool
    total_tokens: int
    choice_index: int
    aggregated_metrics: Dict[str, float]


# ============================================================================
# Abstract Base Classes
# ============================================================================

class DatasetLoader(ABC):
    """Abstract base class for dataset loaders"""
    
    @abstractmethod
    def load(self, dataset_source: str, split: str = "test", **kwargs) -> List[MathProblem]:
        """Load dataset and return list of MathProblem objects"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this loader"""
        pass


class AnswerExtractor(ABC):
    """Abstract base class for answer extraction"""
    
    @abstractmethod
    def extract(self, response_text: str) -> Union[str, int, float, None]:
        """Extract answer from model response"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this extractor"""
        pass


class AnswerGrader(ABC):
    """Abstract base class for answer grading"""
    
    @abstractmethod
    def grade(self, extracted_answer: Union[str, int, float, None], 
              ground_truth: Union[str, int, float]) -> bool:
        """Grade extracted answer against ground truth"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this grader"""
        pass


# ============================================================================
# Dataset Loaders
# ============================================================================

class AIMELoader(DatasetLoader):
    """Loader for AIME datasets"""
    
    def get_name(self) -> str:
        return "AIME"
    
    def load(self, dataset_source: str, split: str = "test", **kwargs) -> List[MathProblem]:
        """Load AIME dataset from HuggingFace or local JSON"""
        
        # Try HuggingFace first
        if HF_AVAILABLE and not dataset_source.endswith('.json'):
            try:
                print(f"Loading AIME dataset from HuggingFace: {dataset_source}")
                dataset = load_dataset(dataset_source, split=split)
                
                problems = []
                for i, item in enumerate(dataset):
                    problem_text = self._extract_field(item, ['problem', 'question', 'text', 'Problem'])
                    answer = self._extract_field(item, ['answer', 'solution', 'Answer'])
                    problem_id = self._extract_field(item, ['id', 'problem_id', 'ID'], default=f"aime_{i}")
                    
                    if problem_text is None or answer is None:
                        continue
                    
                    # Convert answer to integer
                    if isinstance(answer, str):
                        answer_match = re.search(r'\d+', str(answer))
                        if answer_match:
                            answer = int(answer_match.group())
                        else:
                            continue
                    
                    problems.append(MathProblem(
                        problem_id=str(problem_id),
                        problem_text=str(problem_text),
                        answer=int(answer),
                        metadata={'year': item.get('year', 'unknown')}
                    ))
                
                print(f"Loaded {len(problems)} AIME problems")
                return problems
                
            except Exception as e:
                print(f"Failed to load from HF: {e}")
        
        # Try local JSON or JSONL
        if dataset_source.endswith(('.json', '.jsonl')):
            problems = []
            
            if dataset_source.endswith('.jsonl'):
                # Load JSONL (one JSON object per line)
                with open(dataset_source, 'r') as f:
                    data = [json.loads(line) for line in f]
            else:
                # Load JSON
                with open(dataset_source, 'r') as f:
                    data = json.load(f)
            
            for item in data:
                problems.append(MathProblem(
                    problem_id=item.get("id", item.get("problem_id", f"problem_{len(problems)}")),
                    problem_text=item.get("problem", item.get("problem_text", "")),
                    answer=int(item["answer"]),
                    metadata={'year': item.get('year', 'unknown')}
                ))
            
            file_type = "JSONL" if dataset_source.endswith('.jsonl') else "JSON"
            print(f"Loaded {len(problems)} AIME problems from {file_type}")
            return problems
        
        raise ValueError(f"Could not load AIME dataset from {dataset_source}")
    
    def _extract_field(self, item: Dict, possible_keys: List[str], default=None):
        """Extract field from item dict using possible key names"""
        for key in possible_keys:
            if key in item and item[key] is not None:
                return item[key]
        return default


class MATHLoader(DatasetLoader):
    """Loader for MATH dataset (Hendrycks et al.)"""
    
    def get_name(self) -> str:
        return "MATH"
    
    def load(self, dataset_source: str, split: str = "test", **kwargs) -> List[MathProblem]:
        """Load MATH dataset"""
        
        if HF_AVAILABLE:
            try:
                print(f"Loading MATH dataset from HuggingFace: {dataset_source}")
                dataset = load_dataset(dataset_source, split=split)
                
                problems = []
                for i, item in enumerate(dataset):
                    problem_text = self._extract_field(item, ['problem', 'question'])
                    answer = self._extract_field(item, ['solution', 'answer'])
                    problem_id = self._extract_field(item, ['id', 'problem_id'], default=f"math_{i}")
                    level = self._extract_field(item, ['level'], default='')
                    subject = self._extract_field(item, ['type', 'subject'], default='')
                    
                    if problem_text is None or answer is None:
                        continue
                    
                    # Extract boxed answer from solution if needed
                    if '\\boxed{' in str(answer):
                        boxed_answer = extract_boxed_answer(str(answer))
                        if boxed_answer:
                            answer = boxed_answer
                    
                    problems.append(MathProblem(
                        problem_id=str(problem_id),
                        problem_text=str(problem_text),
                        answer=str(answer),
                        metadata={'level': level, 'subject': subject}
                    ))
                
                print(f"Loaded {len(problems)} MATH problems")
                return problems
                
            except Exception as e:
                print(f"Failed to load MATH dataset: {e}")
        
        # Try local JSON or JSONL
        if dataset_source.endswith(('.json', '.jsonl')):
            problems = []
            
            if dataset_source.endswith('.jsonl'):
                # Load JSONL (one JSON object per line)
                with open(dataset_source, 'r') as f:
                    data = [json.loads(line) for line in f]
            else:
                # Load JSON
                with open(dataset_source, 'r') as f:
                    data = json.load(f)
            
            for i, item in enumerate(data):
                problem_text = self._extract_field(item, ['problem', 'question'])
                answer = self._extract_field(item, ['solution', 'answer'])
                problem_id = self._extract_field(item, ['id', 'problem_id'], default=f"math_{i}")
                level = self._extract_field(item, ['level'], default='')
                subject = self._extract_field(item, ['type', 'subject'], default='')
                
                if problem_text is None or answer is None:
                    continue
                
                # Extract boxed answer from solution if needed
                if '\\boxed{' in str(answer):
                    boxed_answer = extract_boxed_answer(str(answer))
                    if boxed_answer:
                        answer = boxed_answer
                
                problems.append(MathProblem(
                    problem_id=str(problem_id),
                    problem_text=str(problem_text),
                    answer=str(answer),
                    metadata={'level': level, 'subject': subject}
                ))
            
            file_type = "JSONL" if dataset_source.endswith('.jsonl') else "JSON"
            print(f"Loaded {len(problems)} MATH problems from {file_type}")
            return problems
        
        raise ValueError("HuggingFace datasets not available")
    
    def _extract_field(self, item: Dict, possible_keys: List[str], default=None):
        for key in possible_keys:
            if key in item and item[key] is not None:
                return item[key]
        return default


class GenericLoader(DatasetLoader):
    """Generic loader for custom datasets"""
    
    def get_name(self) -> str:
        return "Generic"
    
    def load(self, dataset_source: str, split: str = "test", **kwargs) -> List[MathProblem]:
        """Load generic dataset from JSON or JSONL"""
        
        if dataset_source.endswith('.jsonl'):
            # Load JSONL (one JSON object per line)
            with open(dataset_source, 'r') as f:
                data = [json.loads(line) for line in f]
        else:
            # Load JSON
            with open(dataset_source, 'r') as f:
                data = json.load(f)
        
        problems = []
        for i, item in enumerate(data):
            problems.append(MathProblem(
                problem_id=item.get('id', item.get('problem_id', f"problem_{i}")),
                problem_text=item.get('problem', item.get('problem_text', '')),
                answer=item['answer'],
                metadata=item.get('metadata', {})
            ))
        
        file_type = "JSONL" if dataset_source.endswith('.jsonl') else "JSON"
        print(f"Loaded {len(problems)} problems from {file_type}")
        return problems


# ============================================================================
# Answer Extractors
# ============================================================================

class AIMEExtractor(AnswerExtractor):
    """Extractor for AIME integer answers (0-999)"""
    
    def get_name(self) -> str:
        return "AIME"
    
    def extract(self, response_text: str) -> Optional[int]:
        """Extract integer answer (0-999 for AIME)"""
        if not response_text:
            return None
        
        patterns = [
            r'(?:answer|result|solution)(?:\s+is)?\s*[:\=]?\s*(\d{1,3})',
            r'(?:^|\s)(\d{1,3})(?:\s*$|\.$)',
            r'(?:equals?|is)\s+(\d{1,3})',
            r'(\d{1,3})(?:\s*$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text.lower(), re.MULTILINE)
            if matches:
                answer = int(matches[-1])
                if 0 <= answer <= 999:
                    return answer
        
        return None


class MATHExtractor(AnswerExtractor):
    """Extractor for MATH dataset (supports boxed LaTeX and text answers)"""
    
    def get_name(self) -> str:
        return "MATH"
    
    def extract(self, response_text: str) -> Optional[str]:
        """Extract answer from \\boxed{...} notation or text"""
        if not response_text:
            return None
        
        # Look for \boxed{...} with proper handling of nested braces
        boxed_answer = extract_boxed_answer(response_text)
        if boxed_answer:
            return boxed_answer
        
        # Fallback: look for "answer is X" patterns
        answer_patterns = [
            r'(?:answer|result|solution)(?:\s+is)?\s*[:\=]?\s*(.+?)(?:\.|$)',
            r'(?:therefore|thus|hence|so),?\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, response_text.lower(), re.MULTILINE)
            if matches:
                return matches[-1].strip()
        
        return None


class GenericExtractor(AnswerExtractor):
    """Generic extractor that tries all patterns"""
    
    def get_name(self) -> str:
        return "Generic"
    
    def extract(self, response_text: str) -> Optional[str]:
        """Extract answer using multiple strategies"""
        if not response_text:
            return None
        
        # Try MATH boxed format first with proper nested brace handling
        boxed_answer = extract_boxed_answer(response_text)
        if boxed_answer:
            return boxed_answer
        
        # Try GSM8K #### format
        gsm_pattern = r'####\s*(.+?)(?:\n|$)'
        matches = re.findall(gsm_pattern, response_text)
        if matches:
            return matches[-1].strip()
        
        # Try "The final answer is $...$" or "The final answer is ..." patterns
        final_answer_patterns = [
            r'(?:the\s+)?final\s+answer\s+is\s+\$([^$]+)\$',  # "final answer is $110$"
            r'(?:the\s+)?final\s+answer\s+is\s+[:\=]?\s*(.+?)(?:\.|$)',  # "final answer is 110" or "final answer is: 110"
        ]
        
        for pattern in final_answer_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                answer = matches[-1].strip()
                # Clean up common LaTeX/formatting artifacts
                answer = answer.replace('\\', '').replace('$', '').strip()
                if answer:
                    return answer
        
        # Try general answer patterns
        answer_patterns = [
            r'(?:answer|result|solution)(?:\s+is)?\s*[:\=]?\s*(.+?)(?:\.|$)',
            r'(?:therefore|thus|hence|so),?\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, response_text, re.MULTILINE | re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        
        return None


# ============================================================================
# Answer Graders
# ============================================================================

class ExactMatchGrader(AnswerGrader):
    """Exact string/numeric match grader"""
    
    def get_name(self) -> str:
        return "ExactMatch"
    
    def grade(self, extracted_answer: Union[str, int, float, None], 
              ground_truth: Union[str, int, float]) -> bool:
        """Check if extracted answer exactly matches ground truth"""
        if extracted_answer is None:
            return False
        
        return str(extracted_answer).strip() == str(ground_truth).strip()


class MathGrader(AnswerGrader):
    """Grader using the grading module (grade_answer function)"""
    
    def get_name(self) -> str:
        return "MathGrader"
    
    def grade(self, extracted_answer: Union[str, int, float, None], 
              ground_truth: Union[str, int, float]) -> bool:
        """Grade using symbolic math comparison from grading module"""
        if extracted_answer is None:
            return False
        
        if not GRADING_AVAILABLE:
            # Fallback to exact match
            print("Warning: grading module not available, using exact match")
            return str(extracted_answer).strip() == str(ground_truth).strip()
        
        try:
            return grade_answer(str(extracted_answer), str(ground_truth))
        except Exception as e:
            print(f"Grading error: {e}, falling back to exact match")
            return str(extracted_answer).strip() == str(ground_truth).strip()


# ============================================================================
# Factory Classes
# ============================================================================

class LoaderFactory:
    """Factory for creating dataset loaders"""
    
    _loaders = {
        'aime': AIMELoader,
        'math': MATHLoader,
        'generic': GenericLoader,
    }
    
    @classmethod
    def create(cls, loader_type: str) -> DatasetLoader:
        """Create a loader instance"""
        loader_class = cls._loaders.get(loader_type.lower())
        if loader_class is None:
            raise ValueError(f"Unknown loader type: {loader_type}. Available: {list(cls._loaders.keys())}")
        return loader_class()
    
    @classmethod
    def register(cls, name: str, loader_class: type):
        """Register a custom loader"""
        cls._loaders[name.lower()] = loader_class


class ExtractorFactory:
    """Factory for creating answer extractors"""
    
    _extractors = {
        'aime': AIMEExtractor,
        'math': MATHExtractor,
        'generic': GenericExtractor,
    }
    
    @classmethod
    def create(cls, extractor_type: str) -> AnswerExtractor:
        """Create an extractor instance"""
        extractor_class = cls._extractors.get(extractor_type.lower())
        if extractor_class is None:
            raise ValueError(f"Unknown extractor type: {extractor_type}. Available: {list(cls._extractors.keys())}")
        return extractor_class()
    
    @classmethod
    def register(cls, name: str, extractor_class: type):
        """Register a custom extractor"""
        cls._extractors[name.lower()] = extractor_class


class GraderFactory:
    """Factory for creating answer graders"""
    
    _graders = {
        'exact_match': ExactMatchGrader,
        'math_grader': MathGrader,
    }
    
    @classmethod
    def create(cls, grader_type: str, **kwargs) -> AnswerGrader:
        """Create a grader instance"""
        grader_class = cls._graders.get(grader_type.lower())
        if grader_class is None:
            raise ValueError(f"Unknown grader type: {grader_type}. Available: {list(cls._graders.keys())}")
        return grader_class(**kwargs)
    
    @classmethod
    def register(cls, name: str, grader_class: type):
        """Register a custom grader"""
        cls._graders[name.lower()] = grader_class


# ============================================================================
# Configuration Management
# ============================================================================

class EvaluationConfig:
    """Configuration container for evaluation"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        
        # Dataset settings
        self.dataset_source = config_dict['dataset']['source']
        self.dataset_split = config_dict['dataset'].get('split', 'test')
        self.loader_type = config_dict['dataset']['loader_type']
        
        # Extraction and grading
        self.extractor_type = config_dict['answer_handling']['extractor_type']
        self.grader_type = config_dict['answer_handling']['grader_type']
        self.grader_params = config_dict['answer_handling'].get('grader_params', {})
        
        # Generation parameters
        gen_config = config_dict['generation']
        self.n_gen = gen_config['n_gen']
        self.temperature = gen_config['temperature']
        self.top_logprobs = gen_config.get('top_logprobs', 10)
        self.system_prompt = gen_config['system_prompt']
        self.streaming = gen_config.get('streaming', False)
        
        # Evaluation parameters
        eval_config = config_dict['evaluation']
        self.tail_n = eval_config['tail_n']
        self.group_size = eval_config['group_size']
        
        # Performance parameters
        perf_config = config_dict['performance']
        self.max_concurrent = perf_config.get('max_concurrent', 3)
        self.batch_size = perf_config.get('batch_size', 10)
        
        # Output settings
        output_config = config_dict['output']
        self.output_dir = output_config['output_dir']
        self.choices_dir = output_config.get('choices_dir', 'choices')
        self.metrics_file = output_config.get('metrics_file', 'all_response_metrics.jsonl')
        self.strategies_file = output_config.get('strategies_file', 'strategy_results.json')
        
        # OpenAI settings
        openai_config = config_dict.get('openai', {})
        self.model_id = openai_config.get('model_id')
        self.base_url = openai_config.get('base_url', 'http://localhost:8000/v1')
        self.api_key = openai_config.get('api_key', 'token-abc')
        self.timeout = openai_config.get('timeout', 3600)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'EvaluationConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def update_from_args(self, args: argparse.Namespace):
        """Update config from command-line arguments"""
        # Map argparse attributes to config
        arg_mappings = {
            'n_gen': ('generation', 'n_gen'),
            'temperature': ('generation', 'temperature'),
            'tail_n': ('evaluation', 'tail_n'),
            'group_size': ('evaluation', 'group_size'),
            'max_concurrent': ('performance', 'max_concurrent'),
            'batch_size': ('performance', 'batch_size'),
            'output_dir': ('output', 'output_dir'),
            'base_url': ('openai', 'base_url'),
        }
        
        for arg_name, (section, key) in arg_mappings.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                self.config[section][key] = getattr(args, arg_name)
                setattr(self, arg_name, getattr(args, arg_name))


# ============================================================================
# Main Evaluator Class
# ============================================================================

class MathEvaluator:
    """Main evaluator with modular architecture"""
    
    def __init__(self, config: EvaluationConfig, client):
        self.config = config
        self.client = client
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize components
        self.loader = LoaderFactory.create(config.loader_type)
        self.extractor = ExtractorFactory.create(config.extractor_type)
        self.grader = GraderFactory.create(config.grader_type, **config.grader_params)
        
        # Initialize metrics calculator
        self.metrics_calculator = ConfidenceMetrics()
        
        # Storage paths
        self.choices_dir = os.path.join(config.output_dir, config.choices_dir)
        self.metrics_file = os.path.join(config.output_dir, config.metrics_file)
        self.strategies_file = os.path.join(config.output_dir, config.strategies_file)
        
        print(f"Initialized MathEvaluator:")
        print(f"  Loader: {self.loader.get_name()}")
        print(f"  Extractor: {self.extractor.get_name()}")
        print(f"  Grader: {self.grader.get_name()}")
    
    def load_dataset(self) -> List[MathProblem]:
        """Load dataset using configured loader"""
        return self.loader.load(
            self.config.dataset_source,
            split=self.config.dataset_split
        )
    
    def _extract_message(self, choice):
        """Extract full text message from a choice object"""
        """Extract full text message from a vLLM/OpenAI-style choice object."""
        if not hasattr(choice, "logprobs") or not hasattr(choice.logprobs, "content"):
            return ""
        return "".join(cct.token for cct in choice.logprobs.content if hasattr(cct, "token"))

    async def generate_single_response_stream(self, config: Dict[str, Any]) -> Choice:
        """Generate a single response with streaming"""
        stream_config = {**config, "stream": True, "stream_options": {"include_usage": True}}
        
        if "logprobs" not in stream_config or not stream_config.get("logprobs"):
            stream_config["logprobs"] = True
            stream_config["top_logprobs"] = self.config.top_logprobs
        
        stream = await self.client.chat.completions.create(**stream_config)
        
        accumulated_content = ""
        accumulated_logprobs_content = []
        finish_reason = None
        index = 0
        
        async for chunk in stream:
            if not chunk.choices or len(chunk.choices) == 0:
                continue
            
            choice = chunk.choices[0]
            index = choice.index
            
            if hasattr(choice, 'delta') and choice.delta:
                if hasattr(choice.delta, 'content') and choice.delta.content:
                    accumulated_content += choice.delta.content
            
            if hasattr(choice, 'logprobs') and choice.logprobs:
                if hasattr(choice.logprobs, 'content') and choice.logprobs.content:
                    accumulated_logprobs_content.extend(choice.logprobs.content)
            
            if hasattr(choice, 'finish_reason') and choice.finish_reason:
                finish_reason = choice.finish_reason
        
        message = ChatCompletionMessage(
            content=accumulated_content,
            role="assistant",
            function_call=None,
            tool_calls=None
        )
        
        logprobs = ChoiceLogprobs(
            content=accumulated_logprobs_content if accumulated_logprobs_content else None,
            refusal=None
        )
        
        final_choice = Choice(
            finish_reason=finish_reason or "stop",
            index=index,
            logprobs=logprobs,
            message=message
        )
        
        return final_choice
    
    async def generate_multiple_responses_streaming(self, config: Dict[str, Any], n_gen: int) -> List[Choice]:
        """Generate multiple responses concurrently with streaming"""
        tasks = [self.generate_single_response_stream(config) for _ in range(n_gen)]
        choices = await asyncio.gather(*tasks)
        return choices
    
    async def generate_single_response(self, config: Dict[str, Any]) -> Any:
        """Generate a single response asynchronously"""
        response = await self.client.chat.completions.create(**config)
        return response.choices[0]
    
    async def generate_multiple_responses(self, config: Dict[str, Any], n_gen: int) -> List[Any]:
        """Generate multiple responses concurrently"""
        tasks = [self.generate_single_response(config) for _ in range(n_gen)]
        choices = await asyncio.gather(*tasks)
        return choices
    
    def evaluate_responses(self, problem: MathProblem, choices: List[Any]) -> List[ResponseResult]:
        """Evaluate all responses for a problem"""
        results = []
        all_metrics_records = []
        
        for i, choice in enumerate(choices):
            # Compute metrics
            metric_sequences = self.metrics_calculator.compute_all_metrics(choice)
            aggregated = aggregate_metrics(
                metric_sequences,
                tail_n=self.config.tail_n,
                group_size=self.config.group_size
            )
            
            # Extract and grade answer
            response_text = self._extract_message(choice)
            extracted_answer = None
            if response_text:
                extracted_answer = self.extractor.extract(response_text)
            
            is_correct = self.grader.grade(extracted_answer, problem.answer)
            
            result = ResponseResult(
                response_text=response_text,
                extracted_answer=extracted_answer,
                is_correct=is_correct,
                total_tokens=len(metric_sequences.get('mean', [])),
                choice_index=i,
                aggregated_metrics=aggregated
            )
            results.append(result)
            
            # Prepare metrics record
            metrics_record = {
                'problem_id': problem.problem_id,
                'rollout_idx': i,
                'response_text': response_text,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct,
                'expected_answer': problem.answer,
                'total_tokens': result.total_tokens,
                **aggregated
            }
            all_metrics_records.append(metrics_record)
        
        # Save metrics
        MetricsStorage.save_all_metrics_batch(
            all_metrics_records,
            self.metrics_file,
            use_compression=False
        )
        
        return results
    
    def apply_selection_strategies(self, results: List[ResponseResult]) -> Dict[str, ResponseResult]:
        """Apply different selection strategies"""
        strategies = {}
        
        if not results:
            return strategies
        
        # Random selection
        strategies['random'] = random.choice(results)
        
        # Oracle selection
        correct_results = [r for r in results if r.is_correct]
        if correct_results:
            strategies['oracle'] = max(correct_results, key=lambda x: x.aggregated_metrics.get('mean_tail', 0))
        else:
            strategies['oracle'] = max(results, key=lambda x: x.aggregated_metrics.get('mean_tail', 0))
        
        # Generate strategies for all metric aggregations
        metric_keys = list(results[0].aggregated_metrics.keys())
        
        for metric_key in metric_keys:
            strategies[f'highest_{metric_key}'] = max(
                results,
                key=lambda x: x.aggregated_metrics.get(metric_key, 0)
            )
            
            strategies[f'lowest_{metric_key}'] = min(
                results,
                key=lambda x: x.aggregated_metrics.get(metric_key, 0)
            )
        
        return strategies
    
    async def evaluate_problem(self, problem: MathProblem) -> Dict[str, Any]:
        """Evaluate a single problem"""
        print(f"\nEvaluating Problem {problem.problem_id}")
        print(f"Expected answer: {problem.answer}")
        
        gen_config = {
            "model": self.config.model_id,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": self.config.system_prompt
                },
                {
                    "role": "user",
                    "content": problem.problem_text
                }
            ],
            "logprobs": True,
            "top_logprobs": self.config.top_logprobs,
            "extra_body": {"reasoning_effort": "high"}
        }
        
        # Generate responses
        print(f"Generating {self.config.n_gen} responses...")
        time_start = time.time()
        
        if self.config.streaming:
            choices = await self.generate_multiple_responses_streaming(gen_config, self.config.n_gen)
        else:
            choices = await self.generate_multiple_responses(gen_config, self.config.n_gen)
        
        time_end = time.time()
        print(f"Generation complete in {time_end - time_start:.1f}s")
        
        # Save choices
        time_start = time.time()
        ChoiceStorage.save_choices(
            problem.problem_id,
            choices,
            self.choices_dir,
            use_compression=False
        )
        time_end = time.time()
        print(f"Saved choices in {time_end - time_start:.1f}s")
        
        # Evaluate responses
        time_start = time.time()
        results = self.evaluate_responses(problem, choices)
        strategies = self.apply_selection_strategies(results)
        time_end = time.time()
        print(f"Evaluation complete in {time_end - time_start:.1f}s")
        
        return {
            'problem': problem,
            'results': results,
            'strategies': strategies,
            'n_correct': sum(1 for r in results if r.is_correct),
            'total_generated': len(results)
        }
    
    async def evaluate_problem_with_semaphore(self, problem: MathProblem, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Evaluate a single problem with semaphore control"""
        async with semaphore:
            return await self.evaluate_problem(problem)
    
    async def evaluate_problems_batch(self, problems: List[MathProblem]) -> List[Dict[str, Any]]:
        """Evaluate a batch of problems concurrently"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        tasks = [
            self.evaluate_problem_with_semaphore(problem, semaphore)
            for problem in problems
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error evaluating problem {problems[i].problem_id}: {result}")
                dummy_result = {
                    'problem': problems[i],
                    'results': [],
                    'strategies': {},
                    'n_correct': 0,
                    'total_generated': 0,
                    'error': str(result)
                }
                successful_results.append(dummy_result)
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def evaluate_dataset(self) -> Dict[str, Any]:
        """Evaluate entire dataset"""
        problems = self.load_dataset()
        
        print(f"\n{'='*80}")
        print(f"Loaded {len(problems)} problems for evaluation")
        print(f"Will generate {self.config.n_gen} responses per problem")
        print(f"Using tail-{self.config.tail_n} confidence for selection")
        print(f"Processing {self.config.max_concurrent} problems concurrently")
        print(f"Batch size: {self.config.batch_size}")
        print(f"{'='*80}\n")
        
        all_results = []
        total_batches = (len(problems) + self.config.batch_size - 1) // self.config.batch_size
        
        for batch_idx in range(0, len(problems), self.config.batch_size):
            batch_end = min(batch_idx + self.config.batch_size, len(problems))
            current_batch = problems[batch_idx:batch_end]
            current_batch_num = batch_idx // self.config.batch_size + 1
            
            print(f"\n{'='*60}")
            print(f"Processing Batch {current_batch_num}/{total_batches}")
            print(f"Problems {batch_idx + 1}-{batch_end} of {len(problems)}")
            print(f"{'='*60}")
            
            try:
                batch_results = await self.evaluate_problems_batch(current_batch)
                all_results.extend(batch_results)
                
                batch_correct = sum(1 for r in batch_results if r.get('n_correct', 0) > 0)
                print(f"\nBatch {current_batch_num} complete:")
                print(f"  Problems with ≥1 correct: {batch_correct}/{len(current_batch)}")
                
            except Exception as e:
                print(f"Error in batch {current_batch_num}: {e}")
                continue
        
        # Aggregate results
        dataset_info = {
            'source': self.config.dataset_source,
            'split': self.config.dataset_split,
            'loader': self.loader.get_name(),
            'extractor': self.extractor.get_name(),
            'grader': self.grader.get_name(),
            'total_problems': len(problems),
            'n_gen': self.config.n_gen,
            'tail_n': self.config.tail_n,
            'group_size': self.config.group_size,
            'temperature': self.config.temperature,
        }
        
        strategy_results = ResultsAggregator.aggregate_strategy_results(
            all_results,
            dataset_info,
            self.strategies_file
        )
        
        print(f"\n{'='*80}")
        print("RESULTS SAVED")
        print(f"{'='*80}")
        print(f"Strategy results: {self.strategies_file}")
        print(f"All response metrics: {self.metrics_file}")
        print(f"Full choices: {self.choices_dir}/")
        print(f"{'='*80}\n")
        
        return {
            'problem_results': all_results,
            'strategy_results': strategy_results,
            'dataset_info': dataset_info
        }
    
    def _evaluate_problem_worker(self, problem: MathProblem, choices_filename: str, 
                                 tail_n: int, group_size: int, topk_logprobs: Optional[int] = None) -> Dict[str, Any]:
        """
        Worker function that runs in separate process for offline evaluation.
        Must be a top-level function or static method for pickling.
        """
        # Load choices
        choices = ChoiceStorage.load_choices(choices_filename)
        
        # Evaluate responses
        results = self.evaluate_responses(problem, choices)
        
        # Apply selection strategies
        strategies = self.apply_selection_strategies(results)
        
        return {
            'problem': problem,
            'results': results,
            'strategies': strategies,
            'n_correct': sum(1 for r in results if r.is_correct),
            'total_generated': len(results)
        }
    
    async def evaluate_dataset_offline(self, stored_choices_dir: str) -> Dict[str, Any]:
        """
        Evaluate entire dataset based on generated choices already stored on disk.
        All problems are processed in parallel with controlled concurrency.
        
        Args:
            stored_choices_dir: Directory containing stored choices
        """
        problems = self.load_dataset()
        
        print(f"\n{'='*80}")
        print(f"Loaded {len(problems)} problems for offline evaluation")
        print(f"Using tail-{self.config.tail_n} confidence for selection")
        print(f"Processing up to {self.config.max_concurrent} problems concurrently")
        print(f"Reading choices from: {stored_choices_dir}")
        print(f"{'='*80}\n")
        
        # Use ProcessPoolExecutor for CPU-bound work
        num_workers = min(self.config.max_concurrent, os.cpu_count() or 1)
        print(f"Using {num_workers} worker processes")
        
        # Progress tracking
        completed = {'count': 0}
        completed_lock = asyncio.Lock()
        
        # Create process pool
        loop = asyncio.get_event_loop()
        executor = ProcessPoolExecutor(max_workers=num_workers)
        
        try:
            async def evaluate_single_problem_offline(problem, idx):
                """Evaluate a single problem with stored choices"""
                try:
                    print(f"\n[{idx + 1}/{len(problems)}] Starting evaluation for Problem {problem.problem_id}")
                    
                    choices_filename = os.path.join(stored_choices_dir, f"{problem.problem_id}_choices.pkl")
                    
                    # Run entire evaluation in process pool
                    result = await loop.run_in_executor(
                        executor,
                        self._evaluate_problem_worker,
                        problem, choices_filename, self.config.tail_n, self.config.group_size, self.config.top_logprobs
                    )
                    
                    async with completed_lock:
                        completed['count'] += 1
                        n_correct = result['n_correct']
                        status = "✓ HAS CORRECT" if n_correct > 0 else "✗ NO CORRECT"
                        print(f"[{completed['count']}/{len(problems)}] Completed Problem {problem.problem_id}: {n_correct}/{result['total_generated']} correct - {status}")
                    
                    return result
                    
                except Exception as e:
                    print(f"Error evaluating problem {problem.problem_id}: {e}")
                    async with completed_lock:
                        completed['count'] += 1
                    return {
                        'problem': problem,
                        'error': str(e),
                        'n_correct': 0,
                        'total_generated': 0
                    }
            
            # Create tasks for all problems
            tasks = [
                evaluate_single_problem_offline(problem, idx)
                for idx, problem in enumerate(problems)
            ]
            
            print(f"\n{'='*60}")
            print(f"Starting parallel evaluation of all {len(problems)} problems...")
            print(f"{'='*60}\n")
            
            # Run all tasks concurrently
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
        finally:
            # Clean up executor
            executor.shutdown(wait=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                print(f"Exception in problem {i}: {result}")
                valid_results.append({
                    'problem': problems[i],
                    'error': str(result),
                    'n_correct': 0,
                    'total_generated': 0
                })
            else:
                valid_results.append(result)
        
        # Print final summary
        total_correct = sum(1 for r in valid_results if r.get('n_correct', 0) > 0)
        total_responses = sum(r.get('total_generated', 0) for r in valid_results)
        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total problems: {len(problems)}")
        print(f"Problems with ≥1 correct: {total_correct}/{len(problems)}")
        print(f"Total responses evaluated: {total_responses}")
        print(f"{'='*80}\n")
        
        # Aggregate results
        dataset_info = {
            'source': self.config.dataset_source,
            'split': self.config.dataset_split,
            'loader': self.loader.get_name(),
            'extractor': self.extractor.get_name(),
            'grader': self.grader.get_name(),
            'total_problems': len(problems),
            'tail_n': self.config.tail_n,
            'group_size': self.config.group_size,
        }
        
        # Save strategy-level results
        strategy_results = ResultsAggregator.aggregate_strategy_results(
            valid_results,
            dataset_info,
            self.strategies_file
        )
        
        print(f"\n{'='*80}")
        print("RESULTS SAVED")
        print(f"{'='*80}")
        print(f"Strategy results: {self.strategies_file}")
        print(f"All response metrics: {self.metrics_file}")
        print(f"{'='*80}\n")
        
        return {
            'problem_results': valid_results,
            'strategy_results': strategy_results,
            'dataset_info': dataset_info
        }


# ============================================================================
# Command-Line Interface
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Modular Math Evaluation Framework")
    
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML configuration file")
    
    # Optional overrides
    parser.add_argument("--n_gen", type=int, help="Number of generations per problem")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--tail_n", type=int, help="Tail confidence window size")
    parser.add_argument("--group_size", type=int, help="Group size for metrics")
    parser.add_argument("--max_concurrent", type=int, help="Max concurrent problems")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_url", type=str, help="OpenAI API base URL")
    
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = EvaluationConfig.from_yaml(args.config)
    
    # Apply command-line overrides
    config.update_from_args(args)
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize OpenAI client
    client = openai.AsyncOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
    )
    
    # Get model ID if not specified
    if config.model_id is None:
        models = await client.models.list()
        config.model_id = models.data[0].id
    
    print(f"Using model: {config.model_id}")
    
    # Initialize evaluator
    evaluator = MathEvaluator(config, client)
    
    # Run evaluation
    print("\n" + "="*80)
    print("STARTING MATH EVALUATION")
    print("="*80)
    
    start_time = time.time()
    results = await evaluator.evaluate_dataset()
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Total time: {execution_time:.1f}s ({execution_time/60:.1f} min)")
    print(f"Problems evaluated: {results['dataset_info']['total_problems']}")
    
    strategy_results = results['strategy_results']['strategy_results']
    if strategy_results:
        best = max(strategy_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest strategy: {best[0]}")
        print(f"  Accuracy: {best[1]['accuracy']:.2%}")
        print(f"  Correct: {best[1]['correct']}/{best[1]['total']}")
    
    print(f"\n📁 Results saved to: {config.output_dir}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
