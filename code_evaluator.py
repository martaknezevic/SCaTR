"""
Modular Code Evaluation Framework

A flexible, configuration-driven evaluation system for code generation datasets.
Supports HumanEval, Kodcode, and custom datasets through pluggable components.

Key Features:
- Abstract base classes for dataset loaders, answer extractors, and graders
- Factory pattern for component instantiation from config
- Configuration-driven (YAML) with all parameters externalized
- Async generation with vLLM support
- Clean separation of concerns with modular architecture

Usage:
    python code_evaluator.py --config config_humaneval.yaml
    python code_evaluator.py --config config_kodcode.yaml --n_gen 20
"""

import os
import re
import sys
import json
import time
import random
import asyncio
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from abc import ABC, abstractmethod
import aiohttp

import yaml
import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs

# Import existing components
from metrics import ConfidenceMetrics, aggregate_metrics
from storage import ChoiceStorage, MetricsStorage, ResultsAggregator
from code_utils import (
    extract_function_with_dependencies,
    format_reward,
    parse_between_output_tags,
)

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class CodeProblem:
    """Universal structure for code problems"""
    problem_id: str
    problem_text: str
    test_cases: Union[List[str], Dict[str, List[str]]]  # List for HumanEval, Dict for Kodcode
    entry_point: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CodeResponseResult:
    """Structure for code response evaluation"""
    response_text: str
    extracted_code: Optional[str]
    is_correct: bool
    num_tests_passed: int
    total_tests: int
    feedback: str
    format_reward: float
    code_reward: float
    total_reward: float
    total_tokens: int
    choice_index: int
    aggregated_metrics: Dict[str, float]


# ============================================================================
# Abstract Base Classes
# ============================================================================

class DatasetLoader(ABC):
    """Abstract base class for dataset loaders"""
    
    @abstractmethod
    def load(self, dataset_source: str, split: str = "test", **kwargs) -> List[CodeProblem]:
        """Load dataset and return list of CodeProblems"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return loader name"""
        pass


class CodeExtractor(ABC):
    """Abstract base class for code extraction"""
    
    @abstractmethod
    def extract(self, response_text: str, entry_point: Optional[str] = None) -> Optional[str]:
        """Extract code from response"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return extractor name"""
        pass


class CodeGrader(ABC):
    """Abstract base class for code grading"""
    
    @abstractmethod
    async def grade(self, code: str, test_cases: Union[List[str], Dict[str, List[str]]], 
                    timeout: int = 10) -> Tuple[bool, str, List[float]]:
        """
        Grade code against test cases
        Returns: (is_passing, feedback, partial_test_cases)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return grader name"""
        pass


# ============================================================================
# Dataset Loaders
# ============================================================================

class HumanEvalLoader(DatasetLoader):
    """Loader for HumanEval dataset"""
    
    def get_name(self) -> str:
        return "HumanEval"
    
    def _load_from_file(self, dataset_source: str) -> List[CodeProblem]:
        """Load from JSON or JSONL file"""
        problems = []
        
        if dataset_source.endswith('.jsonl'):
            # Load JSONL
            with open(dataset_source, 'r') as f:
                data = [json.loads(line) for line in f]
        else:
            # Load JSON
            with open(dataset_source, 'r') as f:
                data = json.load(f)
        
        for idx, item in enumerate(data):
            test_cases = [
                line.strip().replace("candidate", item.get("entry_point", ""))
                for line in item.get("test", "").split("\n")
                if "assert" in line
            ]
            
            problem = CodeProblem(
                problem_id=item.get("task_id", f"problem_{idx}").replace("/", "_"),
                problem_text=item.get("prompt", ""),
                test_cases=test_cases,
                entry_point=item.get("entry_point"),
                metadata=item.get("metadata", {})
            )
            problems.append(problem)
        
        file_type = "JSONL" if dataset_source.endswith('.jsonl') else "JSON"
        print(f"Loaded {len(problems)} problems from {file_type}")
        return problems
    
    def load(self, dataset_source: str, split: str = "test", **kwargs) -> List[CodeProblem]:
        """Load HumanEval dataset from HuggingFace, JSON, or JSONL"""
        
        # Try local file first (JSON or JSONL)
        if dataset_source.endswith(('.json', '.jsonl')):
            return self._load_from_file(dataset_source)
        
        # Try HuggingFace
        if HF_AVAILABLE:
            try:
                dataset = load_dataset(dataset_source, split=split)
                problems = []
                
                for idx, item in enumerate(dataset):
                    # Extract test cases
                    test_cases = [
                        line.strip().replace("candidate", item.get("entry_point", ""))
                        for line in item.get("test", "").split("\n")
                        if "assert" in line
                    ]
                    
                    problem = CodeProblem(
                        problem_id=item.get("task_id", f"problem_{idx}").replace("/", "_"),
                        problem_text=item.get("prompt", ""),
                        test_cases=test_cases,
                        entry_point=item.get("entry_point"),
                        metadata={
                            "canonical_solution": item.get("canonical_solution", ""),
                        }
                    )
                    problems.append(problem)
                
                print(f"Loaded {len(problems)} problems from HuggingFace")
                return problems
            except Exception as e:
                print(f"Failed to load from HuggingFace: {e}")
        
        raise ValueError(f"Could not load HumanEval dataset from {dataset_source}")


class KodcodeLoader(DatasetLoader):
    """Loader for Kodcode dataset"""
    
    def get_name(self) -> str:
        return "Kodcode"
    
    def _load_from_file(self, dataset_source: str) -> List[CodeProblem]:
        """Load from JSON or JSONL file"""
        problems = []
        
        if dataset_source.endswith('.jsonl'):
            # Load JSONL
            with open(dataset_source, 'r') as f:
                data = [json.loads(line) for line in f]
        else:
            # Load JSON
            with open(dataset_source, 'r') as f:
                data = json.load(f)
        
        for idx, item in enumerate(data):
            # Kodcode has test cases in 'test' field
            test_cases = item.get("test", item.get("tests", []))
            
            problem = CodeProblem(
                problem_id=item.get("problem_id", item.get("task_id", f"problem_{idx}")),
                problem_text=item.get("prompt", ""),
                test_cases=test_cases,  # Dict with 'input' and 'output'
                entry_point=item.get("entry_point"),
                metadata=item.get("metadata", {})
            )
            problems.append(problem)
        
        file_type = "JSONL" if dataset_source.endswith('.jsonl') else "JSON"
        print(f"Loaded {len(problems)} problems from {file_type}")
        return problems
    
    def load(self, dataset_source: str, split: str = "test", **kwargs) -> List[CodeProblem]:
        """Load Kodcode dataset from HuggingFace, JSON, or JSONL"""
        
        # Try local file first (JSON or JSONL)
        if dataset_source.endswith(('.json', '.jsonl')):
            return self._load_from_file(dataset_source)
        
        # Try HuggingFace
        if HF_AVAILABLE:
            try:
                dataset = load_dataset(dataset_source, split=split)
                problems = []
                
                for idx, item in enumerate(dataset):
                    # Kodcode has test cases in 'test' field
                    test_cases = item.get("test", item.get("tests", []))
                    
                    problem = CodeProblem(
                        problem_id=item.get("problem_id", item.get("task_id", f"problem_{idx}")),
                        problem_text=item.get("prompt", ""),
                        test_cases=test_cases,
                        entry_point=item.get("entry_point"),
                        metadata={
                            "difficulty": item.get("difficulty", ""),
                            "source": item.get("source", ""),
                        }
                    )
                    problems.append(problem)
                
                print(f"Loaded {len(problems)} problems from HuggingFace")
                return problems
            except Exception as e:
                print(f"Failed to load from HuggingFace: {e}")
                raise
        
        raise ValueError("HuggingFace datasets not available")


class GenericCodeLoader(DatasetLoader):
    """Generic loader for custom code datasets"""
    
    def get_name(self) -> str:
        return "GenericCode"
    
    def load(self, dataset_source: str, split: str = "test", **kwargs) -> List[CodeProblem]:
        """Load generic code dataset from JSON or JSONL"""
        
        if dataset_source.endswith('.jsonl'):
            # Load JSONL
            with open(dataset_source, 'r') as f:
                data = [json.loads(line) for line in f]
        else:
            # Load JSON
            with open(dataset_source, 'r') as f:
                data = json.load(f)
        
        problems = []
        for i, item in enumerate(data):
            problem = CodeProblem(
                problem_id=item.get("problem_id", f"problem_{i}"),
                problem_text=item.get("prompt", item.get("problem_text", "")),
                test_cases=item.get("test_cases", item.get("tests", item.get("test", []))),
                entry_point=item.get("entry_point"),
                metadata=item.get("metadata", {})
            )
            problems.append(problem)
        
        file_type = "JSONL" if dataset_source.endswith('.jsonl') else "JSON"
        print(f"Loaded {len(problems)} problems from {file_type}")
        return problems


# ============================================================================
# Code Extractors
# ============================================================================

class OutputTagExtractor(CodeExtractor):
    """Extractor for code within <output>...</output> tags"""
    
    def get_name(self) -> str:
        return "OutputTag"
    
    def extract(self, response_text: str, entry_point: Optional[str] = None) -> Optional[str]:
        """Extract code from <output> tags"""
        if not response_text:
            return None
        
        # First extract from output tags
        code = parse_between_output_tags(response_text)
        return code


class MarkdownCodeBlockExtractor(CodeExtractor):
    """Extractor for code within markdown code blocks"""
    
    def get_name(self) -> str:
        return "MarkdownCodeBlock"
    
    def extract(self, response_text: str, entry_point: Optional[str] = None) -> Optional[str]:
        """Extract code from ```python``` blocks, trying from last to first"""
        if not response_text:
            return None
        
        # Look for ```python or ``` code blocks
        pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if not matches:
            return None
        
        # Try code blocks from last to first
        for code in reversed(matches):
            code = code.strip()
            if not code:
                continue
            
            # If entry point specified, try to extract that function
            if entry_point:
                try:
                    extracted = extract_function_with_dependencies(code, entry_point)
                    if extracted:
                        return extracted
                except Exception:
                    # If extraction fails, try next code block
                    continue
            else:
                # No entry point, return the code as-is
                return code
        
        # If we have matches but couldn't extract with entry_point, return last non-empty block
        for code in reversed(matches):
            code = code.strip()
            if code:
                return code
        
        return None


class GenericCodeExtractor(CodeExtractor):
    """Generic extractor that tries multiple strategies"""
    
    def get_name(self) -> str:
        return "GenericCode"
    
    def extract(self, response_text: str, entry_point: Optional[str] = None) -> Optional[str]:
        """Extract code using multiple strategies"""
        if not response_text:
            return None
        
        # Try output tags first
        code = parse_between_output_tags(response_text)
        if code:
            if entry_point:
                try:
                    code = extract_function_with_dependencies(code, entry_point)
                except Exception:
                    pass
            return code
        
        # Try markdown code blocks
        pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            code = matches[-1].strip()
            if entry_point:
                try:
                    code = extract_function_with_dependencies(code, entry_point)
                except Exception:
                    pass
            return code
        
        # If nothing found, return the response as-is (might be plain code)
        return response_text.strip()


# ============================================================================
# Code Graders
# ============================================================================

class ExecutorAPIGrader(CodeGrader):
    """Grader using the code executor API"""
    
    def __init__(self, executor_url: str = "http://localhost:8000/execute", 
                 max_concurrent: int = 6):
        self.executor_url = executor_url
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    def get_name(self) -> str:
        return "ExecutorAPI"
    
    async def grade(self, code: str, test_cases: Union[List[str], Dict[str, List[str]]], 
                    timeout: int = 10) -> Tuple[bool, str, List[float]]:
        """Grade code against test cases using executor API"""
        
        if not code:
            num_tests = len(test_cases.get("input", [])) if isinstance(test_cases, dict) else len(test_cases)
            return False, "No code extracted", [0.0] * num_tests
        
        async with self.semaphore:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.executor_url,
                        json={"code": code, "tests": test_cases, "timeout": timeout},
                        timeout=aiohttp.ClientTimeout(total=timeout + 5),
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                        return (
                            result["is_passing"],
                            result["feedback"],
                            result["partial_test_cases"],
                        )
            except Exception as e:
                num_tests = len(test_cases.get("input", [])) if isinstance(test_cases, dict) else len(test_cases)
                return (
                    False,
                    f"Execution error: {str(e)}",
                    [0.0] * num_tests,
                )


class HybridExtractor(CodeExtractor):
    """Hybrid extractor that tries both <output> tags and markdown, picks best result"""
    
    def __init__(self, grader: ExecutorAPIGrader, execution_timeout: int = 10):
        self.output_extractor = OutputTagExtractor()
        self.markdown_extractor = MarkdownCodeBlockExtractor()
        self.grader = grader
        self.execution_timeout = execution_timeout
    
    def get_name(self) -> str:
        return "Hybrid"
    
    async def extract_and_grade(self, response_text: str, entry_point: Optional[str], 
                                test_cases, extractor) -> tuple:
        """Extract code and grade it"""
        code = extractor.extract(response_text, entry_point)
        if not code:
            return None, 0, []
        
        is_passing, feedback, partial_tests = await self.grader.grade(
            code, test_cases, self.execution_timeout
        )
        
        num_passed = int(np.sum(np.asarray(partial_tests, dtype=np.float32)))
        
        return code, num_passed, partial_tests
    
    async def extract_with_test(self, response_text: str, entry_point: Optional[str] = None,
                               test_cases = None) -> Optional[str]:
        """Extract using both methods and pick the one that passes more tests"""
        if not response_text or test_cases is None:
            # Fallback to output tag if no test cases provided
            return self.output_extractor.extract(response_text, entry_point)
        
        # Try both extractors
        output_code, output_passed, _ = await self.extract_and_grade(
            response_text, entry_point, test_cases, self.output_extractor
        )
        
        markdown_code, markdown_passed, _ = await self.extract_and_grade(
            response_text, entry_point, test_cases, self.markdown_extractor
        )
        
        # Pick the one with more tests passed
        if output_code is None and markdown_code is None:
            return None
        elif output_code is None:
            return markdown_code
        elif markdown_code is None:
            return output_code
        elif markdown_passed > output_passed:
            return markdown_code
        else:
            return output_code
    
    def extract(self, response_text: str, entry_point: Optional[str] = None) -> Optional[str]:
        """Synchronous extract - just try markdown first, then output tags"""
        # This is for non-async usage
        code = self.markdown_extractor.extract(response_text, entry_point)
        if code:
            return code
        return self.output_extractor.extract(response_text, entry_point)


# ============================================================================
# Factory Classes
# ============================================================================

class LoaderFactory:
    """Factory for creating dataset loaders"""
    
    _loaders = {
        'humaneval': HumanEvalLoader,
        'kodcode': KodcodeLoader,
        'generic': GenericCodeLoader,
    }
    
    @classmethod
    def create(cls, loader_type: str) -> DatasetLoader:
        loader_type = loader_type.lower()
        if loader_type not in cls._loaders:
            raise ValueError(f"Unknown loader type: {loader_type}")
        return cls._loaders[loader_type]()
    
    @classmethod
    def register(cls, name: str, loader_class: type):
        cls._loaders[name.lower()] = loader_class


class ExtractorFactory:
    """Factory for creating code extractors"""
    
    _extractors = {
        'output_tag': OutputTagExtractor,
        'markdown': MarkdownCodeBlockExtractor,
        'generic': GenericCodeExtractor,
        'hybrid': HybridExtractor,
    }
    
    @classmethod
    def create(cls, extractor_type: str, **kwargs) -> CodeExtractor:
        extractor_type = extractor_type.lower()
        if extractor_type not in cls._extractors:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
        return cls._extractors[extractor_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, extractor_class: type):
        cls._extractors[name.lower()] = extractor_class


class GraderFactory:
    """Factory for creating code graders"""
    
    _graders = {
        'executor_api': ExecutorAPIGrader,
    }
    
    @classmethod
    def create(cls, grader_type: str, **kwargs) -> CodeGrader:
        grader_type = grader_type.lower()
        if grader_type not in cls._graders:
            raise ValueError(f"Unknown grader type: {grader_type}")
        return cls._graders[grader_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, grader_class: type):
        cls._graders[name.lower()] = grader_class


# ============================================================================
# Configuration Management
# ============================================================================

class CodeEvaluationConfig:
    """Configuration container for code evaluation"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        
        # Dataset settings
        self.dataset_source = config_dict['dataset']['source']
        self.dataset_split = config_dict['dataset'].get('split', 'test')
        self.loader_type = config_dict['dataset']['loader_type']
        self.num_samples = config_dict['dataset'].get('num_samples', None)
        
        # Code handling
        self.extractor_type = config_dict['code_handling']['extractor_type']
        self.grader_type = config_dict['code_handling']['grader_type']
        self.grader_params = config_dict['code_handling'].get('grader_params', {})
        self.execution_timeout = config_dict['code_handling'].get('execution_timeout', 10)
        
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
    def from_yaml(cls, yaml_path: str) -> 'CodeEvaluationConfig':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def update_from_args(self, args: argparse.Namespace):
        """Update config from command-line arguments"""
        arg_mappings = {
            'n_gen': ('generation', 'n_gen'),
            'temperature': ('generation', 'temperature'),
            'tail_n': ('evaluation', 'tail_n'),
            'group_size': ('evaluation', 'group_size'),
            'max_concurrent': ('performance', 'max_concurrent'),
            'batch_size': ('performance', 'batch_size'),
            'output_dir': ('output', 'output_dir'),
            'num_samples': ('dataset', 'num_samples'),
            'base_url': ('openai', 'base_url'),
        }
        
        for arg_name, (section, key) in arg_mappings.items():
            arg_value = getattr(args, arg_name, None)
            if arg_value is not None:
                self.config[section][key] = arg_value
                setattr(self, arg_name, arg_value)


# ============================================================================
# Main Evaluator Class
# ============================================================================

class CodeEvaluator:
    """Main code evaluator with modular architecture"""
    
    def __init__(self, config: CodeEvaluationConfig, client):
        self.config = config
        self.client = client
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize components
        self.loader = LoaderFactory.create(config.loader_type)
        self.grader = GraderFactory.create(config.grader_type, **config.grader_params)
        
        # Initialize extractor (hybrid extractor needs grader)
        if config.extractor_type.lower() == 'hybrid':
            self.extractor = ExtractorFactory.create(
                config.extractor_type,
                grader=self.grader,
                execution_timeout=config.execution_timeout
            )
        else:
            self.extractor = ExtractorFactory.create(config.extractor_type)
        
        # Initialize metrics calculator
        self.metrics_calculator = ConfidenceMetrics()
        
        # Storage paths
        self.choices_dir = os.path.join(config.output_dir, config.choices_dir)
        self.metrics_file = os.path.join(config.output_dir, config.metrics_file)
        self.strategies_file = os.path.join(config.output_dir, config.strategies_file)
        
        print(f"Initialized CodeEvaluator:")
        print(f"  Loader: {self.loader.get_name()}")
        print(f"  Extractor: {self.extractor.get_name()}")
        print(f"  Grader: {self.grader.get_name()}")
    
    def load_dataset(self) -> List[CodeProblem]:
        """Load dataset using configured loader"""
        problems = self.loader.load(
            self.config.dataset_source,
            split=self.config.dataset_split
        )
        
        # Limit number of samples if specified
        if self.config.num_samples and len(problems) > self.config.num_samples:
            np.random.seed(42)
            problems = list(np.random.choice(problems, self.config.num_samples, replace=False))
            print(f"Limited to {len(problems)} samples")
        
        return problems
    
    def _format_prompt(self, problem: CodeProblem) -> str:
        """Format problem text into a complete prompt"""
        #additional_prompt = (
        #    f"\nNote that if you don't complete thinking and the code generation "
        #    f"within {self.config.max_completion_length} tokens, you will get a zero reward."
        #)
        
        return problem.problem_text #+ additional_prompt
    
    def _extract_message(self, choice):
        """Extract full text message from a vLLM/OpenAI-style choice object."""
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            return choice.message.content if choice.message.content else ""
        
        # Fallback: if no message content, return empty string
        return ""
        
    
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
        usage_info = None
        
        async for chunk in stream:
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_info = chunk.usage
                continue
            
            choice = chunk.choices[0]
            
            if hasattr(choice, "index"):
                index = choice.index
            
            if hasattr(choice, "finish_reason") and choice.finish_reason:
                finish_reason = choice.finish_reason
            
            delta = choice.delta
            
            if hasattr(delta, "content") and delta.content:
                accumulated_content += delta.content
            
            if hasattr(choice, "logprobs") and choice.logprobs:
                if hasattr(choice.logprobs, "content") and choice.logprobs.content:
                    accumulated_logprobs_content.extend(choice.logprobs.content)
        
        message = ChatCompletionMessage(role="assistant", content=accumulated_content)
        
        logprobs_obj = None
        if accumulated_logprobs_content:
            logprobs_obj = ChoiceLogprobs(content=accumulated_logprobs_content)
        
        return Choice(
            index=index,
            message=message,
            finish_reason=finish_reason or "stop",
            logprobs=logprobs_obj,
        )
    
    async def generate_multiple_responses_streaming(self, config: Dict[str, Any], n_gen: int) -> List[Choice]:
        """Generate multiple responses using streaming"""
        tasks = [self.generate_single_response_stream(config) for _ in range(n_gen)]
        return await asyncio.gather(*tasks)
    
    async def generate_single_response(self, config: Dict[str, Any]) -> Any:
        """Generate a single response (non-streaming)"""
        response = await self.client.chat.completions.create(**config)
        return response.choices[0]
    
    async def generate_multiple_responses(self, config: Dict[str, Any], n_gen: int) -> List[Any]:
        """Generate multiple responses in parallel (non-streaming)"""
        config_with_n = {**config, "n": n_gen}
        response = await self.client.chat.completions.create(**config_with_n)
        return response.choices
    
    async def evaluate_responses(self, problem: CodeProblem, choices: List[Any]) -> List[CodeResponseResult]:
        """Evaluate all responses for a problem"""
        results = []
        all_metrics_records = []
        
        # Prepare all grading tasks
        grading_tasks = []
        extracted_codes = []
        
        for choice in choices:
            response_text = self._extract_message(choice)
            extracted_code = self.extractor.extract(response_text, problem.entry_point)
            extracted_codes.append(extracted_code)
            
            # Create grading task
            task = self.grader.grade(
                extracted_code if extracted_code else "",
                problem.test_cases,
                self.config.execution_timeout
            )
            grading_tasks.append(task)
        
        # Execute all grading in parallel
        grading_results = await asyncio.gather(*grading_tasks)
        
        # Process results
        for choice_idx, (choice, extracted_code, (is_passing, feedback, partial_tests)) in enumerate(
            zip(choices, extracted_codes, grading_results)
        ):
            response_text = self._extract_message(choice)
            
            # Calculate rewards
            format_rew = format_reward(response_text)
            code_rew = np.mean(np.asarray(partial_tests, dtype=np.float32))
            total_rew = code_rew + format_rew
            
            num_passed = int(np.sum(np.asarray(partial_tests, dtype=np.float32)))
            total_tests = len(partial_tests)
            
            # Compute metrics
            metric_sequences = self.metrics_calculator.compute_all_metrics(choice)
            aggregated = aggregate_metrics(
                metric_sequences,
                tail_n=self.config.tail_n,
                group_size=self.config.group_size
            )
            
            # Count tokens
            total_tokens = len(metric_sequences.get('mean', []))
            
            result = CodeResponseResult(
                response_text=response_text,
                extracted_code=extracted_code,
                is_correct=is_passing,  # True if all tests pass
                num_tests_passed=num_passed,
                total_tests=total_tests,
                feedback=feedback,
                format_reward=format_rew,
                code_reward=code_rew,
                total_reward=total_rew,
                total_tokens=total_tokens,
                choice_index=choice_idx,
                aggregated_metrics=aggregated
            )
            results.append(result)
            
          # Prepare metrics record
            metrics_record = {
                'problem_id': problem.problem_id,
                'rollout_idx': choice_idx,
                'response_text': response_text,
                'extracted_code': extracted_code,
                'is_correct': is_passing,
                'num_tests_passed': num_passed,
                'total_tests': total_tests,
                #'format_reward': format_rew,
                #'code_reward': code_rew,
                #'total_reward': total_rew,
                'total_tokens': total_tokens,
                **aggregated
            }
            all_metrics_records.append(metrics_record)
        
        # Save metrics
        print(f"Saving {len(all_metrics_records)} metrics records to {self.metrics_file}")
        MetricsStorage.save_all_metrics_batch(
            all_metrics_records,
            self.metrics_file,
            use_compression=False
        )
        
        return results
    
    def apply_selection_strategies(self, results: List[CodeResponseResult]) -> Dict[str, CodeResponseResult]:
        """Apply different selection strategies"""
        strategies = {}
        
        if not results:
            return strategies
        
        # Random selection
        strategies['random'] = random.choice(results)
        
        # Oracle selection (best by reward if any passing, else best reward)
        passing_results = [r for r in results if r.is_correct]
        if passing_results:
            strategies['oracle'] = max(passing_results, key=lambda x: x.aggregated_metrics.get('mean_tail', 0))
        else:
            strategies['oracle'] = max(results, key=lambda x: x.total_reward)
        
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
    
    async def evaluate_problem(self, problem: CodeProblem) -> Dict[str, Any]:
        """Evaluate a single problem"""
        print(f"\nEvaluating Problem {problem.problem_id}")
        
        # Format prompt
        user_prompt = self._format_prompt(problem)
        
        # Prepare generation config
        gen_config = {
            "model": self.config.model_id,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # "max_tokens": self.config.max_completion_length,
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
        results = await self.evaluate_responses(problem, choices)
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
    
    async def evaluate_problem_with_semaphore(self, problem: CodeProblem, 
                                             semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Evaluate problem with concurrency control"""
        async with semaphore:
            return await self.evaluate_problem(problem)
    
    async def evaluate_problems_batch(self, problems: List[CodeProblem]) -> List[Dict[str, Any]]:
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
                print(f"  Problems with ≥1 passing: {batch_correct}/{len(current_batch)}")
                
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


# ============================================================================
# Command-Line Interface
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Modular Code Evaluation Framework")
    
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
    parser.add_argument("--num_samples", type=int, help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_url", type=str, help="OpenAI API base URL")
    
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = CodeEvaluationConfig.from_yaml(args.config)
    
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
    evaluator = CodeEvaluator(config, client)
    
    # Run evaluation
    print("\n" + "="*80)
    print("STARTING CODE EVALUATION")
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
