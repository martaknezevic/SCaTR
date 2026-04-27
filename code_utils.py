import ast
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Optional, Set

import jsonlines
import numpy as np
import torch
import yaml
from transformers import TrainerCallback
from transformers import set_seed as hf_set_seed

import wandb


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items += [item]
    return items


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    hf_set_seed(seed)

    print(f"Seeding done with seed = {seed}")


def parse_between_output_tags_old(text):
    match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text


def parse_between_output_tags_old_v2(text):
    matches = re.findall(r"<output>(.*?)</output>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text


def parse_between_output_tags(text):
    pattern = r"<output>((?:(?!<output>).)*?)</output>"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return matches[-1].strip()
    return text


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def format_reward(response: str) -> float:
    cumulative_reward = tag_format_reward(response) + tag_count_reward(response)
    return cumulative_reward / 2.0


def tag_format_reward(response):
    # TODO: Check this once
    format = re.search(r"(.*?)</think>(\s*)<output>(.*?)</output>", response, re.DOTALL)

    if not format:
        return 0.0
    else:
        return 1.0


def tag_count_reward(text) -> float:
    """Reward function that checks if we produce the desired number of think and output tags."""

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("</think>") == 1:
            count += 0.33
        if text.count("<output>") == 1:
            count += 0.33
        if text.count("</output>") == 1:
            count += 0.34
        return count

    return count_tags(text)

class ConfigUploadCallback(TrainerCallback):
    def __init__(self, config_to_upload):
        self.config_to_upload = config_to_upload
        self.uploaded = False

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.uploaded and wandb.run is not None:
            wandb.config.update(self.config_to_upload)
            print("✅ Uploaded full config to wandb")
            self.uploaded = True

class FunctionExtractor(ast.NodeVisitor):
    def __init__(self, target: str):
        self.target = target
        self.function_defs: Dict[str, ast.FunctionDef] = {}
        self.called_functions: Set[str] = set()
        self.required_functions: Set[str] = set()
        self.imports: Set[ast.AST] = set()

    def visit_Import(self, node: ast.Import):
        self.imports.add(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        self.imports.add(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_defs[node.name] = node
        self.generic_visit(node)

    def collect_called_functions(self, node: ast.FunctionDef):
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    self.called_functions.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    # Handles cases like module.func()
                    self.called_functions.add(child.func.attr)

    def resolve_dependencies(self, func_name: str):
        if func_name not in self.function_defs or func_name in self.required_functions:
            return
        node = self.function_defs[func_name]
        self.required_functions.add(func_name)
        self.collect_called_functions(node)
        for called in self.called_functions - self.required_functions:
            self.resolve_dependencies(called)


def extract_function_with_dependencies(code: str, func_name: str) -> Optional[str]:
    tree = ast.parse(code)
    extractor = FunctionExtractor(func_name)
    extractor.visit(tree)
    extractor.resolve_dependencies(func_name)

    lines = code.splitlines()
    result_blocks = []

    # Collect import lines
    for node in sorted(extractor.imports, key=lambda n: n.lineno):
        start = node.lineno - 1
        end = node.end_lineno
        result_blocks.append("\n".join(lines[start:end]))

    # Collect function definitions
    required_defs = [
        extractor.function_defs[name] for name in extractor.required_functions
    ]
    required_defs.sort(key=lambda n: n.lineno)

    for node in required_defs:
        start = node.lineno - 1
        end = node.end_lineno
        result_blocks.append("\n".join(lines[start:end]))

    return "\n\n".join(result_blocks) if result_blocks else None