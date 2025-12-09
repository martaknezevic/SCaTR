import asyncio

import aiohttp
import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from code_utils import (
    extract_function_with_dependencies,
    format_reward,
    parse_between_output_tags,
)

# Global tokenizer variable
tokenizer = None


def init_tokenizer(model_name):
    """
    Must be called once from the main process to initialize global tokenizer.
    """
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def process_dataset(
    dataset_path,
    data_original,
    system_prompt,
    num_samples=None,
    max_completion_length=2048,
) -> Dataset:
    """
    Processes the input dataset to format prompts and extract test cases.

    Args:
        data_original (list): Original data samples.
        system_prompt (str): System prompt to prepend to each sample.
        max_completion_length (int): Max tokens allowed for code generation.

    Returns:
        Dataset: A Hugging Face Dataset object with processed prompts and test cases.
    """
    data = []

    additional_prompt = (
        f"\nNote that if you don't complete thinking and the code generation "
        f"within {max_completion_length} tokens, you will get a zero reward."
    )
    # additional_prompt = ""

    if num_samples is not None and len(data_original) > num_samples:
        np.random.seed(42)
        data_original = np.random.choice(data_original, num_samples, replace=False)
        print(f"Num data samples picked: {len(data_original)}")

    for _, item in tqdm(enumerate(data_original), total=len(data_original)):
        updated_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["prompt"] + additional_prompt},
                {"role": "assistant", "content": "Let me think step by step.\n<think>"},
            ],
            add_generation_prompt=False,
            continue_final_message=True,
            tokenize=False,
        )

        test_cases = None
        if "HumanEval" in dataset_path:
            test_cases = [
                line.strip().replace("candidate", item["entry_point"])
                for line in item["test"].split("\n")
                if "assert" in line
            ]
        elif "kodcode_light_rl" in dataset_path:
            test_cases = item["test"]
        elif "code_contests" in dataset_path:
            test_cases = item["public_tests"]
        else:
            raise ValueError("Please enter the correct dataset.")

        data_sample = {
            "prompt": updated_prompt,
            "test_cases": test_cases,
            "system_prompts": system_prompt,
            "original_prompts": item["prompt"] + additional_prompt,
            "entry_points": item.get("entry_point"),
        }

        data.append(data_sample)

    return Dataset.from_list(data)


async def compute_reward_async(
    prompts,
    completions,
    test_cases,
    system_prompts,
    original_prompts,
    entry_points,
    max_concurrent=6,
    **kwargs,
) -> list:

    # Pre-compute formatting rewards
    formatting_rewards = [format_reward(completion) for completion in completions]

    async def process_item(session, completion, tests, entry_point):
        try:
            cur_func_impl = parse_between_output_tags(completion)
            if entry_point is not None:
                cur_func_impl = extract_function_with_dependencies(
                    cur_func_impl, entry_point
                )
            async with session.post(
                "http://localhost:8000/execute",
                json={"code": cur_func_impl, "tests": tests, "timeout": 5},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return (
                    result["is_passing"],
                    result["feedback"],
                    result["partial_test_cases"],
                )

        except Exception:
            return (
                False,
                f"Output not in the correct format or code timed out. Please recheck.",
                [0.0],
            )

    # Limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_process(session, completion, tests, entry_point):
        async with semaphore:
            return await process_item(session, completion, tests, entry_point)

    # Process all items concurrently - gather() preserves order automatically
    async with aiohttp.ClientSession() as session:
        parallel_results = await asyncio.gather(
            *[
                bounded_process(session, completion, tests, entry_point)
                for completion, tests, entry_point in zip(
                    completions, test_cases, entry_points
                )
            ]
        )

    # Build final results
    rewards, feedback_list, code_reward_list, format_reward_list = [], [], [], []
    num_test_cases_passed_list, num_total_tests_list = [], []
    total_cases_passed = 0

    for i, (tests, (is_passing, feedback, partial_test_cases)) in enumerate(
        zip(test_cases, parallel_results)
    ):          
        num_tests = len(tests.get("input")) if isinstance(tests, dict) else len(tests)
        num_total_tests_list.append(num_tests)
        feedback_list.append(feedback)
        total_cases_passed += int(is_passing)

        code_reward = np.mean(np.asarray(partial_test_cases, dtype=np.float32))
        formatting_reward = formatting_rewards[i]

        code_reward_list.append(code_reward)
        format_reward_list.append(formatting_reward)
        rewards.append(code_reward + formatting_reward)
        num_test_cases_passed_list.append(
            np.sum(np.asarray(partial_test_cases, dtype=np.float32))
        )

    return (
        rewards,
        feedback_list,
        system_prompts,
        original_prompts,
        code_reward_list,
        format_reward_list,
        num_test_cases_passed_list,
        num_total_tests_list,
    )


# Sync wrapper
def compute_reward(
    prompts,
    completions,
    test_cases,
    system_prompts,
    original_prompts,
    entry_points,
    **kwargs,
) -> list:
    return asyncio.run(
        compute_reward_async(
            prompts,
            completions,
            test_cases,
            system_prompts,
            original_prompts,
            entry_points,
            **kwargs,
        )
    )


# Test code to verify order preservation
if __name__ == "__main__":
    import random
    import time

    # Mock functions for testing
    def format_reward(completion):
        return random.uniform(0.1, 0.5)  # Random formatting score

    def parse_between_output_tags(completion):
        return f"def test_func_{completion}(): return {completion}"

    def extract_function_with_dependencies(code, entry_point):
        return code

    # Mock server response for testing
    class MockResponse:
        def __init__(self, completion_id):
            self.completion_id = completion_id

        def raise_for_status(self):
            pass

        async def json(self):
            # Simulate processing time with random delay
            await asyncio.sleep(random.uniform(0.1, 0.5))
            return {
                "is_passing": random.choice([True, False]),
                "feedback": f"Test result for completion {self.completion_id}",
                "partial_test_cases": [random.uniform(0, 1) for _ in range(3)],
            }

    # Mock aiohttp session
    class MockSession:
        def post(self, url, json=None, timeout=None):
            completion_id = json["code"].split("_")[-1].split("(")[0]
            return MockResponse(completion_id)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    # Patch aiohttp for testing
    original_client_session = aiohttp.ClientSession
    aiohttp.ClientSession = MockSession

    async def test_order_preservation():
        print("Testing order preservation...")

        # Create test data with identifiable order
        completions = [f"completion_{i}" for i in range(50)]
        test_cases = [[f"test_{i}_case_{j}" for j in range(3)] for i in range(50)]
        entry_points = [f"entry_{i}" for i in range(50)]

        # Mock other parameters
        prompts = ["prompt"] * 50
        system_prompts = ["system"] * 50
        original_prompts = ["original"] * 50

        print(f"Input order: {completions}")

        start_time = time.time()
        results = await compute_reward_async(
            prompts,
            completions,
            test_cases,
            system_prompts,
            original_prompts,
            entry_points,
            max_concurrent=3,
        )
        end_time = time.time()

        # Check that feedback preserves order
        feedback_list = results[1]
        print(f"Feedback order: {[f.split()[-1] for f in feedback_list]}")

        # Verify order is preserved
        for i, feedback in enumerate(feedback_list):
            expected = f"completion_{i}"
            if expected not in feedback:
                print(f"❌ Order mismatch at index {i}!")
                return False

        print(f"✅ Order preserved correctly!")
        print(f"⏱️  Processed {len(completions)} items in {end_time - start_time:.2f}s")
        return True

    # Restore original aiohttp after test
    def cleanup():
        aiohttp.ClientSession = original_client_session

    try:
        # Run the test
        asyncio.run(test_order_preservation())
    finally:
        cleanup()

    print(
        "\n🧪 Test completed! The async version preserves order even with concurrent processing."
    )