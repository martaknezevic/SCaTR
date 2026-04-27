import subprocess
import sys
import tempfile
import os
import resource
import json
from typing import List, NamedTuple, Tuple
from abc import ABC, abstractmethod
from typing import List, NamedTuple, Tuple


class ExecuteResultReturn(NamedTuple):
    is_passing: bool
    state: Tuple[bool]


class Executor(ABC):
    @abstractmethod
    def execute(
        self, func: str, tests: List[str], timeout: int = 5
    ) -> ExecuteResultReturn: ...

    @abstractmethod
    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool: ...

# RLIMIT settings (except address-space, set at runtime)
RLIMIT_SETTINGS = {
    resource.RLIMIT_CPU: (5, 5),  # 5 sec CPU time
    resource.RLIMIT_NOFILE: (16, 16),  # max 16 open files
}

# Cap for captured stdout/stderr (10 MB)
MAX_CAPTURE_BYTES = 250 * 1024 * 1024

class IOTestCase(NamedTuple):
    input_data: str
    expected_output: str
    name: str = ""

class ExecuteResult(NamedTuple):
    is_passing: bool
    state: Tuple[bool, ...]
    passed_tests: List[str]
    failed_tests: List[str]

class ContestExecutor:
    """Executor designed for competitive programming problems"""
    
    def __init__(self, timeout=10, mem_limit: int = 256):
        self.timeout = timeout
        self.memory_limit_mb = mem_limit
    
    def test_solution(self, solution_code: str, test_cases: List[IOTestCase]) -> ExecuteResult:
        """Test a solution against given test cases"""
        return self._execute_batch(solution_code, test_cases)
    
    def test_from_examples(self, solution_code: str, examples: dict) -> ExecuteResult:
        """
        Test solution using contest-style input/output examples
        
        Args:
            solution_code: The code to test
            examples: Dict with 'input' and 'output' lists
            
        Example:
            examples = {
                "inputs": ["10 1 14\n", "20 10 50\n"], 
                "outputs": ["1\n", "0\n"]
            }
        """
        #print(f"Testing solution with {len(examples.get('inputs', []))} example test cases...")
        test_cases = []
        inputs = examples.get("inputs", [])
        outputs = examples.get("outputs", [])
        
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            test_cases.append(IOTestCase(inp, out, f"Example {i+1}"))
        
        return self.test_solution(solution_code, test_cases)
    
    def test_single_example(self, solution_code: str, input_str: str, expected_output: str) -> ExecuteResult:
        """Test solution with a single input/output pair"""
        test_case = IOTestCase(input_str, expected_output, "Single Test")
        return self.test_solution(solution_code, [test_case])
    
    def _execute_batch(self, func: str, tests: List[IOTestCase]) -> ExecuteResult:
        passed_tests = []
        failed_tests = []
        
        # Create the batch execution script
        batch_script = self._create_batch_script(func, tests)
        print(batch_script)
        print(f"################################################################################")
        print(f"Executing batch script with {len(tests)} test cases")
        print(f"Batch script preview (first 500 chars):\n{batch_script}...")
        print(f"################################################################################")
        
        # Set up resource limits
        mem_bytes = self.memory_limit_mb * 1024 * 1024
        preexec_fn = self._make_preexec_fn(mem_bytes)
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(batch_script)
                script_path = f.name
            
            # Run the batch script with resource limits
            process = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                preexec_fn=preexec_fn
            )
            
            # Parse results
            results = self._parse_results(process.stdout, process.stderr, tests)
            for i, (success, output, error) in enumerate(results):
                test_case = tests[i]
                
                # Format input/output for display (truncate if too long)
                input_display = self._format_for_display(test_case.input_data)
                expected_display = self._format_for_display(test_case.expected_output)
                
                test_description = f"{test_case.name}: Input={input_display} Expected Output={expected_display}"
                
                if success:
                    passed_tests.append(f"{test_description} | Execution Details:=  PASSED")
                else:
                    # Truncate error messages if too long
                    if len(error) > MAX_CAPTURE_BYTES:
                        error = error[:MAX_CAPTURE_BYTES] + "...[truncated]"
                    failed_tests.append(f"{test_description} | Execution Details:= {error}")
            
            os.unlink(script_path)
            
        except subprocess.TimeoutExpired:
            for i, test in enumerate(tests):
                input_display = self._format_for_display(test.input_data)
                expected_display = self._format_for_display(test.expected_output)
                test_description = f"{test.name}: Input={input_display} Expected Output={expected_display}"
                failed_tests.append(f"{test_description} | Execution Details:= TIMEOUT")
        except Exception as e:
            for i, test in enumerate(tests):
                input_display = self._format_for_display(test.input_data)
                expected_display = self._format_for_display(test.expected_output)
                test_description = f"{test.name}: Input={input_display} Expected Output={expected_display}"
                failed_tests.append(f"{test_description} | Execution Details := ERROR - {e}")
        
        # Calculate results
        is_passing = (len(failed_tests) == 0) and (len(passed_tests) == len(tests))
        state = [True] * len(passed_tests) + [False] * len(failed_tests)
        
        return ExecuteResult(is_passing, state, passed_tests, failed_tests)
    
    def _format_for_display(self, text: str) -> str:
        """Format text for display, handling long inputs"""
        return repr(text)
    
    def _make_preexec_fn(self, memory_limit_bytes: int):
        """Return a pre-exec function that sets resource caps (without privilege dropping for compatibility)."""
        
        def _preexec():
            try:
                os.setpgrp()
                
                # Set memory limit
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
                
                # Set CPU and file limits
                for lim, (soft, hard) in RLIMIT_SETTINGS.items():
                    resource.setrlimit(lim, (soft, hard))
                    
            except Exception as e:
                # If resource limits fail, continue without them (for compatibility)
                pass

        return _preexec
    
    def _create_batch_script(self, func: str, tests: List[IOTestCase]) -> str:
        """Create a script that runs all test cases sequentially"""
        
        # Build test cases as Python code
        test_cases_code = "test_cases = [\n"
        for test in tests:
            input_repr = repr(test.input_data)
            expected_repr = repr(test.expected_output)
            name_repr = repr(test.name)
            test_cases_code += f"    ({input_repr}, {expected_repr}, {name_repr}),\n"
        test_cases_code += "]\n"
        
        return f'''
import sys
import io
import traceback
import json
from typing import *
import math

{test_cases_code}

results = []

for i, (input_data, expected_output, test_name) in enumerate(test_cases):
    try:
        # Redirect stdin and stdout
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        
        sys.stdin = io.StringIO(input_data)
        sys.stdout = io.StringIO()
        
        # Execute user code
{self._indent_code(func, 8)}
        
        # Get the output
        actual_output = sys.stdout.getvalue().strip()
        expected_clean = expected_output.strip()
        
        # Restore stdin/stdout
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        
        # Check result
        if actual_output == expected_clean:
            results.append({{"success": True, "output": actual_output, "error": ""}})
        else:
            error_msg = f"Expected: {{repr(expected_clean)}}, Got: {{repr(actual_output)}}"
            results.append({{"success": False, "output": actual_output, "error": error_msg}})
            
    except Exception as e:
        # Restore stdin/stdout
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        
        error_msg = f"{{type(e).__name__}}: {{str(e)}}"
        results.append({{"success": False, "output": "", "error": error_msg}})

# Output results as JSON
print("BATCH_RESULTS_JSON")
print(json.dumps(results))
print("BATCH_RESULTS_END")
'''
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces"""
        indent = " " * spaces
        return "\n".join(indent + line for line in code.split("\n"))
    
    def _parse_results(self, stdout: str, stderr: str, tests: List[IOTestCase]) -> List[Tuple[bool, str, str]]:
        """Parse the batch results from stdout"""
        lines = stdout.split('\n')
        
        # Debug output to see what we received
        if not stdout.strip():
            print(f"DEBUG - Empty stdout! stderr: {repr(stderr)}")
            return [(False, "", f"No output - {stderr}") for _ in tests]
        
        try:
            json_start = lines.index("BATCH_RESULTS_JSON")
            json_end = lines.index("BATCH_RESULTS_END")
            
            json_lines = lines[json_start + 1:json_end]
            json_str = '\n'.join(json_lines)
            
            json_results = json.loads(json_str)
            
            results = []
            for result in json_results:
                success = result["success"]
                output = result["output"]
                error = result["error"]
                results.append((success, output, error))
                    
            return results
            
        except (ValueError, IndexError, KeyError) as e:
            print(f"DEBUG - Parse error: {e}")
            print(f"DEBUG - Stdout: {repr(stdout[:200])}")
            print(f"DEBUG - Stderr: {repr(stderr)}")
            return [(False, "", f"Parse error - {stderr}") for _ in tests]
    
    
    def execute(self, func: str, tests: str, timeout: int = 20, imports = None) -> ExecuteResultReturn:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        
        exec_result = self.test_from_examples(func, tests)
        result = exec_result.is_passing
        state = exec_result.state # does not matter for this (in matters during training)
        
        return ExecuteResultReturn(result, state)

    def evaluate(self, name: str, func: str, tests: str, timeout: int = 20, imports = None) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        exec_result = self.test_from_examples(func, tests)
        return exec_result.is_passing



# ============================================================================
# SOLUTION FOR THE CARD SWAP PROBLEM
# ============================================================================

def solve_card_swap():
    """
    Solution for the card swap problem.
    
    Logic:
    - "abc" is already correct → YES
    - One swap away from "abc" → YES  
    - Two swaps needed → NO
    
    We can check this by counting how many positions are wrong:
    - 0 wrong positions → already "abc" → YES
    - 2 wrong positions → one swap fixes both → YES
    - 3 wrong positions → impossible with one swap → NO
    """
    
    card_solution = '''
t = int(input())
for _ in range(t):
    s = input().strip()
    target = "abc"
    
    # Count positions that differ from target
    diff_count = sum(1 for i in range(3) if s[i] != target[i])
    
    # 0 differences: already correct
    # 2 differences: one swap fixes both positions  
    # 3 differences: impossible with one swap
    if diff_count == 0 or diff_count == 2:
        print("YES")
    else:
        print("NO")
'''
    
    return card_solution

# Test the solution
if __name__ == "__main__":
    executor = ContestExecutor()
    
    # print("=== Card Swap Problem Test ===")
    
    solution = solve_card_swap()
    
    # Test with the provided sample
    sample_input = """6
abc
acb
bac
bca
cab
cba"""
    
    expected_output = """YES
YES
YES
NO
Yes
YES"""
    
    result = executor.test_from_examples(solution, {"inputs": [sample_input], "outputs": [expected_output]})
    print(f"\nResult: {'ALL PASSED' if result.is_passing else 'SOME FAILED'}")