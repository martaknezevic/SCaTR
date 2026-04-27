import grp
import json
import os
import pwd
import resource
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Tuple

# RLIMIT settings (except address-space, set at runtime)
RLIMIT_SETTINGS = {
    resource.RLIMIT_CPU: (5, 5),  # 5 sec CPU time
    resource.RLIMIT_NOFILE: (16, 16),  # max 16 open files
}

# Cap for captured stdout/stderr (10 MB)
MAX_CAPTURE_BYTES = 10 * 1024 * 1024


# Result of execution: pass/fail, test states, plus detailed lists
class ExecuteResult(NamedTuple):
    is_passing: bool
    state: Tuple[bool, ...]
    passed_tests: List[str]
    failed_tests: List[str]


# Abstract executor interface
class Executor(ABC):
    @abstractmethod
    def execute(
        self, func: str, tests: List[str], timeout: int = 5, memory_limit_mb: int = 200
    ) -> ExecuteResult: ...

    @abstractmethod
    def evaluate(
        self,
        name: str,
        func: str,
        test: str,
        timeout: int = 5,
        memory_limit_mb: int = 200,
    ) -> bool: ...


def _make_preexec_fn(memory_limit_bytes: int):
    """
    Return a pre-exec function that sets resource caps and drops privileges.
    """

    def _preexec():
        try:
            # Create new process group to make killing easier
            os.setpgrp()

            # address-space limit
            resource.setrlimit(
                resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes)
            )
            # other limits
            for lim, (soft, hard) in RLIMIT_SETTINGS.items():
                resource.setrlimit(lim, (soft, hard))
            # clear supplementary groups
            try:
                os.setgroups([])
            except Exception:
                pass
            # drop to nobody:nogroup
            try:
                gid = grp.getgrnam("nogroup").gr_gid
                os.setgid(gid)
            except Exception:
                pass
            try:
                uid = pwd.getpwnam("nobody").pw_uid
                os.setuid(uid)
            except Exception:
                pass
        except Exception:
            os._exit(1)

    return _preexec


def _kill_process_group(process):
    """Forcefully kill a process and its children"""
    if process.poll() is None:  # Process is still running
        try:
            # Kill the entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            # Process might have already died
            pass

        # Wait a bit for cleanup
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            # Force kill if still alive
            try:
                process.kill()
                process.wait(timeout=1)
            except:
                pass


def _generate_unique_filename(prefix: str = "test", extension: str = "py") -> str:
    """
    Generate a unique filename using UUID and timestamp to avoid conflicts.
    Format: prefix_timestamp_pid_uuid.extension

    This ensures uniqueness even with:
    - Multiple concurrent requests
    - Multiple processes
    - Rapid sequential execution
    """
    timestamp = int(time.time() * 1000000)  # microsecond precision
    unique_id = str(uuid.uuid4())[:8]  # First 8 chars of UUID
    pid = os.getpid()  # Add process ID for extra uniqueness
    return f"{prefix}_{timestamp}_{pid}_{unique_id}.{extension}"


def _safe_create_script(td: str, prefix: str, content: str) -> str:
    """
    Safely create a script file with a unique name and verify it doesn't exist.
    Returns the full path to the created file.
    """
    max_attempts = 5
    for attempt in range(max_attempts):
        filename = _generate_unique_filename(prefix, "py")
        script_path = os.path.join(td, filename)

        # Check if file already exists (very unlikely but possible)
        if not os.path.exists(script_path):
            with open(script_path, "w") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            return script_path

        # If file exists, wait a tiny bit and try again
        time.sleep(0.001)  # 1ms delay

    # Fallback: use a guaranteed unique name with counter
    fallback_filename = (
        f"{prefix}_{int(time.time()*1000000)}_{os.getpid()}_{uuid.uuid4()}.py"
    )
    script_path = os.path.join(td, fallback_filename)
    with open(script_path, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    return script_path


def _create_test_wrapper(func: str, test: str) -> str:
    """
    Create a test script that cleanly separates test results from user output.
    Uses structured output to communicate results reliably.
    """
    # Indent the user function properly
    indented_func = "\n".join("    " + line for line in func.splitlines())
    indented_test = "\n".join("    " + line for line in test.splitlines())

    return f"""
import sys
import traceback
import json
from typing import *

# Standard libraries
import os
import re
import math
import random
import string

# Custom stdout capture to separate user prints from test results
class OutputCapture:
    def __init__(self):
        self.captured_output = []
        self.original_stdout = sys.stdout
        
    def write(self, text):
        self.captured_output.append(text)
        
    def flush(self):
        pass
        
    def get_output(self):
        return ''.join(self.captured_output)

# Set up output capture
output_capture = OutputCapture()
original_stdout = sys.stdout
sys.stdout = output_capture

test_result = {{"status": "unknown", "test": {json.dumps(test)}, "user_output": "", "error": None, "error_type": None}}

try:
    # Execute user function
{indented_func}

    # Execute the test
{indented_test}
    
    # If we get here, test passed
    user_output = output_capture.get_output().strip()
    test_result.update({{
        "status": "passed",
        "user_output": user_output
    }})
    
except AssertionError as e:
    # Test failed due to assertion
    user_output = output_capture.get_output().strip()
    error_msg = str(e) if str(e) else "Assertion failed"
    
    test_result.update({{
        "status": "failed",
        "user_output": user_output,
        "error": error_msg,
        "error_type": "AssertionError"
    }})
    
except Exception as e:
    # Other error occurred
    user_output = output_capture.get_output().strip()
    test_result.update({{
        "status": "error",
        "user_output": user_output,
        "error": str(e),
        "error_type": type(e).__name__,
        "traceback": traceback.format_exc()
    }})

finally:
    # Restore stdout and output result
    sys.stdout = original_stdout
    print("TEST_RESULT_JSON:" + json.dumps(test_result))
    
    # Exit with appropriate code
    if test_result["status"] == "passed":
        sys.exit(0)
    else:
        sys.exit(1)
"""


def _parse_test_result(
    stdout: str, stderr: str, returncode: int, test: str
) -> Dict[str, Any]:
    """
    Parse the structured test result from the wrapper script output.
    Falls back to legacy parsing if structured output is not found.
    """

    # Look for our structured result
    lines = stdout.split("\n")
    for line in lines:
        if line.startswith("TEST_RESULT_JSON:"):
            try:
                result_json = line[17:]  # Remove prefix
                return json.loads(result_json)
            except json.JSONDecodeError:
                pass

    # Fallback to legacy parsing
    user_output = stdout.strip()
    if returncode == 0:
        return {
            "status": "passed",
            "test": test,
            "user_output": user_output,
            "error": None,
            "error_type": None,
        }
    else:
        return {
            "status": "failed" if "assert" in test.lower() else "error",
            "test": test,
            "user_output": user_output,
            "error": stderr.strip() or "Unknown error",
            "error_type": "Unknown",
        }


def _get_assertion_details(func: str, test: str, preexec_fn, timeout: int) -> str:
    """
    Try to extract the actual value from a failed assertion by evaluating the LHS.
    This helps provide better error messages for assertion failures.
    """
    try:
        if not (test.strip().startswith("assert ") and "==" in test):
            return "Could not extract assertion details"

        expr = test.strip()[7:]  # Remove 'assert '
        if "==" not in expr:
            return "Non-equality assertion"

        lhs, rhs = expr.split("==", 1)
        code_to_evaluate = lhs.strip()

        with tempfile.TemporaryDirectory() as td:
            # Create evaluation script with basic imports
            indented_func = "\n".join("    " + line for line in func.splitlines())
            indented_eval = "    " + f"result = {code_to_evaluate}"

            script_content = f"""
# Standard libraries
import os
import sys
import re
import math
import random
import string
from typing import *

try:
    # User's function
{indented_func}

    # Evaluate the expression
{indented_eval}
    print(f"Actual value: {{repr(result)}}")
except Exception as e:
    print(f"Error evaluating expression: {{e}}")
"""
            eval_script = _safe_create_script(td, "assertion_eval", script_content)

            process = subprocess.Popen(
                [sys.executable, "-u", eval_script],
                preexec_fn=preexec_fn,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return (
                    stdout.strip() or stderr.strip() or "Could not evaluate expression"
                )
            except subprocess.TimeoutExpired:
                _kill_process_group(process)
                return "Timeout while evaluating expression"
            finally:
                _kill_process_group(process)

    except Exception as e:
        return f"Error during assertion analysis: {e}"


class PyExecutor(Executor):
    """Sandboxed Python executor with robust output handling and better error reporting"""

    def execute(
        self,
        func: str,
        tests: List[str],
        timeout: int = 5,
        memory_limit_mb: int = 200,
    ) -> ExecuteResult:
        passed_tests: List[str] = []
        failed_tests: List[str] = []

        mem_bytes = memory_limit_mb * 1024 * 1024
        preexec_fn = _make_preexec_fn(mem_bytes)

        for test_idx, test in enumerate(tests):
            process = None
            try:
                with tempfile.TemporaryDirectory() as td:
                    # Create script with unique filename
                    script_content = _create_test_wrapper(func, test)
                    script_path = _safe_create_script(
                        td, f"test_{test_idx}", script_content
                    )

                    # Start process with Popen for better control
                    process = subprocess.Popen(
                        [sys.executable, "-u", script_path],
                        preexec_fn=preexec_fn,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )

                    try:
                        # Wait for completion with timeout
                        stdout, stderr = process.communicate(timeout=timeout)
                        returncode = process.returncode

                        # Parse structured result
                        result = _parse_test_result(stdout, stderr, returncode, test)

                        if result["status"] == "passed":
                            # Format successful test
                            if result["user_output"]:
                                output = result["user_output"]
                                if len(output) > MAX_CAPTURE_BYTES:
                                    output = (
                                        output[:MAX_CAPTURE_BYTES] + "...[truncated]"
                                    )
                                passed_tests.append(f"{test}  # printed: {output}")
                            else:
                                passed_tests.append(test)

                        else:
                            # Format failed test with enhanced error info
                            error_info = result["error"] or "Unknown error"

                            # For assertion errors, try to get more details
                            if (
                                result["error_type"] == "AssertionError"
                                and result["status"] == "failed"
                            ):
                                assertion_details = _get_assertion_details(
                                    func, test, preexec_fn, timeout
                                )
                                if (
                                    assertion_details
                                    and "Could not" not in assertion_details
                                ):
                                    error_info = assertion_details

                            # Truncate long error messages
                            if len(error_info) > MAX_CAPTURE_BYTES:
                                error_info = (
                                    error_info[:MAX_CAPTURE_BYTES] + "...[truncated]"
                                )

                            failure_msg = f"{test}  # {result['status']}: {error_info}"
                            failed_tests.append(failure_msg)

                    except subprocess.TimeoutExpired:
                        failed_tests.append(f"{test}  # timeout after {timeout}s")
                    finally:
                        # Always ensure process is killed
                        if process:
                            _kill_process_group(process)

            except Exception as e:
                failed_tests.append(f"{test}  # execution error: {e}")
            finally:
                # Final cleanup - make sure process is dead
                if process:
                    _kill_process_group(process)

        # Calculate test state
        is_passing = len(passed_tests) == len(tests) and len(failed_tests) == 0
        state = state = tuple(
            test.split("  #")[0] in [t.split("  #")[0] for t in passed_tests]
            for test in tests
        )

        def filter_imports(text):
            lines = text.split("\n")
            filtered_lines = [
                line
                for line in lines
                if not line.strip().startswith(("import ", "from "))
            ]
            return "\n".join(filtered_lines)

        return ExecuteResult(is_passing, state, passed_tests, failed_tests)

    def evaluate(
        self,
        name: str,
        func: str,
        test: str,
        timeout: int = 5,
        memory_limit_mb: int = 200,
    ) -> bool:
        res = self.execute(func, [test], timeout, memory_limit_mb)
        return res.state[0]


if __name__ == "__main__":
    example_func = '''
def example_func(x: int) -> int:
    """Returns the square of x"""
    print(f"Computing square of {x}")
    return x * x
'''

    test_function = "z = 3\nassert example_func(z) == 16\n\nassert example_func(2) == 4"
    tests = [
        "assert example_func(2) == 4",
        "assert example_func(3) == 9",
        "assert example_func(0) == 0",
        "assert example_func(0) == 4",  # This will fail
        'print("Square of 5 is", example_func(5))',
        # "while True: pass",  # Commented out to avoid hanging
        # 'x = "x" * (150 * 1024 * 1024); print(len(x))',  # Commented out for memory
        test_function,
    ]
    executor = PyExecutor()
    result = executor.execute(example_func, tests, timeout=10, memory_limit_mb=100)
    print("\n=== Raw Results ===")
    print("Passed tests:", result.passed_tests)
    print("Failed tests:", result.failed_tests)