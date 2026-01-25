import ast
import signal
from typing import List

import astunparse

from executor_types import ExecuteResult, Executor
from executor_utils import function_with_timeout

import subprocess
import re
import tempfile
import textwrap
import os

MAIN_BLOCK = textwrap.dedent("""
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCases)
    unittest.TextTestRunner(verbosity=2).run(suite)
""")


class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 5, imports= None) -> ExecuteResult:
        # Combine function code and assert statement
        imports = "from typing import *" + "\n" + (imports if imports is not None else "") 
        func_test_list = [f"{imports}\n{func}\n{test}" for test in tests]

        # Run the tests and collect the results
        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):
            try:
                function_with_timeout(exec, (func_test_list[i], globals()), timeout)

                success_tests += [tests[i]]
            except Exception:
                output = get_output(func, tests[i], timeout=timeout)
                failed_tests += [f"{tests[i]} # output: {output}"]
                is_passing = False

        state = []
        for test in tests:
            if test in success_tests:
                state += [True]
            else:
                state += [False]

        state = tuple(state)

        feedback = "Tested passed:"
        for test in success_tests:
            feedback += f"\n{test}"
        feedback += "\n\nTests failed:"
        for test in failed_tests:
            feedback += f"\n{test}"
        
        return ExecuteResult(is_passing, feedback, state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5, imports=None) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        imports = "from typing import *" + "\n" + (imports if imports is not None else "")

        code = f"""{func}

{test}

check({name})
    """
        code = imports + "\n" + code

        try:

            function_with_timeout(exec, (code, globals()), timeout)

            return True
        except Exception:
            return False

    def run_unittest_code_sandboxed(self, code_str, timeout=5):
        # Auto-append main block only if it's not already present
        if '__name__' not in code_str:
            code_str = code_str.rstrip() + "\n\n" + MAIN_BLOCK

        # Write to a temporary .py file
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as tmp:
            tmp.write(code_str)
            tmp_filename = tmp.name

        try:
            # Run the test file in a sandboxed subprocess
            result = subprocess.run(
                ["python", tmp_filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True,
            )
        except subprocess.TimeoutExpired:
            os.remove(tmp_filename)
            return "Timeout: Code execution took too long.", [], [True]

        os.remove(tmp_filename)

        output = result.stdout + result.stderr

        # Parse passed and failed tests from output
        passed = re.findall(r'\.\.\. ok', output)
        failed = re.findall(r'\.\.\. FAIL', output)

        return output, passed, failed

    def execute_bigcode_bench(self, func: str, tests: str, timeout: int = 20, imports = None) -> ExecuteResult:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """

        imports = "from typing import *" + "\n" + (imports if imports is not None else "") 
        code_str = f"{imports}\n{func}\n{tests}"

        output, passed, failed = self.run_unittest_code_sandboxed(code_str, timeout=timeout)

        feedback = "Test Suite:\n\n" + tests + "\n\n" + "Test Output:\n\n" + output

        state = tuple([True] * len(passed) + [False] * len(failed)) # Doesnt matter

        result = False if (len(failed) > 0 or len(passed) == 0 or 'FAIL' in output) else True
        
        if not result:
            print(f'output: {output}')
            print(f'result: {result}')
            print(f'FUNC:\n{func}')
            print("=" * 100 + "\n")

        return ExecuteResult(result, feedback, state)

    def evaluate_bigcode_bench(self, name: str, func: str, tests: str, timeout:20, imports = None) -> ExecuteResult:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        exec_result = self.execute_bigcode_bench(func, tests, timeout=timeout, imports=imports)
        return exec_result.is_passing


def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left  # type: ignore
    except:
        call_str = ast_parsed.body[0].test  # type: ignore

    return astunparse.unparse(call_str).strip()


def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        exec(f"from typing import *\n{func}", globals())
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, globals()), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    # Test the function
    func = "def add(a, b):\n    while True:\n        x = 1\n    return a + b"
    func = "def add(a: str, b):\n    return a + b"

    tests = ["assert add(1, 2) == 3", "assert add(1, 2) == 3"]
    print(PyExecutor().execute(func, tests, timeout=1))