import logging
import os
from datetime import datetime
from typing import List, Union, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import time

from executor.safe_executor import PyExecutor
from executor.code_contests_executor import ContestExecutor

EXECUTOR_TYPE = os.getenv("EXECUTOR_TYPE", "kodcode")
print("\n" + "=" * 60)
print("🚀 EXECUTOR API STARTUP")
print("=" * 60)
print(f"📋 Executor Type: {EXECUTOR_TYPE.upper()}")
print("=" * 60)


# ─ ensure log directory exists ───────────────────────────────────────────────
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)

# ─ build a timestamped filename ──────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"executor_stats_{timestamp}.log"
log_path = os.path.join(log_dir, log_filename)

# ─ configure your logger ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s:%(thread)d] %(levelname)s %(message)s",
    filename=log_path,
    filemode="a",
)
logger = logging.getLogger(__name__)

app = FastAPI()


class ExecutionRequest(BaseModel):
    code: str
    tests: Union[List[str], Dict[str, List[str]]]  # Accept both list and input/output dict
    timeout: int = 10


class ExecutionResponse(BaseModel):
    is_passing: bool
    feedback: str
    partial_test_cases: List[float]


@app.post("/execute", response_model=ExecutionResponse)
def execute_code(req: ExecutionRequest):
    # proc = psutil.Process()
    
    try:
        # Fix 1: Create an instance, not use the class directly
        if EXECUTOR_TYPE == "kodcode":
            exe = PyExecutor()
        elif EXECUTOR_TYPE == "code_contests":
            exe = ContestExecutor()
        else:
            raise ValueError(f"Unknown executor type: {EXECUTOR_TYPE}")
        # Fix 2: Call execute method and get the result object
        result = exe.execute(req.code, req.tests, timeout=req.timeout)

        # Fix 3: Extract values from the ExecuteResult object
        is_passing = result.is_passing
        feedback = result.feedback
        code = req.code
        
        print("="*100)
        print(code)
        print()
        print(result.feedback)
        print("="*100)

        # Fix 4: Convert boolean state to float partial scores (1.0 for pass, 0.0 for fail)
        partial = [1.0 if test_passed else 0.0 for test_passed in result.state]

        del exe

        return ExecutionResponse(
            is_passing=is_passing, feedback=feedback, partial_test_cases=partial
        )
    except Exception as e:
        print("=" * 100)
        print("Execution Error:")
        print(e)
        print("=" * 100)
        logger.exception("Execution error")
        raise HTTPException(status_code=500, detail=f"Execution error: {e}")