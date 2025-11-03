#!/bin/bash

# Bash script to run optimize_logprob_weights.py on all 10 turns sequentially
# Author: Auto-generated
# Usage: bash run_all_turns.sh [optional arguments to pass to Python script]
#
# Examples:
#   bash run_all_turns.sh
#   bash run_all_turns.sh --num-train-problems 20 --subsample-size 32
#   bash run_all_turns.sh --num-problems 5 --solver ECOS

set -e  # Exit on error

# Configuration
SCRIPT_DIR="/home/ubuntu/cactts/proposed_method"
RESULTS_DIR="/home/ubuntu/cactts/proposed_method/results"
PYTHON_SCRIPT="$SCRIPT_DIR/optimize_logprob_weights.py"
CONDA_ENV="oss"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/run_all_turns.log"
echo "============================================================" | tee "$LOG_FILE"
echo "Starting optimization runs at $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Parse optional arguments (these will be passed to the Python script)
EXTRA_ARGS="$@"
if [ -n "$EXTRA_ARGS" ]; then
    echo -e "${BLUE}Extra arguments: $EXTRA_ARGS${NC}" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Counter for tracking progress
TOTAL_TURNS=10
SUCCESS_COUNT=0
FAIL_COUNT=0
declare -a FAILED_TURNS

# Run optimization for each turn
for TURN_NUM in {1..10}; do
    TURN_NAME="turn$TURN_NUM"
    echo -e "${BLUE}============================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}Processing $TURN_NAME ($TURN_NUM/$TOTAL_TURNS)${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}============================================================${NC}" | tee -a "$LOG_FILE"

    # Run the optimization with --turn argument
    START_TIME=$(date +%s)

    if conda run -n "$CONDA_ENV" python "$PYTHON_SCRIPT" --turn "$TURN_NAME" $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "${GREEN}✓ Successfully completed $TURN_NAME in ${DURATION}s${NC}" | tee -a "$LOG_FILE"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "${RED}✗ Failed to complete $TURN_NAME after ${DURATION}s${NC}" | tee -a "$LOG_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_TURNS+=("$TURN_NAME")
        # Continue to next turn even if this one failed
    fi

    echo "" | tee -a "$LOG_FILE"
done

# Summary
echo -e "${BLUE}============================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}SUMMARY${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}============================================================${NC}" | tee -a "$LOG_FILE"
echo "Total turns processed: $TOTAL_TURNS" | tee -a "$LOG_FILE"
echo -e "${GREEN}Successful: $SUCCESS_COUNT${NC}" | tee -a "$LOG_FILE"

if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}Failed: $FAIL_COUNT${NC}" | tee -a "$LOG_FILE"
    echo -e "${YELLOW}Failed turns: ${FAILED_TURNS[*]}${NC}" | tee -a "$LOG_FILE"
else
    echo "Failed: 0" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Finished at $(date)" | tee -a "$LOG_FILE"
echo "Full log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Exit with error if any turn failed
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi

echo -e "${GREEN}All turns completed successfully!${NC}"
