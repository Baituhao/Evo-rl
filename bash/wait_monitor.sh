# for DLC to monitor multiple background processes
# Get all background job PIDs
PIDS=($(jobs -p))

if [ ${#PIDS[@]} -eq 0 ]; then
    echo "No background processes to monitor"
    exit 0
fi

echo "Monitoring ${#PIDS[@]} background process(es): ${PIDS[*]}"

# Array to store exit codes
declare -a EXIT_CODES
OVERALL_EXIT_CODE=0

# Wait for each process and collect exit codes
for PID in "${PIDS[@]}"; do
    if kill -0 "${PID}" 2>/dev/null; then
        echo "Waiting for process ${PID}..."
        wait "${PID}" 2>/dev/null
        EXIT_CODE=$?

        if [ "${EXIT_CODE}" -ne 0 ]; then
            # If wait fails (process not a child), poll until it finishes
            while kill -0 "${PID}" 2>/dev/null; do
                sleep 1
            done
            wait "${PID}" 2>/dev/null
            EXIT_CODE=$?
        fi
    else
        echo "Process ${PID} already finished or not found"
        EXIT_CODE=127
    fi

    EXIT_CODES+=("${EXIT_CODE}")

    if [ "${EXIT_CODE}" -ne 0 ]; then
        echo "Process ${PID} failed with exit code ${EXIT_CODE}"
        OVERALL_EXIT_CODE="${EXIT_CODE}"
    else
        echo "Process ${PID} completed successfully"
    fi
done

echo "All processes finished. Exit codes: ${EXIT_CODES[*]}"

if [ "${OVERALL_EXIT_CODE}" -ne 0 ]; then
    echo "At least one process failed with exit code ${OVERALL_EXIT_CODE}"
    exit "${OVERALL_EXIT_CODE}"
fi

echo "All processes completed successfully"
