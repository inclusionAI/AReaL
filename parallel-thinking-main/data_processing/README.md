# Parallel Thinking Prototype

Here we prototype parallel thinking with a small example. Plase refer to `main_thread.xml` and `thread1.xml`.

In `main_thread.xml`, the main thread performs planning within `<think type="planning">...</think>` and launches threads with `<launch_therads>...</launch_threads>`. After both threads finish the allocated tasks, the thread results are fed back into the main thread with the `<thread_resolution>...</thread_solution>` block. Then the main thread continues reasoning. When the final answer is ready, the main thread stops reasoning and produces the final solution and answer.

For a sub thread, for example, in `thread_1.xml`, thread 1 recieves the allocated task and processes the task. 
# Utility
- Put the data `problem.jsonl`, and it need to contain keys `problem`, `CoT` and `answer` just as examples shown in `problem.jsonl`
- The output result will contain the following things:
`problem_index`, `original_problem`, `main_thread`, `thread_1`
- run 
    ```bash
    python3 agent.py
    ```
    to process single problem


    ```bash
    python3 batch_processor.py
    ```
    to process jsonl
# Caveat

If the data contains latex backlash like `\`, please change it to `\\`. I will add something to process it if this issue exist in original data.


