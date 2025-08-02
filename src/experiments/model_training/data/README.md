Trajectories are taken from tau2 evaluations rollouts.
Contains all domain, all tasks, 4 trials per tasks.
3 models:
- GPT-4.1-mini, o4-mini, claude-3.5-sonnet
Model not selected: GPT-4.1 (worst performing one)

Trajectories belong to train or test based on the domain split definition.
Succesfull trajectories are selected and used to create a train and test set in openai format.
- train_full-v1.jsonl
- test_full-v1.jsonl

Running training using Openai API.
JobID: ftjob-4E0M0su1djHZJhnNjMR9jkh7
Base model: gpt-4.1-2025-04-14
Trained model: ft:gpt-4.1-2025-04-14:sierra:tau2-full-sft:BzrYdMNm

Autoselected hyperparams.
Training time: ~1 day.

Base model and ft model are then tested on all test sets (1 trial per task) and results are saved in the results folder.


Note:
- I did not clean the data from messages that contain both content and tool calls. This should be done (better for )