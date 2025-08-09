The hyperparameters given in gsm8k_grpo.yaml is the set that we found to achieve the highest max "grpo-eval/task_reward/avg" during training. You are free to try out more of the hyperparameters listed below!

Model: Qwen2.5-1.5b-Instruct

| lr         | weightdecay | optimizer | batchsize | max_val |
|------------|-------------|-----------|-----------|---------|
| 1.70E-05   | 0.017       | adam      | 4         | 0.79570 |
| 1.30E-05   | 0.015       | adam      | 8         | 0.79355 |
| 1.50E-05   | 0.01        | adam      | 4         | 0.79043 |
| 1.50E-05   | 0.02        | adam      | 4         | 0.78984 |
| 1.00E-05   | 0.02        | adam      | 4         | 0.78311 |
| 1.00E-05   | 0.01        | adam      | 8         | 0.78066 |

