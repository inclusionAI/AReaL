

1. data/tau2:
    domains：包含多个场景，目前只用了ariline
        场景包含：
            db.json: 数据库初始状态，需要保留
            policy.md: 会在agent的prompt里使用，需要保留或者怎么着跟agent.py放一起
            task.json: 测试任务，可以删除
    results：
        可以删掉

    user_simulator:
        会用在user simulator的prompt里，需要保留或者怎么着跟user_simulator.py放一起

2. data_model:
    一些类的定义，需要保留。

3. domains:
    一些场景相关的环境定义和tools

4. environment:
    一些通用的环境定义和tools相关的工具

5. evaluator：
    几种不同的evaluation metric。

6. utils：
    一些辅助函数，这块我不太确定有哪些没用到

7. orchestrator.py
    agent, user 和环境的交互逻辑，基本跟原始tau2保持一致

8. user_simulator.py
    user simulator的逻辑，逻辑跟基本跟原始tau2保持一致

9. agent.py
    agent生成，调用areal engine

10. workflow.py  / workflow_megatron.py
    fsdp 和 megatron 的训练脚本


怎么跑起来：
    1. 启动一个sglang server当作 user simulator
    2. 指定 user_base_url 和 user_model

    ```
    PYTHONPATH="modules/AReaL/" \
    python3 -m areal.launcher.slurm tau2_train/workflow_megatron.py \
    --config tau2_train/train_debug_megatron.yaml \
    +user_base_url=http://33.180.161.71:30000/v1 \
    +user_model=/storage/openpsi/models/Qwen3-Next-80B-A3B-Instruct_031825d6b716d454/ \
    ```



