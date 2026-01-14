# ASearcher-Web-QwQ

Launch Qwen2.5-72B-Instruct for LLM-as-Judge:

```
sudo python3 -m areal.launcher.slurm ../ASearcher/train/asearcher_reasoning.py --config ../ASearcher/configs/asearcher_web_qwq.yaml experiment_name=asearcher-qwen72b-inst-server-only trial_name=run0 cluster.n_nodes=1 allocation_mode=sglang.d2t4p1 actor.path=/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct 
```

Launch ASearcher-QwQ-Training:

```
python3 -m areal.launcher.slurm \
    ../ASearcher/train/asearcher_reasoning.py \
    --config ../ASearcher/configs/asearcher_web_qwq.yaml \
    experiment_name=gjx-asearcher-qwq-lite-train \
    trial_name=run1 cluster.n_nodes=6 allocation_mode=sglang.d2t8+d4t8 \
    actor.path=/storage/openpsi/models/Qwen__QwQ-32B \
    train_dataset.batch_size=32 \
    train_dataset.path=/storage/openpsi/users/gjx/ASearcher/data/ASearcher-LRM-35k.jsonl \
    judge_engine.experiment_name=asearcher-qwen72b-inst-server-only \
    judge_engine.trial_name=run0
```
