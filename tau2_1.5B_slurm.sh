WANDB_API_KEY=local-e94a768686930cfc13051e562b807fc2d56bc4dd  \
WANDB_BASE_URL=http://8.150.1.98:8080 \
SERPER_API_KEY=607608ef262aa0020bd69512b4c6a60eb53fb4a5 \
JINA_API_KEY=jina_89db187a69f44004ae58fd7ff1615232FbzlayNZYcsAHd7zW3-1H8KGXmIF \
PYTHONPATH="modules/AReaL/" \
python3 -m areal.launcher.slurm tau2_train/workflow.py \
    --config tau2_train/train_debug.yaml \
    experiment_name=xss-tau2-1.5B-train-v2 \
    trial_name=trial-2 \
    cluster.fileroot=/storage/openpsi/experiments \
    cluster.name_resolve.type=nfs \
    cluster.name_resolve.nfs_record_root=/storage/openpsi/experiments/name_resolve \
    launcher.slurm.trainer_image=/storage/openpsi/images/areal-25.02-v0.3.0.post2-v4.sif \
    launcher.slurm.inference_server_image=/storage/openpsi/images/areal-25.02-v0.3.0.post2-v4.sif  \
    allocation_mode=sglang.d32p1t1+d32p1t1 \
    cluster.n_nodes=8 \
    cluster.n_gpus_per_node=8 \
    train_dataset.path=training_data.jsonl \
    actor.path=/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct/
