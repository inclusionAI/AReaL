RUNNING_SCRIPT=tau2_train/tau2_sft_workflow.py
train_dataset_path=
valid_dataset_path=
config_file=tau2_train/tau2_sft_config.yaml

PYTHONPATH="modules/AReaL/" \
NCCL_DEBUG=INFO \
python3 -m areal.launcher.slurm $RUNNING_SCRIPT \
    --config $config_file \
    experiment_name= \
    trial_name= \
    cluster.fileroot= \
    cluster.name_resolve.type=nfs \
    cluster.name_resolve.nfs_record_root= \
    allocation_mode=megatron:d2p4t2c2e2 \
    train_dataset.batch_size=32 \
    cluster.n_nodes=8 \
    cluster.n_gpus_per_node=8 \
    train_dataset.path=$train_dataset_path \
    valid_dataset.path=$valid_dataset_path \
    model.path= \
    +is_reasoning_model=true \
    total_train_epochs=10 \
    max_context_length=32768 \
    