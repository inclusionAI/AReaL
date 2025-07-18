# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

# This file standardizes the name-resolve names used by different components of the system.
import getpass

USER_NAMESPACE = getpass.getuser()


def registry_root(user):
    return f"trial_registry/{user}"


def trial_registry(experiment_name, trial_name):
    return f"trial_registry/{USER_NAMESPACE}/{experiment_name}/{trial_name}"


def trial_root(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}"


def worker_status(experiment_name, trial_name, worker_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/status/{worker_name}"


def worker_root(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/worker/"


def worker(experiment_name, trial_name, worker_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/worker/{worker_name}"


def worker_key(experiment_name, trial_name, key):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/worker_key/{key}"


def request_reply_stream(experiment_name, trial_name, stream_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/request_reply_stream/{stream_name}"


def request_reply_stream_root(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/request_reply_stream/"


def distributed_root(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/distributed/"


def distributed_peer(experiment_name, trial_name, model_name):
    return (
        f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/distributed/peer/{model_name}"
    )


def distributed_local_peer(experiment_name, trial_name, host_name, model_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/distributed/local_peer/{host_name}/{model_name}"


def distributed_master(experiment_name, trial_name, model_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/distributed/master/{model_name}"


def model_version(experiment_name, trial_name, model_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/model_version/{model_name}"


def metric_server_root(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/metrics"


def metric_server(experiment_name, trial_name, group, name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/metrics/{group}/{name}"


def push_pull_stream(experiment_name, trial_name, stream_name):
    # Used to write addresses
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/push_pull_stream/{stream_name}"


def push_pull_stream_root(experiment_name, trial_name):
    # Used to collect addresses
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/push_pull_stream/"


def stream_pullers(experiment_name, trial_name):
    # Used to claim identities so that pushers know the number of pullers
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/push_pull_stream_peers/"


def gen_servers(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/gen_servers"


def used_ports(experiment_name, trial_name, host_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/{host_name}/"


def gen_server_manager(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/gen_server_manager"


def training_samples(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/training_samples"


def experiment_status(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/experiment_status"


def update_weights_from_disk(experiment_name, trial_name, model_version):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/update_weights_from_disk/{model_version}"
