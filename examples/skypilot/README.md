# Running AReaL with SkyPilot

This README includes examples and guidelines to running AReaL experiments with SkyPilot.
Make sure you have SkyPilot properly installed following
[our installation guide](../../docs/tutorial/installation.md#optional-install-skypilot)
before running this example.

## Setup Shared Storage

AReaL requires a shared file system (such as NFS) for basic functionalities such as name
resolve and distributed checkpointing. The following guideline shows how to use SkyPilot
volumes to setup a high-performance shared storage.

1. **Define the volume.** Create a YAML file describing the volume you want SkyPilot to
   manage. The example below provisions a 100 GiB ReadWriteMany Persistent Volume Claim
   on a Kubernetes cluster. Adjust the storage class, namespace, or size to match your
   environment.

```yaml
# storage-volume.yaml
name: areal-shared-storage
type: k8s-pvc
infra: kubernetes            # or k8s/context if you manage contexts manually
size: 100Gi                  # requested capacity
labels:
    project: areal
config:
    storage_class_name: csi-mounted-fs-path-sc
    access_mode: ReadWriteMany
```

2. **Create the volume.** Apply the definition once; SkyPilot will reuse the volume for
   future launches.

```bash
sky volumes apply storage-volume.yaml
sky volumes ls -v  # optional: confirm status and mount info
```

3. **Mount the volume in your tasks.** Add a `volumes` section to your SkyPilot YAML so
   every node in the cluster sees the shared path.

```yaml
volumes:
  /storage: areal-shared-storage
```

Then in your AReaL yaml file, you could use the shared storage by configuring fileroot
and NFS record root used by NFS name resolve:

```yaml
cluster:
  # ...
  fileroot: /storage/experiments
  name_resolve:
    # If you use a ray cluster, you can use KV store implemented in Ray
    # by setting `type: ray`.
    type: nfs
    nfs_record_root: /tmp/areal/name_resolve
```

To remove the volume when you no longer need it, run
`sky volumes delete areal-shared-storage`. For more information, checkout
[SkyPilot Volume Documentation](https://docs.skypilot.co/en/latest/reference/volumes.html).

## Option 1: Running AReaL with Ray Launcher

The following example shows how to setup a ray cluster with SkyPilot and then use AReaL
to run GRPO with GSM8K dataset on 2 8xH100 nodes.

### Example SkyPilot Cluster Spec

First, prepare your SkyPilot yaml:

```yaml
resources:
  accelerators: H100:8
  image_id: docker:ghcr.io/inclusionai/areal-runtime:v0.3.3
  memory: 256+
  cpus: 32+

num_nodes: 2

workdir: .

volumes:
  # shared storage setup by SkyPilot Volume
  /storage: areal-shared-storage

setup: |
  pip3 install -e .

run: |
  # Get the Head node's IP and total number of nodes (environment variables injected by SkyPilot).
  head_ip=$(echo "$SKYPILOT_NODE_IPS" | head -n1)
  num_nodes=$(echo "$SKYPILOT_NODE_IPS" | wc -l)

  if [ "$SKYPILOT_NODE_RANK" = "0" ]; then
    echo "Starting Ray head node..."
    ray start --head --port=6379

    while [ $(ray nodes | grep NODE_ID | wc -l) -lt $num_nodes ]; do
      echo "Waiting for all nodes to join..."
      sleep 5
    done

    echo "Executing training script on head node..."
    python3 -m areal.launcher.ray examples/math/gsm8k_grpo.py \
            --config examples/skypilot/gsm8k_grpo_ray.yaml \
            experiment_name=<your experiment name> \
            trial_name=<your trial name> \
            trainer_env_vars="WANDB_API_KEY=$WANDB_API_KEY"
  else
    sleep 10
    echo "Starting Ray worker node..."
    ray start --address $head_ip:6379
    sleep 5
    fi

  echo "Node setup complete for rank $SKYPILOT_NODE_RANK."
```

### Launch the Ray Cluster and AReaL

Then you are ready to run AReaL with command line:

```bash
export WANDB_API_KEY=<your-wandb-api-key>
sky launch -c areal --secret WANDB_API_KEY examples/skypilot/ray_cluster.yaml
```

You should be able to see your AReaL running and producing training logs in your
terminal.

<!--- TODO: add logging screenshots --->

## Option 2: Running AReaL with SkyPilot Launcher

<!--- TODO: to be finished and tested --->

You could also run a multi-node AReaL experiment in a single command, without having to
launch the Ray cluster:

```bash
python3 -m areal.launcher.skypilot examples/math/gsm8k_grpo.py \
    --config examples/skypilot/gsm8k_grpo_skypilot.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name>
```
