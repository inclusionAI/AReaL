# Agents

![](../../media/pipeline.png)

## Training

TextDAgger Training:
```bash
$ python scripts/train_dagger.py config/base_config.yaml
```

By default, you should be able to reproduce **BUTLERg** All Tasks results reported in [Table 2](https://arxiv.org/pdf/2010.03768.pdf). Run the [eval script](#evaluation) with `unstick_by_beam_search: True` to reproduce **BUTLER** evaluation results with a trained model.

VisionDAgger Training (unimodal baseline):
```bash
$ python scripts/train_vision_dagger.py config/base_config.yaml
```

TextDQN Training (not reported in the paper):
```bash
$ python scripts/train_dqn.py config/base_config.yaml
```

Seq2Seq Training (no DAgger baseline):
```bash

# collect dataset for offline training (not needed if python scripts/alfworld-download --extra)
# python scripts/collect_seq2seq_dataset.py config/base_config.yaml

# train seq2seq model
$ python scripts/train_seq2seq.py config/base_config.yaml
```

Run `python -m visdom.server` and set `visdom: True` in the [config file](../../configs/base_config.yaml) to plot training and evaluation curves.

## Config

Modify [base_config.yaml](../../configs/base_config.yaml) to your needs.

Dataset:
```yaml
dataset:
  data_path: '$ALFWORLD_DATA/json_2.1.1/train'
  eval_id_data_path: '$ALFWORLD_DATA/json_2.1.1/valid_seen'     # null/None to disable
  eval_ood_data_path: '$ALFWORLD_DATA/json_2.1.1/valid_unseen'  # null/None to disable
  num_train_games: -1                                           # max training games (<=0 indicates full dataset)
  num_eval_games: -1                                            # max evaluation games (<=0 indicates full dataset)
```

Environment:
```yaml
env:
  type: 'AlfredTWEnv'                                       # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
  # regen_game_files: False                                   # [Deprecated] Use script `alfworld-generate` instead.
  domain_randomization: False                               # shuffle Textworld print order and object id nums
  task_types: [1, 2, 3, 4, 5, 6]                            # task-type ids: 1 - Pick & Place, 2 - Examine in Light, 3 - Clean & Place, 4 - Heat & Place, 5 - Cool & Place, 6 - Pick Two & Place
  expert_timeout_steps: 150                                 # max steps before timeout for expert to solve the task
  expert_type: "handcoded"                                  # 'handcoded' or 'downward'. Note: the downward planner is very slow for real-time use
  goal_desc_human_anns_prob: 0.0                            # prob of using human-annotated goal language instead of templated goals (1.0 indicates all human annotations from ALFRED)

  hybrid:
    start_eps: 100000                                       # starting episode of hybrid training, tw-only training upto this point
    thor_prob: 0.5                                          # prob of AlfredThorEnv during hybrid training
    eval_mode: "tw"                                         # 'tw' or 'thor' - env used for evaluation during hybrid training

  thor:
    screen_width: 300                                       # width of THOR window
    screen_height: 300                                      # height of THOR window
    smooth_nav: False                                       # smooth rotations, looks, and translations during navigation (very slow)
    save_frames_to_disk: False                              # save frame PNGs to disk (useful for making videos of agent interactions)
    save_frames_path: '../videos/'                          # path to save frame PNGs
```

Controller:
```yaml
controller:
  type: 'oracle'                                            # 'oracle' or 'oracle_astar' or 'mrcnn' or 'mrcnn_astar' (aka BUTLER)
  debug: False
  load_receps: True                                         # load receptacle locations from precomputed dict (if available)
```
`mrcnn_astar` corresponds to **BUTLER**.

General:
```yaml
general:
  random_seed: 42
  use_cuda: False                                           # disable this when running on machine without cuda
  visdom: False                                             # plot training/eval curves, run with visdom server
  task: 'alfred'
  training_method: 'dagger'                                 # 'dqn' or 'dagger'
  save_path: '.'                                            # path to save pytorch models
  observation_pool_capacity: 3                              # k-size queue, 0 indicates no observation
  hide_init_receptacles: False                              # remove initial observation containing navigable receptacles

  training:
    batch_size: 10
    max_episode: 50000
    smoothing_eps: 0.1
    optimizer:
      learning_rate: 0.001
      clip_grad_norm: 5

  evaluate:
    run_eval: False
    batch_size: 10
    env:
      type: "AlfredTWEnv"

  checkpoint:
    report_frequency: 1000                                  # report every N episode
    experiment_tag: 'test'                                  # name of experiment
    load_pretrained: False                                  # during test, enable this so that the agent load your pretrained model
    load_from_tag: 'not loading anything'                   # name of pre-trained model to load in save_path

  model:
    encoder_layers: 1
    decoder_layers: 1
    encoder_conv_num: 5
    block_hidden_dim: 64
    n_heads: 1
    dropout: 0.1
    block_dropout: 0.1
    recurrent: True
```

General DAgger:
```yaml
dagger:
  action_space: "generation"                                # 'admissible' (candidates from text engine) or 'generation' (seq2seq-style generation) or 'exhaustive' (not working)
  max_target_length: 20                                     # max token length for seq2seq generation
  beam_width: 10                                            # 1 means greedy
  generate_top_k: 5
  unstick_by_beam_search: False                             # use beam-search for failed actions, set True during evaluation

  training:
    max_nb_steps_per_episode: 50                            # terminate after this many steps

  fraction_assist:
    fraction_assist_anneal_episodes: 50000
    fraction_assist_anneal_from: 1.0
    fraction_assist_anneal_to: 0.01

  fraction_random:
    fraction_random_anneal_episodes: 0
    fraction_random_anneal_from: 0.0
    fraction_random_anneal_to: 0.0

  replay:
    replay_memory_capacity: 500000
    update_per_k_game_steps: 5
    replay_batch_size: 64
    replay_sample_history_length: 4
    replay_sample_update_from: 2
```

Vision DAgger:
```yaml
vision_dagger:
  model_type: "resnet"                                      # 'resnet' (whole image features) or 'maskrcnn_whole' (whole image MaskRCNN feats) or 'maskrcnn' (top k MaskRCNN detection feats) or 'no_vision' (zero vision input)
  resnet_fc_dim: 64
  maskrcnn_top_k_boxes: 10                                  # top k box features
  use_exploration_frame_feats: False                        # append feats from initial exploration (memory intensive!)
  sequence_aggregation_method: "average"                    # 'sum' or 'average' or 'rnn'
```

General DQN:
```yaml
rl:
  action_space: "admissible"                                # 'admissible' (candidates from text engine) or 'generation' (seq2seq-style generation) or 'beam_search_choice' or 'exhaustive' (not working)
  max_target_length: 20                                     # max token length for seq2seq generation
  beam_width: 10                                            # 1 means greedy
  generate_top_k: 3

  training:
    max_nb_steps_per_episode: 50                            # terminate after this many steps
    learn_start_from_this_episode: 0                        # delay updates until this epsiode
    target_net_update_frequency: 500                        # sync target net with online net per this many epochs

  replay:
    accumulate_reward_from_final: True
    count_reward_lambda: 0.0                                # 0 to disable
    novel_object_reward_lambda: 0.0                         # 0 to disable
    discount_gamma_game_reward: 0.9
    discount_gamma_count_reward: 0.5
    discount_gamma_novel_object_reward: 0.5
    replay_memory_capacity: 500000                          # adjust this depending on your RAM size
    replay_memory_priority_fraction: 0.5
    update_per_k_game_steps: 5
    replay_batch_size: 64
    multi_step: 3
    replay_sample_history_length: 4
    replay_sample_update_from: 2

  epsilon_greedy:
    noisy_net: False                                        # if this is true, then epsilon greedy is disabled
    epsilon_anneal_episodes: 1000                           # -1 if not annealing
    epsilon_anneal_from: 0.3
    epsilon_anneal_to: 0.1
```

## Evaluation

The training script evaluates every `report_frequency` episodes. But additionally, you can also independently evaluate pre-trained agents:

```bash
$ python scripts/run_eval.py config/eval_config.yaml
```

Modify [eval_config.yaml](../../configs/eval_config.yaml) to your needs:
```yaml
general:
...
  evaluate:
    run_eval: True
    batch_size: 3                                           # number of parallel eval threads
    repeats: 1                                              # number of times to loop over eval games (we used 3 in paper experiments)
    controllers:                                            # different controllers to evaluate with
      - 'oracle'
      - 'mrcnn_astar'
    envs:                                                   # different environments to evaluate in
      - 'AlfredTWEnv'
      - 'AlfredThorEnv'
    eval_paths:                                             # different splits to evaluate on
      - '$ALFWORLD_DATA/json_2.1.1/valid_seen'
      - '$ALFWORLD_DATA/json_2.1.1/valid_unseen'
    eval_experiment_tag: "eval_run_001"                     # save results json with this prefix

  checkpoint:
    report_frequency: 10                                    # report eval results every N episode
    load_pretrained: True
    load_from_tag: 'pretrained agent'                       # name of pre-trained model to load in save_path

```

The script will sequentially evaluate all `envs`, `controllers`, and `eval_paths` in that nested order.


## Folder Structure

```
/configs
    base_config.yaml               (basic settings for all experiments)
    eval_config.yaml               (settings for batch evaluations)
/scripts
    train_dagger.py                (training script for TextDAgger agents)
    train_vision_dagger.py         (training script for VisionDAgger agents)
    train_dqn.py                   (training script for TextDQN agents)
    train_seq2seq.py               (training script for TextSeq2seq agents)
    run_eval.py                    (evaluation script for batch evals)
    collect_seq2seq_dataset.py     (data collection script for expert demonstrations)
    train_mrcnn.py                 (training script for MaskRCNN state-estimator)
/alfworld
    /agent
        base_agent.py              (base class for agents)
        text_dagger_agent.py       (TextDAgger agent used for BUTLER)
        text_dqn_agent.py          (TextDQN agent not reported in the paper)
        vision_dagger_agent.py     (VisionDAgger agent used for unimodal baselines)
    /environment
        alfred_tw_env.py           (ALFRED TextWorld environment)
        alfred_thor_env.py         (ALFRED embodied environment with THOR)
        alfred_hybrid.py           (hybrid training manager)
    /detector
        mrcnn.py                   (MaskRCNN state-estimator)
    /controller
        base.py                    (base class for controllers)
        oracle.py                  (GT object detections and teleport navigation)
        oracle_astar.py            (GT object detections and A* navigator)
        mrcnn.py                   (MaskRCNN object detections and teleport navigation)
        mrcnn_astar.py             (MaskRCNN object detections and A* navigator aka BUTLER)
    /eval
        evaluate_dagger.py         (evaluation loop for TextDAgger agents)
        evaluate_vision_dagger.py  (evaluation loop for VisionDAgger agents)
        evaluate_dqn.py            (evaluation loop for TextDQN agents)
```
