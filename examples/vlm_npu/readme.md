# Training VLMs with GRPO on NPU:
In this instruction, we will introduce how to train VLMs with GRPO on NPU. 

### Hardware

The following hardware configuration has been extensively tested:

- **NPU**: 16x NPU per node
- **CPU**: 64 cores per node
- **Memory**: 1TB per node
- **Network**: RoCE 3.2 Tbps
- **Storage**:
  - 1TB local storage for single-node experiments
  - 10TB shared storage (NAS) for distributed experiments

### Key Contributions
-	Trained Qwen2.5VL-3B-instruct model upto 70 epochs with (4 cards+ 4 cards) train-infer configuration. Took around 19hr to finish full training.
- Trained model is tested with more than one benchmark using VLMEvalKit.


### System configuration: 
-	Vllm==0.11.0 ; vllm-ascend==0.11.0rc0 ;
-	Torch==0.7.1+cpu ; torch_npu==2.7.1.dev20250724
-	Areal==0.4.1
-	CANN==8.1RC1 ; 910c npus (65 gigs X 16)

### Results:
We trained Qwen2.5-VL-3B for 70 epochs on Geometry3K and evaluated the checkpoints using VLMEvalKit on out of distribution tasks such as MathVision, MathVista, and LogicVista. The training was performed on both NPU and GPU and results are as follows:

| Method     | LogicVista | MathVision_mini | MathVista_mini | Avg.  |
|------------|------------|------------------|----------------|-------|
| Base Model | 31.0       | 18.3             | 52.3           | 33.8     |
| GRPO-GPU   | 35.4       | 20.9             | 55.9           | **37.4** |
| GRPO-NPU   | 35.3       | 20.5             | 54.7           | **36.8** |
