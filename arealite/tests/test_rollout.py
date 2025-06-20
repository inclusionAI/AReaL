import json
import uuid

from omegaconf import OmegaConf

from arealite.api.cli_args import (
    GenerationHyperparameters,
    LLMClientConfig,
    TrainingArgs,
)
from arealite.api.io_struct import Trajectory
from arealite.api.rollout_api import RolloutWorkflowFactory
from arealite.impl.sglang_client import SGLangClient
from realhf.base import name_resolve

jsonl_content = [
{"prompt": "<\uff5cUser\uff5c>\nBaron Munchausen told a story. \"There were a whole crowd of us. We reached a crossroads. Then half of our group turned left, a third turned right, and a fifth went straight.\" \"But wait, the Duke remarked, the sum of half, a third, and a fifth isn't equal to one, so you are lying!\" The Baron replied, \"I'm not lying, I'm rounding. For example, there are 17 people. I say that a third turned. Should one person split in your opinion? No, with rounding, six people turned. From whole numbers, the closest to the fraction $17 / 3$ is 6. And if I say that half of the 17 people turned, it means 8 or 9 people.\" It is known that Baron Munchausen never lies. What is the largest number of people that could have been in the crowd?\nPlease reason step by step, and put your final answer within \\boxed{}.<\uff5cAssistant\uff5c><think>", "task": "math", "query_id": "00006d8f079c739f", "solutions": ["\\boxed{37}"]},
{"prompt": "<\uff5cUser\uff5c>What is the unit digit of the product\n\n$$\n(5+1)\\left(5^{3}+1\\right)\\left(5^{6}+1\\right)\\left(5^{12}+1\\right) ?\n$$\n\n(a) 0  \n(b) 1  \n(c) 2  \n(d) 5  \n(e) 6\nPlease reason step by step, and put your final answer within \\boxed{}.<\uff5cAssistant\uff5c><think>", "task": "math", "query_id": "000316109ea516b3", "solutions": ["\\boxed{e}"]},
{"prompt": "<\uff5cUser\uff5c>Given points \\( A(4,0) \\) and \\( B(2,2) \\) are inside the ellipse \\( \\frac{x^{2}}{25}+\\frac{y^{2}}{9}=1 \\), and \\( M \\) is a point on the ellipse, find the maximum value of \\( |MA| + |MB| \\).\nPlease reason step by step, and put your final answer within \\boxed{}.<\uff5cAssistant\uff5c><think>", "task": "math", "query_id": "000adcfa66ee4270", "solutions": ["\\boxed{10+2\\sqrt{10}}"]},
{"prompt": "<\uff5cUser\uff5c>There is a schoolbag containing 12 cards labeled $1, 1, 2, 2, \\cdots, 6, 6$. A person draws one card at a time without replacement. If a card is drawn that has the same number as a previously drawn card, both cards are discarded. The process ends when the person has 3 single cards in hand or all cards in the schoolbag have been drawn. Find the probability that all cards in the schoolbag are drawn.\nPlease reason step by step, and put your final answer within \\boxed{}.<\uff5cAssistant\uff5c><think>", "task": "math", "query_id": "001354647264e663", "solutions": ["\\boxed{\\frac{9}{385}}"]},
{"prompt": "<\uff5cUser\uff5c>For the sequence of numbers \\( n_{1}, n_{2}, n_{3}, \\ldots \\), the relation \\( n_{i} = 2 n_{i-1} + a \\) holds for all \\( i > 1 \\). If \\( n_{2} = 5 \\) and \\( n_{8} = 257 \\), what is \\( n_{5} \\)?\nPlease reason step by step, and put your final answer within \\boxed{}.<\uff5cAssistant\uff5c><think>", "task": "math", "query_id": "0014142e5f3c28a7", "solutions": ["\\boxed{33}"]},
]  # fmt: skip

jsonl_path = f"/tmp/{uuid.uuid4()}.jsonl"

with open(jsonl_path, "w") as f:
    for l in jsonl_content:
        f.write(json.dumps(l) + "\n")

args = TrainingArgs(experiment_name="test_rollout", trial_name="test_rollout")
name_resolve.reconfigure(args.cluster.name_resolve)
args.rollout.llm_client = LLMClientConfig(
    server_backend="sglang",
    tokenizer_path="Qwen/Qwen2-0.5B",
    request_timeout=10,
)
args.rollout.gconfig = GenerationHyperparameters(max_new_tokens=16)

collector = RolloutWorkflowFactory(args).make_workflow(args.rollout.workflow)

# Test the rollout workflow with the provided JSONL data
# with open(jsonl_path, "r") as f:
#     for i, l in enumerate(f.readlines()):
#         data = json.loads(l)
#         print(data.keys())
#         res = collector.run_episode(
#             env_option=dict(qid=data["query_id"], prompt=data["prompt"])
#         )
#         assert isinstance(res, Trajectory)
#         assert isinstance(res.data, dict)
#         shape = res.data["input_ids"].shape
#         for k in ["prompt_mask", "logprobs", "versions"]:
#             assert res.data[k].shape == shape
#         assert res.stats["ret"] in [-5.0, 5.0], res.stats["ret"]
