# Copyright 2025 Ant Group Inc.
import json
from typing import List, Tuple
from PIL import Image
from areal.utils import logging
from AVoyager.utils.voyage_client import make_voyage_client


logger = logging.getLogger("Voyage ToolBox")

def load_metadata(dataset_path):
    data=[json.loads(ff) for ff in open(dataset_path)]
    for d in data:
        if "idx" in d:
            d["idx"] = str(d["idx"])
        elif "qid" in d:
            d["idx"] = str(d["qid"])
        else:
            d["idx"] = str(d["id"])
    id2info = {d["idx"]: d for d in data}
    return id2info


class VoyageToolBox:
    def __init__(self, voyage_client_types: list):


        self.voyage_client_types = voyage_client_types    
        self.grounding_client = None
        
    def init_grounding_client(self, original_image: Image.Image, processor):
        if "image-grounding" in self.voyage_client_types:
            self.grounding_client = make_voyage_client("image-grounding", original_image=original_image, processor=processor)
        else:
            raise ValueError(f"Image grounding client not specified in voyage_client_types: {self.voyage_client_types}")            
    
    async def step(self, qid_actions: Tuple[str, List[str]], current_iteration: int) -> List[dict]:
        qid, actions = qid_actions

        results = []
        for action in actions:
            result = dict(image=None, input_ids=None, type=None)

            # tool calling: for instant tool like image grounding and calculation, we adopt sync calling.
            if "<grounding>" in action and "</grounding>" in action:
                query = action.split("<grounding>")[-1].split("</grounding>")[0].strip()
                req_meta = {
                    "query": query,
                    "current_iteration": current_iteration,
                }

                # send grounding query to server
                response =self.grounding_client.query(req_meta)

                if response["image"] is not None:
                    result={
                        "image": response["image"],
                        "input_ids": response["input_ids"],
                        "text": response["text"],
                        "type": "grounding",
                        "status": "success",
                        "error_info": response["error_info"],
                    }
                else:
                    result={
                        "image": None,
                        "input_ids": response["input_ids"],
                        "text": response["text"],
                        "type": "grounding",
                        "status": "error",
                        "error_info": response["error_info"],
                    }

            # # compute rewards
            # ground_truth = self.id2info[qid.split("@")[0]]["answer"]
            # if isinstance(ground_truth, list) or isinstance(ground_truth, tuple):
            #     ground_truth = [str(gt) for gt in ground_truth]
            # else:
            #     ground_truth = str(ground_truth)
            
            # if self.reward_type == "F1":
            #     extracted, score = compute_score_f1(action, ground_truth, method="strict")
            # elif self.reward_type == "EM":
            #     extracted, score = compute_score_em(action, ground_truth, method="strict")
            
            results.append(result)
        return results
