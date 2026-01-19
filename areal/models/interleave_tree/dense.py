import torch
from typing import List, Union
import tqdm

def train(model, inputs: List[torch.LongTensor], weights: List[float]):
    total_loss = 0.0
    model.train()

    for i in tqdm.tqdm(range(len(inputs))):
        input_ids = inputs[i].unsqueeze(0).to(model.device)
        weight = weights[i]

        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss * weight

        loss.backward()

        total_loss += loss.item()
    
    return total_loss

        
