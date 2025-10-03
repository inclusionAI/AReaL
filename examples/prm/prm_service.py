import torch
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch.nn.functional as F
import sys, os
logfile = open("prm_server.log", "a", buffering=1)
sys.stdout = logfile
sys.stderr = logfile

# 配置
MODEL_PATH = "/data/yanglu/model/Qwen/Qwen2.5-Math-PRM-7B"
DEVICE = "cuda:3"  # 固定 PRM 用的卡

# 加载模型
print("Loading PRM model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True
).to(DEVICE).eval()
max_pos = model.config.max_position_embeddings
end_tokens = ["<extra_0>", "<|im_end|>", "<|endoftext|>"]
end_ids = [tokenizer.convert_tokens_to_ids(t) for t in end_tokens]
allowed_txt_len = max_pos - len(end_ids)

# 定义 API
app = FastAPI()

class PRMRequest(BaseModel):
    text: str

@app.post("/score")
def score(req: PRMRequest):
    # print(f"req.text: {req.text}")
    input_ids = tokenizer.encode(req.text, return_tensors="pt").to(DEVICE)
    if input_ids.shape[1] >= max_pos:
        input_ids = input_ids.cpu().squeeze(0).tolist()
        truncated_ids = input_ids[:allowed_txt_len]
        input_ids = torch.tensor([truncated_ids+end_ids], device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        step_sep_id = tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)
        probabilities = F.softmax(outputs[0], dim=-1)* token_masks.unsqueeze(-1)
        sample = probabilities[0]
        prm_reward = sample[sample != 0].view(-1, 2)[:, 1].cpu().tolist() # list
    return {"reward": prm_reward}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
