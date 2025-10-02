import torch
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch.nn.functional as F

# 配置
MODEL_PATH = "/data/yl/model/Qwen/Qwen2.5-Math-PRM-7B"
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

# 定义 API
app = FastAPI()

class PRMRequest(BaseModel):
    text: str

@app.post("/score")
def score(req: PRMRequest):
    input_ids = tokenizer.encode(req.text, return_tensors="pt").to(DEVICE)
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
