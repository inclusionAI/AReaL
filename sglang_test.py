def sgl_gen_local(prompt: str,
                  model_path: str,
                  max_new_tokens: int = 128,
                  temperature: float = 0.0) -> str:

    from sglang import Engine
    

    eng = Engine(model_path=model_path)

    out = eng.generate(
        prompt=prompt,
        return_logprob=True,
        logprob_start_len=-1,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    return getattr(out, "text", out)

if __name__ == "__main__":
    model_path = "/path/to/your/local/model"   # 本地模型路径
    prompt = "请用一句话解释黑洞。"
    print(sgl_gen_local(prompt, model_path))
