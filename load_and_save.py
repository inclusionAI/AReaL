from transformers import AutoModelForImageTextToText, AutoProcessor


src_path = "/storage/openpsi/models/NVILA-Lite-8B-hf-0626"


dst_path = "/storage/openpsi/models/NVILA-Lite-8B-hf-0626_resave"


model = AutoModelForImageTextToText.from_pretrained(src_path)
tokenizer = AutoProcessor.from_pretrained(src_path)


model.save_pretrained(dst_path)
tokenizer.save_pretrained(dst_path)

print(f"模型已保存到 {dst_path}")
