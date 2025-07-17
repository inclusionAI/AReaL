# Use a pipeline as a high-level helper
from transformers import AutoProcessor
processor=AutoProcessor.from_pretrained(pretrained_model_name_or_path="/storage/openpsi/models/Qwen2.5-VL-32B-Instruct")
input_ids = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 151652,  151655, 151653, 4340, 1657, 3589, 525, 1052, 304, 279, 2168, 30, 151645, 198, 151644, 77091, 198]

decoded_text = processor.tokenizer.decode(input_ids)
print(decoded_text)

# pipe = pipeline("image-text-to-text", model="/storage/openpsi/models/Qwen2.5-VL-32B-Instruct",device_map="auto" )
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": "output_image.jpg"},
#             {"type": "text", "text": "What is shown in the image??"}
#         ]
#     },
# ]
# print(pipe(text=messages))