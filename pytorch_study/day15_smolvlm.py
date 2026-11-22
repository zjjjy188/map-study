from PIL import Image

image_path = r"D:\project\map-study\MapReader\data\MapEval-DB\Country\China\Map\12_16.jpg"
question = "Does the G70 highway cross the river in this map? Answer briefly."

image = Image.open(image_path).convert("RGB")
print("Image size:", image.size)
print("Question:", question)

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = r"D:\AI-models\SmolVLM-500M-Instruct"

processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.float16,
    local_files_only=True,
    attn_implementation="sdpa",
).to("cuda").eval()

messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ],
}]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to("cuda", dtype=torch.float16)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
print("\nModel answer:\n", answer)