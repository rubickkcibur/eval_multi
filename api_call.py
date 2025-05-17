from vllm import LLM, SamplingParams
import torch
from datasets import load_dataset
from PIL import Image
import ray
import jsonlines
from prompt import format_prompts
import base64
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key="sk-Co7aVxrhEYKOomDszhPgOvksE1sLEaOLl4MVV5g1AnOaTE1z", base_url="https://api3.xhub.chat/v1/")
from io import BytesIO

def PIL_image_to_base64(image, format="PNG"):
    buffered = BytesIO()  # 创建内存缓冲区
    image.save(buffered, format=format)  # 将图像保存到缓冲区
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")  # 转换为 Base64 字符串
    return img_str

def call_api_with_image(prompt, image):
    base64_image = PIL_image_to_base64(image)
    response = client.chat.completions.create(model="gpt-4o",  # 指定模型
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.split("<image>")[0]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {"type": "text", "text": prompt.split("<image>")[1]}
                ]
            }
        ])

    # 返回结果
    return response.choices[0].message.content

dataset = load_dataset("parquet", data_files="/mnt/maclabcv2/rubickjiang/public_dataset/bench_final.parquet")
# inputs = [
#     {
#         "prompt": "USER: {}\nASSISTANT:".format(format_prompts(item["question"], item["response"]["modified_process"])),
#         "multi_modal_data": {"image": item["image"]},
#     }
#     for item in dataset["train"]
# ]
pbar = tqdm(total=len(dataset["train"]))
with jsonlines.open("output_api.jsonl", "w") as writer:
    for data in dataset["train"]:
        prompt = format_prompts(data["question"], data["response"]["modified_process"])
        image = data["image"]
        try:
            ret_content = call_api_with_image(prompt, image)
        except Exception as e:
            writer.write({
                "ground": data["response"]["error_steps"],
                "pred": "Wrong"
            })
        else:
            writer.write({
                "ground": data["response"]["error_steps"],
                "pred": ret_content
            })
        pbar.update(1)
