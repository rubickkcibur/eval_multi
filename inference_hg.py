import transformers
import torch
from datasets import load_dataset
from prompt import format_prompts
import jsonlines
from tqdm import tqdm
from io import BytesIO
import base64
from qwen_vl_utils import process_vision_info

device = "cuda:0"
model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/Qwen2.5-VL-7B-Instruct", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to(device)

def PIL_image_to_base64(image, format="PNG"):
    buffered = BytesIO()  # 创建内存缓冲区
    image.save(buffered, format=format)  # 将图像保存到缓冲区
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")  # 转换为 Base64 字符串
    return img_str

processor = transformers.AutoProcessor.from_pretrained("/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/Qwen2.5-VL-7B-Instruct")
dataset = load_dataset("parquet", data_files="/mnt/maclabcv2/rubickjiang/public_dataset/bench_final.parquet")
pbar = tqdm(total=len(dataset["train"]))
# item = dataset["train"][128]
# question = item["question"]
# steps = item["response"]["modified_process"]
# gt = item["response"]["error_steps"]
# input = format_prompts(question, steps)
# print(input)

with jsonlines.open("/mnt/maclabcv2/rubickjiang/codes/vllm_inference/output_qwen2.5-VL-instruct.jsonl", "w") as writer:
    for data in dataset["train"]:
        question = data["question"]
        steps = data["response"]["modified_process"]
        gt = data["response"]["error_steps"]
        input = format_prompts(question, steps)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input.split("<image>")[0]},
                    {"type": "image", "image": data["image"]},
                    {"type": "text", "text": input.split("<image>")[1]},
                ],
            }
        ]
        # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        # print(type(image_inputs))
        # print(type(video_inputs))
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # inputs = processor(images=data["image"].convert("RGB"), text=input, return_tensors='pt', max_length=512, truncation=True).to(device, torch.float16)
        # for key in inputs.keys():
        #     print("shape of {} is {}".format(key, inputs[key].shape))
        # output = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        # output = output[0]
        # output = output[len(inputs["input_ids"][0]):]
        # answer = processor.decode(output, skip_special_tokens=True)
        # answer = answer.split("ASSISTANT:")[-1]
        writer.write({
            "ground": data["response"]["error_steps"],
            "pred": output_text[0]
        })
        pbar.update(1)