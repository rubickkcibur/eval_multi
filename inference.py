from vllm import LLM, SamplingParams
import torch
from datasets import load_dataset
from PIL import Image
import ray
import jsonlines
from prompt import format_prompts
from transformers import AutoProcessor

ray.init()

@ray.remote(num_gpus=1)
class VLLMActor:
    def __init__(self, model_name_or_path):
        self.llm = LLM(
            model=model_name_or_path,
            max_model_len=2048,
            max_num_seqs=1024, 
            limit_mm_per_prompt={"image": 1}
        )
    def generate(self, prompts, sampling_params):
        return self.llm.generate(prompts, sampling_params)
    
def split_data(data, num_workers):
    chunk_size = len(data) // num_workers
    return [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]
    
num_workers = 8
sampling_params = SamplingParams(max_tokens=256, seed=114514)
dataset = load_dataset("parquet", data_files="/mnt/maclabcv2/rubickjiang/public_dataset/bench_final.parquet")
# processor = AutoProcessor.from_pretrained("/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/llava-1.5-7b-hf")
# for data in dataset["train"]:
#     question = data["question"]
#     steps = data["response"]["modified_process"]
#     gt = data["response"]["error_steps"]
#     input = format_prompts(question, steps)
#     conversation = [
#         {

#         "role": "user",
#         "content": [
#             {"type": "text", "text": input.split("<image>")[0]},
#             {"type": "image"},
#             {"type": "text", "text": input.split("<image>")[1]},
#             ],
#         },
#     ]
#     prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#     print(prompt)
#     quit()
#     inputs = processor(images=data["image"], text=prompt, return_tensors='pt').to(0, torch.float16)
#     print(inputs)
#     quit()
# prompt = "USER: <image>\nWhat does this image show?\nASSISTANT:"
inputs = [
    {
        "prompt": "USER: {}\nASSISTANT:".format(format_prompts(item["question"], item["response"]["modified_process"])),
        "multi_modal_data": {"image": item["image"]},
    }
    for item in dataset["train"]
]
data_chunks = split_data(inputs, num_workers)
actors = [VLLMActor.remote(model_name_or_path="/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/llava-1.5-7b-hf") for _ in range(num_workers)]
futures = [actor.generate.remote(chunk, sampling_params) 
           for actor, chunk in zip(actors, data_chunks)]
results = ray.get(futures)
final_results = []
for result in results:
    final_results.extend(result)
with jsonlines.open("/mnt/maclabcv2/rubickjiang/codes/vllm_inference/output_ray.jsonl", "w") as writer:
    for o in final_results:
        generated_text = o.outputs[0].text
        writer.write({
            "text": generated_text
        })


# if __name__ == "__main__":
#     dataset = load_dataset("parquet", data_files="/mnt/maclabcv2/rubickjiang/public_dataset/bench_final.parquet")
#     llm = LLM(
#         model="/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/llava-1.5-7b-hf",
#         max_model_len=2048,
#         max_num_seqs=1024, 
#         limit_mm_per_prompt={"image": 1})
    
#     sampling_params = SamplingParams(
#         max_tokens=256,
#         seed=114514)

#     prompt = "USER: <image>\nWhat does this image show?\nASSISTANT:"
#     inputs = [
#         {
#             "prompt": "USER: <image>\nWhat does this image show?\nASSISTANT:",
#             "multi_modal_data": {"image": item["image"]},
#         }
#         # {
#         #     "prompt": "USER: Repeat the following words: '{}'\nASSISTANT:".format(item["id"]),
#         # }
#         for item in dataset["train"]
#     ]

#     outputs = llm.generate(
#         inputs,
#         sampling_params=sampling_params,
#     )


#     with jsonlines.open("/mnt/maclabcv2/rubickjiang/codes/vllm_inference/output.jsonl", "w") as writer:
#         for o in outputs:
#             generated_text = o.outputs[0].text
#             writer.write({
#                 "text": generated_text
#             })
        