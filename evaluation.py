from dataclasses import dataclass, field
from typing import Dict, Optional, List
import transformers
import torch
import random
import numpy as np
from tqdm import tqdm
import os
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, gather_object
import logging
import tqdm
from datasets import load_dataset
from prompt import format_prompts
import jsonlines
import copy
# from dataset_processor.processor_registers import *

'''
This script is to evaluate the LLM's performance on test dataset
'''

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)
device = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/home/incoming/LLM/llama2/llama2-7b")
    vocab_size: int = field(default=0)
    peft_model_path: str = field(default="")
    mode: str = field(default="chat")


@dataclass
class DataArguments:
    data_path: str = field(
        default="/home/LAB/jiangcy/AdaDF/samples/gsm8k_test.jsonl", metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    valid_data_path: str = field(
        default=None, metadata={"help": "valid data path, name:split"}
    )
    dataset_name: str = field(
        default=None
    )

    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=800,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def evaluation_main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # output_dir = "modelL-filter_strategy_{}-time_{}".format(data_args.data_filter_mode, int(time.time()))
    # training_args.output_dir = os.path.join(training_args.output_dir, output_dir)
    # os.makedirs(training_args.output_dir, exist_ok=True)
    # ROLE_CONTENT = "You are a calculation assistant. You will be given an arithmetic question. Please think step by step and give the answer. After giving your thoughts, use 'The answer is:' followed by the answer."
    accelerator = Accelerator()
    device = accelerator.device

    logger.info('Loading causal model...')
    model = transformers.InstructBlipForConditionalGeneration.from_pretrained(
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/instructblip-vicuna-13b", 
        torch_dtype=torch.bfloat16, 
        device_map="balanced_low_0"
    )
    # tokenizerL = transformers.AutoTokenizer.from_pretrained(
    #     "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/Llama-3.2-1B" 
    #         if model_args.mode == "chat" 
    #         else "/mnt/{}/rubickjiang/proj_storage/huggingface_models/Llama-3.2-1B".format(os.environ["MACLAB_NAS_NAME"]),
    #     model_max_length=training_args.model_max_length,
    #     use_fast=False,
    #     padding_side="left")
    # tokenizerL.pad_token_id = tokenizerL.eos_token_id

    # terminators = [
    #     tokenizerL.eos_token_id,
    #     tokenizerL.convert_tokens_to_ids("<|eot_id|>")
    # ] if model_args.mode == "chat" else tokenizerL.eos_token_id

    # test_dataset = TEST_DATA[data_args.dataset_name]()
    # answers = test_dataset["ground"]
    # prompts = test_dataset["inputs"]
    processor = transformers.InstructBlipProcessor.from_pretrained("/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/instructblip-vicuna-13b", padding_side="left", use_fast=False)
    dataset = load_dataset("parquet", data_files="/mnt/maclabcv2/rubickjiang/public_dataset/bench_final.parquet")
    prompts = []
    for data in dataset["train"]:
        question = data["question"]
        steps = data["response"]["modified_process"]
        gt = data["response"]["error_steps"]
        input = format_prompts(question, steps)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input.split("<image>")[0]},
                    {"type": "image"},
                    {"type": "text", "text": input.split("<image>")[1]},
                    ],
            },
        ]
        # text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        # text = input
        prompts.append({
            "text": input.replace("<image>", ""),
            "image": data["image"].convert("RGB"),
            "gt": gt
        })
    print("copying......")
    groundtruth = [p["gt"] for p in prompts]

    def prepare_prompts(prompts, processor, model_args, batch_size=16):
        text_batches = []
        image_batches = []
        for i in range(0, len(prompts), batch_size):
            text_batch = [prompts[j]["text"] for j in range(i, min(i + batch_size, len(prompts)))]
            image_batch = [prompts[j]["image"] for j in range(i, min(i + batch_size, len(prompts)))]
            text_batches.append(text_batch)
            image_batches.append(image_batch)
        batches_tok = []
        for t_batch, i_batch in zip(text_batches, image_batches):
            batches_tok.append(
                processor(images=i_batch, text=t_batch, return_tensors='pt', padding="longest", max_length=512, truncation=True).to(device)
            )
        return batches_tok

    model.eval()
    # model.to(device)
    accelerator.wait_for_everyone()

    with accelerator.split_between_processes(prompts) as prompts:
        results = dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches = prepare_prompts(prompts, processor, model_args, batch_size=training_args.per_device_eval_batch_size)
        pbar = tqdm.tqdm(total=len(prompt_batches), disable=(not accelerator.is_local_main_process))

        for prompts_tokenized in prompt_batches:
            with torch.no_grad():
                outputs_tokenized = model.generate(**prompts_tokenized, max_new_tokens=512, do_sample=False)

            # remove prompt from gen. tokens
            outputs_tokenized = [tok_out[len(tok_in):]
                                 for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)]

            # count and decode gen. tokens
            num_tokens = sum([len(t) for t in outputs_tokenized])
            outputs = processor.batch_decode(outputs_tokenized, skip_special_tokens=True)

            # store in results{} to be gathered by accelerate
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens
            if accelerator.is_local_main_process:
                pbar.update(1)
            torch.cuda.empty_cache()
        results = [results]  # transform to list, otherwise gather_object() will not collect correctly
    results_gathered = gather_object(results)
    if accelerator.is_main_process:
        total_results = []
        for r in results_gathered:
            total_results += r["outputs"]
        total_results = [txt for txt in total_results]
        # dump results
        with jsonlines.open("/mnt/maclabcv2/rubickjiang/codes/vllm_inference/output_insBLIP-T5-xxl.jsonl", "w") as writer:
            for i in range(len(groundtruth)):
                writer.write({
                    "ground": groundtruth[i],
                    "pred": total_results[i]
                })


if __name__ == "__main__":
    # 注意seed，原设置是没有do_sample的
    seed = os.environ.get("SEED", 114514)
    seed = int(seed)
    print("================set global random seed to {}================".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    evaluation_main()


