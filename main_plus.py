from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2VLForConditionalGeneration, AutoModel, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoModelForVision2Seq
import datasets
from datasets import load_dataset, concatenate_datasets
from tools import base64_to_image, clean_string, split_model, load_image
import os
import json
import re
import argparse
import torch
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser(description="ReasonMap Evaluation")
parser.add_argument(
    "--model_path",
    type=str,
    default="moonshotai/Kimi-VL-A3B-Thinking",
    help="Path to the model directory or model name from Hugging Face Hub.",
)
parser.add_argument(
    "--cache_path",
    type=str,
    default="/mnt/data/fsc/cache",
    help="Path to the cache directory.",
)

args = parser.parse_args()

model_path = args.model_path
cache_path = args.cache_path


dataset = load_dataset("FSCCS/ReasonMap-Plus", split="test")

print(f"Filtered Dataset: {dataset}")


if model_path == "moonshotai/Kimi-VL-A3B-Thinking" or model_path == "moonshotai/Kimi-VL-A3B-Instruct":
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_path,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
elif model_path == "Qwen/Qwen2.5-VL-3B-Instruct" or model_path == "Qwen/Qwen2.5-VL-32B-Instruct" or model_path == "Qwen/Qwen2.5-VL-72B-Instruct" or model_path == "Qwen/Qwen2.5-VL-7B-Instruct":
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_path,
    ).eval()
else:
    raise ValueError(f"Unsupported model path: {model_path}")



correct_count_wo_via_stops_short = 0
weighted_correct_count_wo_via_stops_short = 0
correct_count_wo_via_stops_long = 0
weighted_correct_count_wo_via_stops_long = 0

correct_count_w_via_stops = 0


for i_sample, sample in enumerate(tqdm(dataset, desc="Evaluating samples")):
    country = sample["country"]
    city = sample["city"]

    # continue inference
    model_shortname = model_path.split("/")[-1].strip()
    out_dir = f"./results_plus/{country}/{city}"
    filename = f"{i_sample}_{model_shortname}.json"
    filepath = os.path.join(out_dir, filename)

    if os.path.exists(filepath):
        continue

    question = sample["question"]

    station1 = sample["station_1"]
    station2 = sample["station_2"]

    image_path = f"./maps/{sample['country']}/{sample['city']}.png"
    
    # load metro data in json
    with open(f"./stations/{country}/{city}.json", "r", encoding="utf-8") as f:
        metro_data = json.load(f)

    for route_name, stations in metro_data.items():
        metro_data[route_name] = [
            s.replace("（换乘站）", "")
            .replace("（支线起始站）", "")
            .replace(" (Transfer Station)", "")
            .replace(" (Branch-starting Station)", "")
            for s in stations
        ]

    ######################################################################################################

    if model_path == "moonshotai/Kimi-VL-A3B-Thinking" or model_path == "moonshotai/Kimi-VL-A3B-Instruct":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}
                ] + [{"type": "text", "text": question}],
            },
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        image = Image.open(sample["figure"])
        inputs = processor(images=[image], text=text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # token count for short response
        tokens = processor.tokenizer(response, return_tensors="pt").input_ids
        token_count = tokens.size(1)
        print("Token count:", token_count)

        final_answer = response.split("◁/think▷")[-1].strip()

    elif model_path == "Qwen/Qwen2.5-VL-3B-Instruct" or model_path == "Qwen/Qwen2.5-VL-32B-Instruct" or model_path == "Qwen/Qwen2.5-VL-72B-Instruct" or model_path == "Qwen/Qwen2.5-VL-7B-Instruct" or model_path == "ChenShawn/DeepEyes-7B":
        image_path = sample["figure"]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{image_path}",
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # token count for short response
        tokens = processor.tokenizer(response, return_tensors="pt").input_ids
        token_count = tokens.size(1)
        print("Token count:", token_count)

        final_answer = response[0]

    print(f"Final Answer:\n")
    print(final_answer)
    print("\n")

    #############################################################################################
    # Evaluation pipeline

    # get answer 
    model_answer = final_answer
    if 'oxed{' not in model_answer:
            for flag in ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']:
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split('\n')[0].split('. ')[0]
                flag = flag.replace('the', 'The')
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split('\n')[0].split('. ')[0]
    elif model_answer.count('oxed{') > 1:
        model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]

    from utils import find_math_answer
    model_answer = find_math_answer(model_answer).rstrip('.').lstrip(':').strip()

    # evaluate the answer
    ground_truth = int(sample["answer"])
    correct = None

    if "TorF" in sample["type"]:
        ground_truth = "yes" if ground_truth == 1 else "no"
        correct = model_answer.lower() == ground_truth
    elif sample["type"] == "Counting2" or sample["type"] == "Counting3":
        try:
            model_answer = int(model_answer)
            correct = int(model_answer) == ground_truth
        except (ValueError, TypeError):
            correct = False
    elif sample["type"] == "Counting1":
        model_answer = model_answer.lower()

        if model_answer in ['a', 'b', 'c', 'd']:
            mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
            model_answer = int(mapping[model_answer])
            correct = model_answer == ground_truth
        else:   
            correct = False

    # save the question, final answer and metrics to json
    model_shortname = model_path.split("/")[-1].strip()
    os.makedirs(f"./results_plus/{sample['country']}/{sample['city']}", exist_ok=True)
    with open(f"./results_plus/{sample['country']}/{sample['city']}/{i_sample}_{model_shortname}.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_path,
            "country": sample['country'],
            "city": sample['city'],
            "type": sample['type'],
            "question": question, 
            "full answer": response, 
            "token count": token_count,
            "final answer": final_answer, 
            "acc short": int(correct),
            "difficulty city": sample['difficulty_city'],
            "city line count": sample['city_line_count'],
            "city transfer count": sample['city_transfer_count']}, f, ensure_ascii=False, indent=4)

    # clean cache
    torch.cuda.empty_cache()

    # tool code for load previous answer, use for test
    # with open(f"./results_plus/{sample['country']}/{sample['city']}/{i}.json", "r", encoding="utf-8") as f:
    #     data = json.load(f)
    #     final_answer = data["final answer"]

    print("-" * 20)

print("Done")
