from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2VLForConditionalGeneration, AutoModel, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
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

# load datasets
dataset = load_dataset("FSCCS/ReasonMap", split="validation", cache_dir=cache_path)

print(f"Dataset: {dataset}")


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
elif model_path == "moonshotai/Kimi-VL-A3B-Thinking-onlytext" or model_path == "moonshotai/Kimi-VL-A3B-Instruct-onlytext":
    model_path = model_path.replace("-onlytext", "")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_path,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)    
    model_path = model_path+"-onlytext"
elif model_path == "Qwen/QVQ-72B-Preview":
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_path,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
elif model_path == "Skywork/Skywork-R1V-38B" or model_path == "Skywork/Skywork-R1V2-38B":
    # device_map = split_model(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=cache_path,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
elif model_path == "Qwen/Qwen2.5-VL-3B-Instruct" or model_path == "Qwen/Qwen2.5-VL-32B-Instruct" or model_path == "Qwen/Qwen2.5-VL-72B-Instruct":
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_path,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
elif model_path == "Qwen/Qwen2.5-VL-3B-Instruct-onlytext" or model_path == "Qwen/Qwen2.5-VL-32B-Instruct-onlytext" or model_path == "Qwen/Qwen2.5-VL-72B-Instruct-onlytext":
    model_path = model_path.replace("-onlytext", "")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_path,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)    
    model_path = model_path+"-onlytext"
elif model_path == "OpenGVLab/InternVL3-78B" or model_path == "OpenGVLab/InternVL3-38B":
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=cache_path).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
elif model_path == "OpenGVLab/InternVL3-78B-onlytext" or model_path == "OpenGVLab/InternVL3-38B-onlytext":
    model_path = model_path.replace("-onlytext", "")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=cache_path).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_path = model_path+"-onlytext"
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

    metadata = sample["json"]
    # save to ./station/{country}/{city}.json
    os.makedirs(f"./stations/{country}", exist_ok=True)
    with open(f"./stations/{country}/{city}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    # continue inference
    model_shortname = model_path.split("/")[-1].strip()
    out_dir = f"./results/{country}/{city}"
    filename = f"{i_sample}_{model_shortname}.json"
    filepath = os.path.join(out_dir, filename)

    if os.path.exists(filepath):
        continue

    question_short = sample["question_short"]
    question_long = sample["question_long"]

    station1 = sample["station_1"]
    station2 = sample["station_2"]

    print(f"Station1: {station1} to Station2: {station2}")

    # make figure get ready
    image = base64_to_image(sample["figure"])
    os.makedirs(f"./maps/{sample['country']}", exist_ok=True)
    image_path = f"./maps/{sample['country']}/{sample['city']}.png"
    if os.path.exists(image_path):
        pass
    else:
        image.save(image_path)
    
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
    # Short question evaluation

    if model_path == "moonshotai/Kimi-VL-A3B-Thinking" or model_path == "moonshotai/Kimi-VL-A3B-Instruct":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}
                ] + [{"type": "text", "text": question_short}],
            },
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        inputs = processor(images=[image], text=text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        short_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # token count for short response
        tokens = processor.tokenizer(short_response, return_tensors="pt").input_ids
        token_count_short = tokens.size(1)
        print("Token count for short question short response:", token_count_short)

        short_final_answer = short_response.split("◁/think▷")[-1].strip()

    elif model_path == "moonshotai/Kimi-VL-A3B-Thinking-onlytext" or model_path == "moonshotai/Kimi-VL-A3B-Instruct-onlytext":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question_short}
                ],
            },
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        inputs = processor(images=None, text=text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        short_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # token count for short response
        tokens = processor.tokenizer(short_response, return_tensors="pt").input_ids
        token_count_short = tokens.size(1)
        print("Token count for short question short response:", token_count_short)

        short_final_answer = short_response.split("◁/think▷")[-1].strip()

    elif model_path == "Qwen/QVQ-72B-Preview":
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question_short}
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt").to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        short_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # token count for short response
        tokens = processor.tokenizer(short_response, return_tensors="pt").input_ids
        token_count_short = tokens.size(1)
        print("Token count for short question short response:", token_count_short)

        boxed_content_match = re.search(r'boxed{(.+)}', short_response[0])
        if boxed_content_match:
            boxed_content = boxed_content_match.group(1)
            text_contents = re.findall(r'\\text{(.+?)}', boxed_content)
            short_final_answer = "\n".join(text_contents)
        else:
            # If no boxed content is found, use the entire response
            short_final_answer = short_response[0]

    elif model_path == "Skywork/Skywork-R1V-38B" or model_path == "Skywork/Skywork-R1V2-38B":
        pixel_values = [load_image(image_path, max_num=12).to(torch.bfloat16).cuda()]
        pixel_values = pixel_values[0]
        num_patches_list = None
            
        prompt = "<image>\n" + question_short
        generation_config = dict(max_new_tokens=2048, do_sample=True, temperature=0.6, top_p=0.95, repetition_penalty=1.05)
        short_response = model.chat(tokenizer, pixel_values, prompt, generation_config, num_patches_list=num_patches_list)

        # token count for short response
        tokens = tokenizer(short_response, return_tensors="pt").input_ids
        token_count_short = tokens.size(1)
        print("Token count for short question short response:", token_count_short)

        short_final_answer = short_response.split("</think>")[-1].strip()
        # clean cache
        del pixel_values
    elif model_path == "Qwen/Qwen2.5-VL-3B-Instruct" or model_path == "Qwen/Qwen2.5-VL-32B-Instruct" or model_path == "Qwen/Qwen2.5-VL-72B-Instruct":
        # TODO@sicheng: change to image_path
        base64_image = sample["figure"]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image;base64,{base64_image}",
                    },
                    {"type": "text", "text": question_short},
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
        short_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # token count for short response
        tokens = processor.tokenizer(short_response, return_tensors="pt").input_ids
        token_count_short = tokens.size(1)
        print("Token count for short question short response:", token_count_short)

        short_final_answer = short_response[0]
    elif model_path == "Qwen/Qwen2.5-VL-3B-Instruct-onlytext" or model_path == "Qwen/Qwen2.5-VL-32B-Instruct-onlytext" or model_path == "Qwen/Qwen2.5-VL-72B-Instruct-onlytext":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question_short},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = None, None
        inputs = processor(
            text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        short_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # token count for short response
        tokens = processor.tokenizer(short_response, return_tensors="pt").input_ids
        token_count_short = tokens.size(1)
        print("Token count for short question short response:", token_count_short)

        short_final_answer = short_response[0]
    elif model_path == "OpenGVLab/InternVL3-78B" or model_path == "OpenGVLab/InternVL3-38B":
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=2048, do_sample=True)

        question = f'<image>\n{question_short}'
        short_response = model.chat(tokenizer, pixel_values, question, generation_config)

        # token count for short response
        tokens = tokenizer(short_response, return_tensors="pt").input_ids
        token_count_short = tokens.size(1)
        print("Token count for short question short response:", token_count_short)

        short_final_answer = short_response
        # clean cache
        del pixel_values
    elif model_path == "OpenGVLab/InternVL3-78B-onlytext" or model_path == "OpenGVLab/InternVL3-38B-onlytext":
        pixel_values = None
        generation_config = dict(max_new_tokens=2048, do_sample=True)

        question = f'<image>\n{question_short}'
        short_response = model.chat(tokenizer, pixel_values, question, generation_config)

        # token count for short response
        tokens = tokenizer(short_response, return_tensors="pt").input_ids
        token_count_short = tokens.size(1)
        print("Token count for short question short response:", token_count_short)

        short_final_answer = short_response
        # clean cache
        del pixel_values
    print(f"Short Final Answer:\n")
    print(short_final_answer)
    print("\n")


    #############################################################################################
    # Evaluation pipeline

    print(f"Short Question Evaluation!\n")

    route_sections = short_final_answer.split("--")
    route_data = []

    for section in route_sections:
        print(f"section (short question): {section}\n")
        if section.strip():
            if "Route Name:" not in section or "Departure Stop:" not in section or "Arrival Stop:" not in section:
                print("Invalid section format. Skipping...")
                continue
            route_info = {}
            route_name_match = re.search(r"Route Name: (.*?)\n", section)
            departure_match = re.search(r"Departure Stop: (.*?)\n", section)

            route_info['Route Name'] = route_name_match.group(1).strip() if route_name_match else "Wrong"
            route_info['Departure Stop'] = clean_string(departure_match.group(1)) if departure_match else "Wrong"
            # route_info['Route Name'] = re.search(r"Route Name: (.*?)\n", section).group(1).strip()
            # route_info['Departure Stop'] = clean_string(re.search(r"Departure Stop: (.*?)\n", section).group(1))
            
            route_info['Arrival Stop'] = clean_string(section.split("Arrival Stop:")[-1].strip())
            
            route_data.append(route_info)

    # if can not get route data, pass this sample, we assume the answer is wrong (reason: wrong format...)
    if len(route_data) == 0:
        acc_short = 0
        best_map_api_score_short = 0
        save_departure_arrival_score_short = 0
        save_route_name_score_short = 0
        save_stops_score_short = 0
        print("No route data found.")
    else:
        # Metric@1: acc (only decided by the departure stop and arrival stop)
        acc_short = 0

        first_route = route_data[0] # first route section
        last_route = route_data[-1] # last route section
        
        # Check if the first route's Departure Stop matches station1 and the last route's Arrival Stop matches station2
        if clean_string(first_route['Departure Stop']) == clean_string(station1) and clean_string(last_route['Arrival Stop']) == clean_string(station2):
            # Verify each route and ensure transfer points match
            correct_route = True
            
            for i in range(len(route_data)):
                route = route_data[i]
                route_name = route['Route Name']
                if "八通" in route_name:
                    route_name = "Line 1"
                if "大兴" in route_name:
                    route_name = "Line 4"
                departure_stop = route['Departure Stop']
                arrival_stop = route['Arrival Stop']
                
                # Check if the route_name exists in metro_data
                if route_name in metro_data:
                    stations = [clean_string(station) for station in metro_data[route_name]]

                    # Check if departure_stop and arrival_stop are in the correct order
                    if clean_string(departure_stop) in stations and clean_string(arrival_stop) in stations:

                        # For transfer check, if it's not the last route, the Arrival Stop of the current route
                        # should match the Departure Stop of the next route
                        if i < len(route_data) - 1:
                            next_route = route_data[i + 1]
                            if clean_string(arrival_stop) != clean_string(next_route['Departure Stop']):
                                correct_route = False
                                print(f"Route {route_name}: Incorrect transfer from {arrival_stop} to {next_route['Departure Stop']}.")
                            else:
                                pass
                    else:
                        correct_route = False
                        print(f"Route {route_name}: One or both stops not found in route.")
                else:
                    correct_route = False
                    print(f"Route {route_name}: Route name not found in metro data.")
            
            if correct_route:
                acc_short = 1
                print("The route is correct.")
            else:
                print("The route is incorrect.")
        else:
            print(f"Wrong departure or arrival station. Expected {station1} and {station2}, but got {first_route['Departure Stop']} and {last_route['Arrival Stop']}.")

        correct_count_wo_via_stops_short += acc_short
        weight = 1 # TODO@sicheng, song: calculate the weight based on the difficulty of the city figure and question
        weighted_correct_count_wo_via_stops_short += acc_short * weight


        # Metric@2: map_api_score_short
        app_answer = sample['routes']
        best_map_api_score_short = 0
        
        # app may have multiple route plans, so we have to check one by one
        for key, route_plan in app_answer.items():
            map_api_score = 0 
            temp_departure_arrival_score_short = 0
            temp_route_name_score_short = 0
            temp_stops_score_short = 0

            # Check if the first route's Departure Stop matches station1 and the last route's Arrival Stop matches station2
            if clean_string(first_route['Departure Stop']) == clean_string(station1) and clean_string(last_route['Arrival Stop']) == clean_string(station2):
                map_api_score += 1
                temp_departure_arrival_score_short += 1

                for i, (route_data_section, app_route_section) in enumerate(zip(route_data, route_plan)):
                    # get answer info
                    route_name = route_data_section['Route Name']
                    departure_stop = route_data_section['Departure Stop']
                    arrival_stop = route_data_section['Arrival Stop']

                    if country == "china":
                        # get app answer info
                        # 地铁5号线(后关--虎滩新区) -> Line 5
                        # 特殊情况：地铁昌平线(昌平西山口--蓟门桥) -> 昌平线
                        # 特殊情况：地铁2号线(8号线) -> Line 2 (match1也能处理)
                        origin_app_route_name = app_route_section['route_name']
                        match1 = re.search(r'(\d+)', origin_app_route_name)

                        if match1:
                            line_number = int(match1.group(1))
                            result = str(line_number)
                            app_route_name = "Line "+result
                        else:
                            origin_app_route_name = re.sub(r'\(.*?\)', '', origin_app_route_name)
                            app_route_name = origin_app_route_name.replace("地铁", "")
                            print("No number in origin app route name, so we deal with tricks (e.g., 昌平线, 房山线). ")
                        
                        app_departure_stop = app_route_section['departure_stop']+"站" if app_route_section['departure_stop'].endswith("站") == False else app_route_section['departure_stop']
                        app_arrival_stop = app_route_section['arrival_stop']+"站" if app_route_section['departure_stop'].endswith("站") == False else app_route_section['departure_stop']
                    else:
                        # get app answer info

                        # add more regex to match the route name
                        # 1. los angeles: Metro A Line / Metro B-Line -> Line A / Line B
                        match1 = re.search(r"Metro\s+([A-Z0-9]{1,2})[-\s]?Line", route_name, re.IGNORECASE)
                        # 2. rome: A B1(short_name) / Metro B / Linea C -> Line A / Line B / Line C
                        match2 = re.search(r"Metro\s+([A-Z0-9]{1,2})", route_name, re.IGNORECASE)
                        match3 = re.search(r"Linea\s+([A-Z0-9]{1,2})", route_name, re.IGNORECASE)
                        match4 = re.search(r"([A-Z0-9]{1,2})", route_name, re.IGNORECASE)
                        # 3. auckland: West / Sth / East -> Western Line / Southern Line / Eastern Line
                        # 4. budapest: M1-es Metróvonal / M4-es metróvonal / M3-as metróvonal / M2-es metróvonal -> Line 1 / Line 4 / Line 3 / Line 2
                        match5 = re.search(r'M(\d+)', route_name, re.IGNORECASE)
                        # 5. dubai: MRed2 / MGrn / MRed1 -> Red Line / Green Line / Red Line
                        # 6. lisboa: Azul/Amarela/Verde/Vermelha -> Azul/Amarela/Verde/Vermelha (NOT changed)
                        # 7. singapore: North South Line / East West Line / Circle Line -> North South Line / East West Line / Circle Line (NOT changed)
                        # 8. oslo: 1 / 2 / 3 / 4 / 5 -> Line 1 / Line 2 / Line 3 / Line 4 / Line 5 (match4)
                        # 9. geneva: 12 / 18 -> Line 12 / Line 18 (match4)
                        # 10. kl: 
                            # MRT Putrajaya Line -> Line 12
                            # Ampang Line -> Line 3
                            # LRT Kelana Jaya Line -> Line 5
                            # KTM Tanjung Malim - Pelabuhan Klang -> Line 2
                            # LRT Sri Petaling Line -> Line 4
                            # MRT Kajang Line -> Line 9
                            # KTM Batu Caves - Pulau Sebang/Tampin -> Line 1
                        # 11. toronto: same as oslo
                        # 12. miami: Metrorail Orange Line / Metrorail Green Line -> Metrorail Orange Line / Metrorail Green Line (NOT changed)
                        # 13. washington: Red Line / Blue Line / Yellow Line / Orange Line / Silver Line -> Red Line / Blue Line / Yellow Line / Orange Line / Silver Line (NOT changed)
                        # 14. new_york: 1 Line / K Line / 5 Train / Q Train -> Line 1 / Line K / Line 5 / Line Q
                        match6 = re.search(r"([A-Z0-9]{1,2})\s+Line", route_name, re.IGNORECASE)
                        match7 = re.search(r"([A-Z0-9]{1,2})\s+Train", route_name, re.IGNORECASE)
                        # 15. sydney: M1 -> Line M1 (match4)

                        if match1:
                            result = match1.group(1)
                            app_route_name = "Line "+result
                        elif match2:
                            result = match2.group(1)
                            app_route_name = "Line "+result
                        elif match3:
                            result = match3.group(1)
                            app_route_name = "Line "+result
                        elif match4:
                            result = match4.group(1)
                            if result == "B1" and city == "rome":
                                result = "B"
                            app_route_name = "Line "+result
                        elif route_name == "West" and city == "auckland":
                            app_route_name = "Western Line"
                        elif route_name == "Sth" and city == "auckland":
                            app_route_name = "Southern Line"
                        elif route_name == "East" and city == "auckland":
                            app_route_name = "Eastern Line"
                        elif match5:
                            result = match5.group(1)
                            app_route_name = "Line "+result
                        elif route_name == "MRed2" and city == "dubai":
                            app_route_name = "Red Line"
                        elif route_name == "MGrn" and city == "dubai":
                            app_route_name = "Green Line"
                        elif route_name == "MRed1" and city == "dubai":
                            app_route_name = "Red Line"
                        elif route_name == "MRT Putrajaya Line" and city == "kl":
                            app_route_name = "Line 12"
                        elif route_name == "Ampang Line" and city == "kl":
                            app_route_name = "Line 3"
                        elif route_name == "LRT Kelana Jaya Line" and city == "kl":
                            app_route_name = "Line 5"
                        elif route_name == "KTM Tanjung Malim - Pelabuhan Klang" and city == "kl":
                            app_route_name = "Line 2"
                        elif route_name == "LRT Sri Petaling Line" and city == "kl":
                            app_route_name = "Line 4"
                        elif route_name == "MRT Kajang Line" and city == "kl":
                            app_route_name = "Line 9"
                        elif route_name == "KTM Batu Caves - Pulau Sebang/Tampin" and city == "kl":
                            app_route_name = "Line 1"
                        elif match6:
                            result = match6.group(1)
                            app_route_name = "Line "+result
                        elif match7:
                            result = match7.group(1)
                            app_route_name = "Line "+result
                        else:
                            # use original route name
                            app_route_name = route_name

                        
                        app_departure_stop = app_route_section['departure_stop']+" Station" if app_route_section['departure_stop'].endswith("Station") == False else app_route_section['departure_stop']
                        app_arrival_stop = app_route_section['arrival_stop']+" Station" if app_route_section['departure_stop'].endswith("Station") == False else app_route_section['departure_stop']


                    # Map app answer and final test answer
                    if clean_string(app_route_name) == clean_string(route_name):
                        map_api_score += 2
                        temp_route_name_score_short += 2
                    
                    if clean_string(app_departure_stop) == clean_string(departure_stop):
                        map_api_score += 1
                        temp_stops_score_short += 1

                    if clean_string(app_arrival_stop) == clean_string(arrival_stop):
                        map_api_score += 1
                        temp_stops_score_short += 1

            else:
                print(f"Wrong departure or arrival station. Expected {station1} and {station2}, but got {first_route['Departure Stop']} and {last_route['Arrival Stop']}.")

            map_api_score = min(10, map_api_score) # limit the score to 10

            # Make sure that when answer is correct, then it should get higher score than the wrong one
            if acc_short == 1:
                map_api_score += 10

            if map_api_score >= best_map_api_score_short:
                best_map_api_score_short = map_api_score
                save_departure_arrival_score_short = temp_departure_arrival_score_short
                save_route_name_score_short = temp_route_name_score_short
                save_stops_score_short = temp_stops_score_short

        print(f"Best map api score (short question): {best_map_api_score_short}")


    ###################################################################################################
    ###################################################################################################

    print("-" * 20)
    print(f"\nLong Question Evaluation!\n") 

    # long question evaluation
    if model_path == "moonshotai/Kimi-VL-A3B-Thinking" or model_path == "moonshotai/Kimi-VL-A3B-Instruct":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}
                ] + [{"type": "text", "text": question_long}],
            },
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        inputs = processor(images=[image], text=text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        long_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # token count for long response
        tokens = processor.tokenizer(long_response, return_tensors="pt").input_ids
        token_count_long = tokens.size(1)
        print("Token count for long question long response:", token_count_long)

        long_final_answer = long_response.split("◁/think▷")[-1].strip()

    elif model_path == "moonshotai/Kimi-VL-A3B-Thinking-onlytext" or model_path == "moonshotai/Kimi-VL-A3B-Instruct-onlytext":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question_long}
                ],
            },
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        inputs = processor(images=None, text=text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        long_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # token count for long response
        tokens = processor.tokenizer(long_response, return_tensors="pt").input_ids
        token_count_long = tokens.size(1)
        print("Token count for long question long response:", token_count_long)

        long_final_answer = long_response.split("◁/think▷")[-1].strip()


    elif model_path == "Qwen/QVQ-72B-Preview":
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question_long}
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt").to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        long_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # token count for long response
        tokens = processor.tokenizer(long_response, return_tensors="pt").input_ids
        token_count_long = tokens.size(1)
        print("Token count for long question response:", token_count_long)

        boxed_content_match = re.search(r'boxed{(.+)}', long_response[0])
        if boxed_content_match:
            boxed_content = boxed_content_match.group(1)
            text_contents = re.findall(r'\\text{(.+?)}', boxed_content)
            long_final_answer = "\n".join(text_contents)
        else:
            # If no boxed content is found, use the entire response
            long_final_answer = long_response[0]

    elif model_path == "Skywork/Skywork-R1V-38B" or model_path == "Skywork/Skywork-R1V2-38B":
        pixel_values = [load_image(image_path, max_num=12).to(torch.bfloat16).cuda()]
        pixel_values = pixel_values[0]
        num_patches_list = None
            
        prompt = "<image>\n" + question_long
        generation_config = dict(max_new_tokens=2048, do_sample=True, temperature=0.6, top_p=0.95, repetition_penalty=1.05)
        long_response = model.chat(tokenizer, pixel_values, prompt, generation_config, num_patches_list=num_patches_list)

        # token count for long response
        tokens = tokenizer(long_response, return_tensors="pt").input_ids
        token_count_long = tokens.size(1)
        print("Token count for long question response:", token_count_long)

        long_final_answer = long_response.split("</think>")[-1].strip()
    elif model_path == "Qwen/Qwen2.5-VL-3B-Instruct" or model_path == "Qwen/Qwen2.5-VL-32B-Instruct" or model_path == "Qwen/Qwen2.5-VL-72B-Instruct":
        # TODO@sicheng: change to image_path
        base64_image = sample["figure"]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image;base64,{base64_image}",
                    },
                    {"type": "text", "text": question_long},
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
        long_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # token count for long response
        tokens = processor.tokenizer(long_response, return_tensors="pt").input_ids
        token_count_long = tokens.size(1)
        print("Token count for long question response:", token_count_long)

        long_final_answer = long_response[0]
    elif model_path == "Qwen/Qwen2.5-VL-3B-Instruct-onlytext" or model_path == "Qwen/Qwen2.5-VL-32B-Instruct-onlytext" or model_path == "Qwen/Qwen2.5-VL-72B-Instruct-onlytext":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question_long},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = None, None
        inputs = processor(
            text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        long_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # token count for long response
        tokens = processor.tokenizer(long_response, return_tensors="pt").input_ids
        token_count_long = tokens.size(1)
        print("Token count for long question response:", token_count_long)

        long_final_answer = long_response[0]
    elif model_path == "OpenGVLab/InternVL3-78B" or model_path == "OpenGVLab/InternVL3-38B":
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=2048, do_sample=True)

        question = f'<image>\n{question_long}'
        long_response = model.chat(tokenizer, pixel_values, question, generation_config)

        # token count for long response
        tokens = tokenizer(long_response, return_tensors="pt").input_ids
        token_count_long = tokens.size(1)
        print("Token count for long question response:", token_count_long)

        long_final_answer = long_response
    elif model_path == "OpenGVLab/InternVL3-78B-onlytext" or model_path == "OpenGVLab/InternVL3-38B-onlytext":
        pixel_values = None
        generation_config = dict(max_new_tokens=2048, do_sample=True)

        question = f'<image>\n{question_long}'
        long_response = model.chat(tokenizer, pixel_values, question, generation_config)

        # token count for long response
        tokens = tokenizer(long_response, return_tensors="pt").input_ids
        token_count_long = tokens.size(1)
        print("Token count for long question response:", token_count_long)

        long_final_answer = long_response

    print(f"\nLong Final Answer:\n")
    print(long_final_answer)
    print("\n")

    route_sections = long_final_answer.split("--")
    route_data = []

    for section in route_sections:
        print(f"section (long question): {section}\n")
        if section.strip():
            if "Route Name:" not in section or "Departure Stop:" not in section or "Arrival Stop:" not in section:
                print("Invalid section format. Skipping...")
                continue
            route_info = {}
            route_name_match = re.search(r"Route Name: (.*?)\n", section)
            departure_match = re.search(r"Departure Stop: (.*?)\n", section)
            arrival_match = re.search(r"Arrival Stop: (.*?)\n", section)

            route_info['Route Name'] = route_name_match.group(1).strip() if route_name_match else "Wrong"
            route_info['Departure Stop'] = clean_string(departure_match.group(1)) if departure_match else "Wrong"
            route_info['Arrival Stop'] = clean_string(arrival_match.group(1)) if arrival_match else "Wrong"

            # route_info['Route Name'] = re.search(r"Route Name: (.*?)\n", section).group(1).strip()
            # route_info['Departure Stop'] = clean_string(re.search(r"Departure Stop: (.*?)\n", section).group(1))
            # route_info['Arrival Stop'] = clean_string(re.search(r"Arrival Stop: (.*?)\n", section).group(1))

            # two kinds of question format: (1) "Via Stops: " or (2) "Number of Via Stops: "
            if "Number of Via Stops:" in question_long:
                route_info['Number of Via Stops'] = section.split("Number of Via Stops: ")[-1].strip()
            else:
                route_info['Via Stops'] = section.split("Via Stops: ")[-1].strip()
            
            route_data.append(route_info)

    # if can not get route data, pass this sample, we assume the answer is wrong
    if len(route_data) == 0:
        acc_long = 0
        best_map_api_score_long = 0
        save_departure_arrival_score_long = 0
        save_route_name_score_long = 0
        save_stops_score_long = 0
        save_num_via_stop_score = 0
        save_via_stops_score = 0
        print("No route data found.")
    else:
        # Metric@1: acc (only decided by the departure stop and arrival stop)
        acc_long = 0

        first_route = route_data[0] # first route section
        last_route = route_data[-1] # last route section
        
        # Check if the first route's Departure Stop matches station1 and the last route's Arrival Stop matches station2
        if clean_string(first_route['Departure Stop']) == clean_string(station1) and clean_string(last_route['Arrival Stop']) == clean_string(station2):
            # Verify each route and ensure transfer points match
            correct_route = True
            
            for i in range(len(route_data)):
                route = route_data[i]
                route_name = route['Route Name']
                if "八通" in route_name:
                    route_name = "Line 1"
                if "大兴" in route_name:
                    route_name = "Line 4"
                departure_stop = route['Departure Stop']
                arrival_stop = route['Arrival Stop']
                
                # Check if the route_name exists in metro_data
                if route_name in metro_data:
                    stations = [clean_string(station) for station in metro_data[route_name]]

                    # Check if departure_stop and arrival_stop are in the correct order
                    if clean_string(departure_stop) in stations and clean_string(arrival_stop) in stations:

                        # For transfer check, if it's not the last route, the Arrival Stop of the current route
                        # should match the Departure Stop of the next route
                        if i < len(route_data) - 1:
                            next_route = route_data[i + 1]
                            if clean_string(arrival_stop) != clean_string(next_route['Departure Stop']):
                                correct_route = False
                                print(f"Route {route_name}: Incorrect transfer from {arrival_stop} to {next_route['Departure Stop']}.")
                            else:
                                pass
                    else:
                        correct_route = False
                        print(f"Route {route_name}: One or both stops not found in route.")
                else:
                    correct_route = False
                    print(f"Route {route_name}: Route name not found in metro data.")
            
            if correct_route:
                acc_long = 1
                print("The route is correct.")
            else:
                print("The route is incorrect.")
        else:
            print(f"Wrong departure or arrival station. Expected {station1} and {station2}, but got {first_route['Departure Stop']} and {last_route['Arrival Stop']}.")

        correct_count_wo_via_stops_long += acc_long
        weight = 1 # TODO: calculate the weight based on the difficulty of the city figure and question
        weighted_correct_count_wo_via_stops_long += acc_long * weight


        # Metric@2: map_api_score_long
        app_answer = sample['routes']
        best_map_api_score_long = 0
        
        # app may have multiple route plans, so we have to check one by one
        for key, route_plan in app_answer.items():
            map_api_score = 0 
            temp_departure_arrival_score_long = 0
            temp_route_name_score_long = 0
            temp_stops_score_long = 0

            via_stop_union = set()
            via_stop_intersection = set()

            via_stop_map_score = 0
            num_via_stop_score = 0
            via_stops_score = 0

            # Check if the first route's Departure Stop matches station1 and the last route's Arrival Stop matches station2
            if clean_string(first_route['Departure Stop']) == clean_string(station1) and clean_string(last_route['Arrival Stop']) == clean_string(station2):
                map_api_score += 1
                temp_departure_arrival_score_long += 1
                

                for i, (route_data_section, app_route_section) in enumerate(zip(route_data, route_plan)):
                    # get answer info
                    route_name = route_data_section['Route Name']
                    departure_stop = route_data_section['Departure Stop']
                    arrival_stop = route_data_section['Arrival Stop']
                    if "Number of Via Stops" in route_data_section:
                        num_via_stops = route_data_section['Number of Via Stops']
                        match = re.search(r'\d+', num_via_stops)
                        if match:
                            num_via_stops = match.group()
                        else:
                            num_via_stops = 0
                    else:
                        via_stops = route_data_section['Via Stops'].split(",") if "站" in route_data_section['Via Stops'] else []

                    if country == "china":
                        # get app answer info
                        # 地铁5号线(后关--虎滩新区) -> Line 5
                        # 特殊情况：地铁昌平线(昌平西山口--蓟门桥) -> 昌平线
                        # 特殊情况：地铁2号线(8号线) -> Line 2 (match1也能处理)
                        origin_app_route_name = app_route_section['route_name']
                        match1 = re.search(r'(\d+)', origin_app_route_name)

                        if match1:
                            line_number = int(match1.group(1))
                            result = str(line_number)
                            app_route_name = "Line "+result
                        else:
                            origin_app_route_name = re.sub(r'\(.*?\)', '', origin_app_route_name)
                            app_route_name = origin_app_route_name.replace("地铁", "")
                            print("No number in origin app route name, so we deal with tricks (e.g., 昌平线, 房山线). ")
                        
                        app_departure_stop = app_route_section['departure_stop']+"站" if app_route_section['departure_stop'].endswith("站") == False else app_route_section['departure_stop']
                        app_arrival_stop = app_route_section['arrival_stop']+"站" if app_route_section['arrival_stop'].endswith("站") == False else app_route_section['arrival_stop']
                        
                        # only get this for china cities
                        app_via_stops = []
                        for via_stop in app_route_section['via_stops']:
                            if via_stop.endswith("站") == False:
                                app_via_stops.append(via_stop+"站")
                            else:
                                app_via_stops.append(via_stop)

                        app_num_via_stops = app_route_section['num_via_stops']
                    else:
                        # get app answer info

                        # add more regex to match the route name
                        # 1. los angeles: Metro A Line / Metro B-Line -> Line A / Line B
                        match1 = re.search(r"Metro\s+([A-Z0-9]{1,2})[-\s]?Line", route_name, re.IGNORECASE)
                        # 2. rome: A B1(short_name) / Metro B / Linea C -> Line A / Line B / Line C
                        match2 = re.search(r"Metro\s+([A-Z0-9]{1,2})", route_name, re.IGNORECASE)
                        match3 = re.search(r"Linea\s+([A-Z0-9]{1,2})", route_name, re.IGNORECASE)
                        match4 = re.search(r"([A-Z0-9]{1,2})", route_name, re.IGNORECASE)
                        # 3. auckland: West / Sth / East -> Western Line / Southern Line / Eastern Line
                        # 4. budapest: M1-es Metróvonal / M4-es metróvonal / M3-as metróvonal / M2-es metróvonal -> Line 1 / Line 4 / Line 3 / Line 2
                        match5 = re.search(r'M(\d+)', route_name, re.IGNORECASE)
                        # 5. dubai: MRed2 / MGrn / MRed1 -> Red Line / Green Line / Red Line
                        # 6. lisboa: Azul/Amarela/Verde/Vermelha -> Azul/Amarela/Verde/Vermelha (NOT changed)
                        # 7. singapore: North South Line / East West Line / Circle Line -> North South Line / East West Line / Circle Line (NOT changed)
                        # 8. oslo: 1 / 2 / 3 / 4 / 5 -> Line 1 / Line 2 / Line 3 / Line 4 / Line 5 (match4)
                        # 9. geneva: 12 / 18 -> Line 12 / Line 18 (match4)
                        # 10. kl: 
                            # MRT Putrajaya Line -> Line 12
                            # Ampang Line -> Line 3
                            # LRT Kelana Jaya Line -> Line 5
                            # KTM Tanjung Malim - Pelabuhan Klang -> Line 2
                            # LRT Sri Petaling Line -> Line 4
                            # MRT Kajang Line -> Line 9
                            # KTM Batu Caves - Pulau Sebang/Tampin -> Line 1
                        # 11. toronto: same as oslo
                        # 12. miami: Metrorail Orange Line / Metrorail Green Line -> Metrorail Orange Line / Metrorail Green Line (NOT changed)
                        # 13. washington: Red Line / Blue Line / Yellow Line / Orange Line / Silver Line -> Red Line / Blue Line / Yellow Line / Orange Line / Silver Line (NOT changed)
                        # 14. new_york: 1 Line / K Line / 5 Train / Q Train -> Line 1 / Line K / Line 5 / Line Q
                        match6 = re.search(r"([A-Z0-9]{1,2})\s+Line", route_name, re.IGNORECASE)
                        match7 = re.search(r"([A-Z0-9]{1,2})\s+Train", route_name, re.IGNORECASE)
                        # 15. sydney: M1 -> Line M1 (match4)

                        if match1:
                            result = match1.group(1)
                            app_route_name = "Line "+result
                        elif match2:
                            result = match2.group(1)
                            app_route_name = "Line "+result
                        elif match3:
                            result = match3.group(1)
                            app_route_name = "Line "+result
                        elif match4:
                            result = match4.group(1)
                            if result == "B1" and city == "rome":
                                result = "B"
                            app_route_name = "Line "+result
                        elif route_name == "West" and city == "auckland":
                            app_route_name = "Western Line"
                        elif route_name == "Sth" and city == "auckland":
                            app_route_name = "Southern Line"
                        elif route_name == "East" and city == "auckland":
                            app_route_name = "Eastern Line"
                        elif match5:
                            result = match5.group(1)
                            app_route_name = "Line "+result
                        elif route_name == "MRed2" and city == "dubai":
                            app_route_name = "Red Line"
                        elif route_name == "MGrn" and city == "dubai":
                            app_route_name = "Green Line"
                        elif route_name == "MRed1" and city == "dubai":
                            app_route_name = "Red Line"
                        elif route_name == "MRT Putrajaya Line" and city == "kl":
                            app_route_name = "Line 12"
                        elif route_name == "Ampang Line" and city == "kl":
                            app_route_name = "Line 3"
                        elif route_name == "LRT Kelana Jaya Line" and city == "kl":
                            app_route_name = "Line 5"
                        elif route_name == "KTM Tanjung Malim - Pelabuhan Klang" and city == "kl":
                            app_route_name = "Line 2"
                        elif route_name == "LRT Sri Petaling Line" and city == "kl":
                            app_route_name = "Line 4"
                        elif route_name == "MRT Kajang Line" and city == "kl":
                            app_route_name = "Line 9"
                        elif route_name == "KTM Batu Caves - Pulau Sebang/Tampin" and city == "kl":
                            app_route_name = "Line 1"
                        elif match6:
                            result = match6.group(1)
                            app_route_name = "Line "+result
                        elif match7:
                            result = match7.group(1)
                            app_route_name = "Line "+result
                        else:
                            # use original route name
                            app_route_name = route_name


                        app_departure_stop = app_route_section['departure_stop']+" Station" if app_route_section['departure_stop'].endswith("Station") == False else app_route_section['departure_stop']
                        app_arrival_stop = app_route_section['arrival_stop']+" Station" if app_route_section['departure_stop'].endswith("Station") == False else app_route_section['departure_stop']
                        app_num_via_stops = app_route_section['num_via_stops']


                    # Map app answer and final test answer
                    if clean_string(app_route_name) == clean_string(route_name):
                        map_api_score += 2
                        temp_route_name_score_long += 2
                    
                    if clean_string(app_departure_stop) == clean_string(departure_stop):
                        map_api_score += 1
                        temp_stops_score_long += 1

                    if clean_string(app_arrival_stop) == clean_string(arrival_stop):
                        map_api_score += 1
                        temp_stops_score_long += 1

                    if "Number of Via Stops" in route_data_section:
                        error = abs(int(num_via_stops) - int(app_num_via_stops))
                        if error == 0:
                            num_via_stop_score += 4
                        else:
                            if int(app_num_via_stops) == 0:
                                num_via_stop_score += 0
                            else:
                                num_via_stop_score += max(0, 4 - (error / int(app_num_via_stops)) * 4)

                    else:
                        # calculate via stop score (only for china cities)
                        for via_stop in via_stops:
                            if via_stop in app_via_stops:
                                via_stop_map_score += 1
                            else:
                                pass

                        # calculate iou
                        if clean_string(route_name) == clean_string(app_route_name):
                            via_stop_union.update(app_via_stops)
                            via_stop_union.update(via_stops)
                            via_stop_intersection.update(set(app_via_stops).intersection(set(via_stops)))
                        else:
                            pass
                    

                via_stop_map_score = min(10, via_stop_map_score) # limit the score to 10
                num_via_stop_score = min(10, num_via_stop_score) # limit the score to 10

                iou = len(via_stop_intersection) / len(via_stop_union) if len(via_stop_union) > 0 else 0
                map_api_score += (iou * 10 + via_stop_map_score) / 2 + (num_via_stop_score) # two ways to calculate the score (two long question types)
                via_stops_score = (iou * 10 + via_stop_map_score) / 2

            else:
                print(f"Wrong departure or arrival station. Expected {station1} and {station2}, but got {first_route['Departure Stop']} and {last_route['Arrival Stop']}.")

            map_api_score = min(20, map_api_score) # limit the score to 20

            # Make sure that when answer is correct, then it should get higher score than the wrong one
            if acc_long == 1:
                map_api_score += 20

            if map_api_score >= best_map_api_score_long:
                best_map_api_score_long = map_api_score
                save_departure_arrival_score_long = temp_departure_arrival_score_long
                save_route_name_score_long = temp_route_name_score_long
                save_stops_score_long = temp_stops_score_long
                save_num_via_stop_score = num_via_stop_score
                save_via_stops_score = via_stops_score

        print(f"Best map api score (long question): {best_map_api_score_long}")




    # save the question, final answer and metrics to json
    model_shortname = model_path.split("/")[-1].strip()
    os.makedirs(f"./results/{sample['country']}/{sample['city']}", exist_ok=True)
    with open(f"./results/{sample['country']}/{sample['city']}/{i_sample}_{model_shortname}.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_path,
            "country": sample['country'],
            "city": sample['city'],
            "question_short": question_short, 
            "question_long": question_long, 
            "short full answer": short_response, 
            "short token count": token_count_short,
            "short final answer": short_final_answer, 
            "acc short": acc_short,
            "best map api score short": best_map_api_score_short,
            "save departure arrival score short": save_departure_arrival_score_short,
            "save route name score short": save_route_name_score_short,
            "save stops score short": save_stops_score_short,
            "long full answer": long_response, 
            "long token count": token_count_long,
            "long final answer": long_final_answer, 
            "acc long": acc_long,
            "best map api score long": best_map_api_score_long,
            "save departure arrival score long": save_departure_arrival_score_long,
            "save route name score long": save_route_name_score_long,
            "save stops score long": save_stops_score_long,
            "save num via stop score": save_num_via_stop_score,
            "save via stops score": save_via_stops_score,
            "difficulty city": sample['difficulty_city'],
            "difficulty question": sample['difficulty_question'],
            "city line count": sample['city_line_count'],
            "city transfer count": sample['city_transfer_count'],
            "question transfer count": sample['question_transfer_count']}, f, ensure_ascii=False, indent=4)

    # clean cache
    torch.cuda.empty_cache()

    print("-" * 20)

print("Done")
