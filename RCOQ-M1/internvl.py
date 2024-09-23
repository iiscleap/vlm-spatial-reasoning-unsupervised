from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch
import csv
import sys

def get_prompt_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['prompt']
    return None

path = "OpenGVLab/InternVL-Chat-V1-2"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
image_processor = CLIPImageProcessor.from_pretrained(path)

object_dict = {}
with open("../data/BLINK/object_list.txt","r") as obj_file:
    for line in obj_file:
        idx, objs = line.split(" ", maxsplit = 1)
        object_dict[idx] = objs 

for i in range(1,144):
    idx =  f"val_Spatial_Relation_{i}"
    image = Image.open(f'f"../data/BLINK/orig_images/val_Spatial_Relation_{i}.jpg').resize((448, 448))
    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=256, do_sample=False)
    
    objs = object_dict[idx]
    obj1,obj2 = objs.split("," , maxsplit = 1)
    q1 = f"Q1. Is there a {obj1} in the image?"
    q2 = f"Q2. Is there a {obj2} in the image?"
    q3 = f"Q3. Where is the {obj1} in the image?"
    q4 = f"Q4. Where is the {obj2} in the image?"
    q5 = f"Q5. Are the {obj1} and the {obj2} interacting with each other?"
    q6 = f"Q6. What is the spatial relationship between the {obj1} and the {obj2}?"
    all_q = f"You are a spatial reasoning bot that must answer all of the following questions:\n{q1}\n{q2}\n{q3}\n{q4}\n{q5}\n{q6}\nYou must provide coherent explanations and answers to all the questions above."

    response, history = model.chat(tokenizer, pixel_values, all_q, generation_config, history=None, return_history=True)
    with open(f"output/internvl/{idx}.txt", "a") as baseline:
        baseline.write(f"{response}\n")
    
    final_q = get_prompt_for_idx("../data/BLINK/prompt.csv", idx)
    final_q = f"You are a spatial reasoning bot that must answer the following question:\n{final_q}\nYou must provide an explanation for your answer. Your answer must be either (A) or (B)."
    response, history = model.chat(tokenizer, pixel_values, final_q, generation_config, history=history, return_history=True)

    with open(f"output/internvl/{idx}.txt", "a") as baseline:
        baseline.write(f"{response}\n")