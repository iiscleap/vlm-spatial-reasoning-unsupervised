from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
import csv
import os

def get_prompt_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['prompt']
    return None

## quantization and model config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
distributed_state = PartialState()

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-34b-hf", quantization_config=bnb_config, device_map= "auto") 

## creating dictionary to store the names of objects mentioned in the orig prompt (eg car,cat)
object_dict = {}
with open("../data/BLINK/object_list.txt","r") as obj_file:
    for line in obj_file:
        idx, objs = line.split(" ", maxsplit = 1)
        object_dict[idx] = objs 

num_list = list(range(1,144))

for i in num_list:
    ## setting paths and idx
    idx = f"val_Spatial_Relation_{i}"
    img_path = f"../data/BLINK/orig_images/val_Spatial_Relation_{i}.jpg"
    image = Image.open(img_path)
    conversation = []

    ## crafting the prompt
    objs = object_dict[idx]
    obj1,obj2 = objs.split("," , maxsplit = 1)
    q1 = f"Q1. Is there a {obj1} in the image?"
    q2 = f"Q2. Is there a {obj2} in the image?"
    q3 = f"Q3. Where is the {obj1} in the image?"
    q4 = f"Q4. Where is the {obj2} in the image?"
    q5 = f"Q5. Are the {obj1} and the {obj2} interacting with each other?"
    q6 = f"Q6. What is the spatial relationship between the {obj1} and the {obj2}?"
    all_q = f"You are a spatial reasoning bot that must answer all of the following questions:\n{q1}\n{q2}\n{q3}\n{q4}\n{q5}\n{q6}\nYou must provide coherent explanations and answers to all the questions above."
        
    ## inferencing - CoT
    conversation.append({ "role": "user", "content": [ {"type": "text", "text": f"{all_q}"}, {"type": "image"}]})
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=250)
    response = processor.decode(output[0], skip_special_tokens=True)
    answer = response.split("assistant")[-1]
    answer = answer.strip("\n")
    
    with open(f"output/llava/{idx}.txt", "a") as all_answers:
        all_answers.write(f"{answer}\n")
    
    conversation.append({ "role": "assistant", "content": [ {"type": "text", "text": f"{answer}"} ]})
        
    ## inferencing - final question
    final_q = get_prompt_for_idx("../data/BLINK/prompt.csv", idx)
    final_q = f"You are a spatial reasoning bot that must answer the following question:\n{final_q}\nYou must provide an explanation for your answer. Your answer must be either (A) or (B)."
    conversation.append({ "role": "user", "content": [ {"type": "text", "text": f"{final_q}"} ]})
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(prompt, image, return_tensors="pt").to(distributed_state.device)
    output = model.generate(**inputs, max_new_tokens=200)
    response = processor.decode(output[0], skip_special_tokens=True)
    answer = response.split("assistant")[-1]
    answer = answer.strip("\n")
    with open(f"output/llava/{idx}.txt", "a") as all_answers:
        all_answers.write(f"{answer}\n")