from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch
import csv
import os

def load_questions(file_path):
    questions = {}
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("Q1"):
                questions['Q1'] = line
            elif line.startswith("Q2"):
                questions['Q2'] = line
            elif line.startswith("Q3"):
                questions['Q3'] = line
            elif line.startswith("Q4"):
                questions['Q4'] = line
            elif line.startswith("Q5"):
                questions['Q5'] = line
            elif line.startswith("Q6"):
                questions['Q6'] = line
            elif line.startswith("Q7"):
                questions['Q7'] = line
            else:
                continue
    if 'Q7' not in questions and 'Q6' in questions:
        questions['Q7'] = questions['Q6']
    
    return questions


path = "OpenGVLab/InternVL-Chat-V1-2"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
image_processor = CLIPImageProcessor.from_pretrained(path)

output_dir = '../InternVL/InternVL40BLINK_CoT/'

for i in range(1,144):
    idx =  f"val_Spatial_Relation_{i}"
    image = Image.open(f'../data/BLINK/orig_images/{idx}.jpg').resize((448, 448))
    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=512, do_sample=False)
    
    questions_file = f"../data/BLINK/Ques_breakdown/GQ_options/{idx}.txt"
    questions = load_questions(questions_file)
    
    first_six_questions = "\n".join([questions[q] for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']])
    final_question = questions['Q7']  

    all_q = f"<image>\nYou are a spatial reasoning bot that must answer all of the following questions:\n{first_six_questions}\nYou must provide coherent explanations and answers to all the questions above. Format your answer in the following format: \nAns1: Verbose Answer \nAns2: Verbose Answer \nAns3: Verbose Answer \nAns4: Verbose Answer \nAns5: Verbose Answer \nAns6: Verbose Answer"
    response, history = model.chat(tokenizer, pixel_values, all_q, generation_config, history=None, return_history=True)
    print(f'Assistant: {response}')
    
    with open(f"{output_dir}/{idx}.txt", "a") as baseline:
        baseline.write(f"{response}")

    final_q = f"You are a spatial reasoning bot that must answer the following question:\n{final_question}\nYou must provide an explanation for your answer and finally your answer must be either (A) or (B)."
    response, history = model.chat(tokenizer, pixel_values, final_q, generation_config, history=history, return_history=True)
    print(f'Assistant: {response}')

    with open(f"{output_dir}/{idx}.txt", "a") as res:
        res.write(f"\n{response}\n")
