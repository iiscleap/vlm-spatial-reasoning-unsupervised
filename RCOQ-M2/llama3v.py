import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
import requests
from PIL import Image
from io import BytesIO
import re
from llava.utils import disable_torch_init
import csv

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

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
            
    return questions

disable_torch_init()

model_path = "LLaVA-Phi-3-mini-4k-instruct"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

for i in range(1,144):
    idx = f"val_Spatial_Relation_{i}"
    image_path = f"../data/BLINK/orig_images/{idx}.jpg"
    questions_file = f"../data/BLINK/Ques_breakdown/GQ_options/{idx}.txt"
    questions = load_questions(questions_file)
    first_six_questions = "\n".join([questions[q] for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']])
    final_question = questions['Q7']    
    qs = f"<image>\nYou are a spatial reasoning bot that must answer all of the following questions:\n{first_six_questions}\nYou must provide coherent explanations and answers to all the questions above. Format your answer in the following format: \nAns1: Verbose Answer \nAns2: Verbose Answer \nAns3: Verbose Answer \nAns4: Verbose Answer \nAns5: Verbose Answer \nAns6: Verbose Answer"

    
    conv_mode = "phi3_instruct"

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    image_files = [image_path]
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    temperature = 0.6
    top_p = 0.7
    num_beams = 1
    max_new_tokens = 512

    with torch.inference_mode():
        output_ids = model.generate(
        input_ids,
        images=images_tensor,
        image_sizes=image_sizes,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    outputs = outputs.replace("<|end|>", "").strip()
    print(f"\n{outputs}\n")

    with open(f"../LLaMA-3-V/CoT_Answers/{idx}.txt", "a") as cot:
        cot.write(f"{outputs}")
    final_q = final_question
    qs_next = f"You are a spatial reasoning bot that must answer the following question:\n{final_q} Select from the following choices.(A) True (B) False\nYou must provide an explanation for your answerand finally your answer must be either (A) or (B)."
    conv_next = conv_templates[conv_mode].copy()
    conv_next.append_message(conv_next.roles[0], qs)
    conv_next.append_message(conv_next.roles[1], outputs)

    conv_next.append_message(conv_next.roles[0], qs_next)
    conv_next.append_message(conv_next.roles[1], None)

    prompt = conv_next.get_prompt()


    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    outputs = outputs.replace("<|end|>", "").strip()
    print(f"\n{outputs}\n")

    with open(f"../LLaMA-3-V/CoT_Answers/{idx}.txt", "a") as res:
        res.write(f"{outputs}\n")
