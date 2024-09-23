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

def get_prompt_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['prompt']
    return None

disable_torch_init()

model_path = "LLaVA-Phi-3-mini-4k-instruct"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

object_dict = {}
with open("vlm-spatial-reasoning-unsupervised/data/BLINK/object_list.txt","r") as obj_file:
    for line in obj_file:
        idx, objs = line.split(" ", maxsplit = 1)
        object_dict[idx] = objs

for i in range(1,144):
    idx = f"val_Spatial_Relation_{i}"
    image_path = f"vlm-spatial-reasoning-unsupervised/data/BLINK/orig_images/{idx}.jpg"

    ## crafting the prompt
    objs = object_dict[idx]
    obj1,obj2 = objs.split("," , maxsplit = 1)
    q1 = f"Q1. Is there a {obj1} in the image?"
    q2 = f"Q2. Is there a {obj2} in the image?"
    q3 = f"Q3. Where is the {obj1} in the image?"
    q4 = f"Q4. Where is the {obj2} in the image?"
    q5 = f"Q5. Are the {obj1} and the {obj2} interacting with each other?"
    q6 = f"Q6. What is the spatial relationship between the {obj1} and the {obj2}?"
    qs = f"You are a spatial reasoning bot that must answer all of the following questions:\n{q1}\n{q2}\n{q3}\n{q4}\n{q5}\n{q6}\nYou must provide coherent explanations and answers to all the questions above."

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

    temperature = 0.2
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

    with open(f"vlm-spatial-reasoning-unsupervised/RCOQ-M1/output/llama3v/{idx}.txt", "a") as cot:
        cot.write(f"{outputs}\n")

    qs_next = get_prompt_for_idx("vlm-spatial-reasoning-unsupervised/data/BLINK/prompt.csv", idx)
    qs_next = f"Now answer only the following question:\n{qs_next}\nAnswer only this question, nothing else. You must provide an explanation for your answer. Your answer must be either (A) or (B)."
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

    with open(f"vlm-spatial-reasoning-unsupervised/RCOQ-M1/output/llama3v/{idx}.txt", "a") as cot:
        cot.write(f"{outputs}\n")