import argparse
import os
import random
from collections import defaultdict
import cv2
import re
import numpy as np
from PIL import Image
import torch
import html
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import csv
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cudnn.benchmark = False
cudnn.deterministic = True

# Initialize model and config
print('Initializing Chat')
args = parse_args()
cfg = Config(args)
device = 'cuda:{}'.format(args.gpu_id)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
bounding_box_size = 100

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

model = model.eval()

# Conversation configuration with separator for chat history
CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],  # Store conversation history here
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

chat = Chat(model, vis_processor, device=device)
print('Initialization Finished')

def get_prompt_for_idx(filename, target_idx):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['idx'] == target_idx:
                return row['prompt']
    return None

def load_image(image_path):
    """Loads the image and processes it."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} not found. Skipping...")
    img = Image.open(image_path)
    img_tensor = vis_processor(img)
    return img_tensor

def ask_question(chat, chat_state, img_list, question, temperature=0.6):
    """Function to ask a question and return the response."""
    # Append the new question to the chat state
    chat.ask(question, chat_state)
    # Generate the answer based on the current chat state and image
    answer = chat.answer(conv=chat_state, img_list=img_list, num_beams=1, temperature=temperature, max_new_tokens=500, max_length=2000)[0]
    return answer

object_dict = {}
with open("vlm-spatial-reasoning-unsupervised/data/BLINK/object_list.txt","r") as obj_file:
    for line in obj_file:
        idx, objs = line.split(" ", maxsplit = 1)
        object_dict[idx] = objs

for i in range(1,144):
    idx = f"val_Spatial_Relation_{i}"
    image_path =  f"vlm-spatial-reasoning-unsupervised/data/BLINK/orig_images/{idx}.jpg"

    ## crafting the prompt
    objs = object_dict[idx]
    obj1,obj2 = objs.split("," , maxsplit = 1)
    q1 = f"Q1. Is there a {obj1} in the image?"
    q2 = f"Q2. Is there a {obj2} in the image?"
    q3 = f"Q3. Where is the {obj1} in the image?"
    q4 = f"Q4. Where is the {obj2} in the image?"
    q5 = f"Q5. Where are the {obj1} and the {obj2} with respect to each other?"
    qs = f"<ImageHere> You are a spatial reasoning bot that must answer all of the following questions:\n{q1}\n{q2}\n{q3}\n{q4}\n{q5}\nYou must provide coherent and verbose explanations and answers to all the questions above."

    temperature = 0.6

    # Load and process image
    img_tensor = load_image(image_path)
    chat_state = CONV_VISION.copy()
    img_list = [img_tensor]

    # Encode the image (this step is required once at the start)
    chat.encode_img(img_list)

    # Ask the initial question (which includes the image)
    initial_answer = ask_question(chat, chat_state, img_list, qs, temperature)
    
    with open(f"vlm-spatial-reasoning-unsupervised/RCOQ-M1/output/minigpt/{idx}.txt", "a") as cot:
        cot.write(f"{initial_answer}\n")

    # No <ImageHere> placeholder needed for follow-up questions
    qs_next = get_prompt_for_idx("vlm-spatial-reasoning-unsupervised/data/BLINK/prompt.csv", idx)
    follow_up_question = f"Now answer the following question:\n{qs_next}\nProvide a coherent and verbose explanation, based on the previous answers. Your answer must be either (A) or (B)."

    # Ask a follow-up question (chat state will contain the previous Q&A history)
    follow_up_answer = ask_question(chat, chat_state, img_list, follow_up_question, temperature)

    with open(f"vlm-spatial-reasoning-unsupervised/RCOQ-M1/output/minigpt/{idx}.txt", "a") as cot:
        cot.write(f"{follow_up_answer}\n")