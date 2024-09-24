import argparse
import os
import random
import torch
import csv
from collections import defaultdict
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
cudnn.benchmark = False
cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="MiniGPT Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecated),"
    )
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config(args)
device = 'cuda:{}'.format(args.gpu_id)
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
model = model.eval()

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],  
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)
chat = Chat(model, vis_processor, device=device)

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
    
    if 'Q7' not in questions:
        if 'Q6' in questions:
            questions['Q7'] = questions['Q6']  
        else:
            questions['Q7'] = "No Question" 
    return questions

def load_image(image_path):
    """Loads the image and processes it."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} not found. Skipping...")
    img = Image.open(image_path)
    img_tensor = vis_processor(img)
    return img_tensor

def ask_question(chat, chat_state, img_list, question, temperature=0.6):
    """Function to ask a question and return the response."""
    chat.ask(question, chat_state)
    answer = chat.answer(conv=chat_state, img_list=img_list, num_beams=1, temperature=temperature, max_new_tokens=500, max_length=2000)[0]
    return answer


for i in range(1, 144):
    idx = f"val_Spatial_Relation_{i}"
    image_path = f"../data/BLINK/orig_images/{idx}.jpg"
    questions_file = f"../data/BLINK/Ques_breakdown/GQ_options/{idx}.txt"
    

    questions = load_questions(questions_file)
    first_six_questions = "\n".join([questions[q] for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']])
    final_question = questions['Q7']
    
    qs = f"You are a spatial reasoning bot that must answer all of the following questions:\n{first_six_questions}\nYou must provide coherent explanations and answers to all the questions above. Format your answer in the following format: \nAns1: Verbose Answer \nAns2: Verbose Answer \nAns3: Verbose Answer \nAns4: Verbose Answer \nAns5: Verbose Answer \nAns6: Verbose Answer\n"

    img_tensor = load_image(image_path)
    chat_state = CONV_VISION.copy()
    img_list = [img_tensor]

    chat.encode_img(img_list)

    # Ask the first set of questions
    initial_answer = ask_question(chat, chat_state, img_list, qs)
    print(f"Answer for First 6 Questions: {initial_answer}")

    with open(f"../MiniGPT_BLINK_CoT/{idx}.txt", "a") as cot:
        cot.write(f"{initial_answer}")

    # Ask the final question 
    follow_up_question = f"{final_question}. Select only from the following choices (A) Yes (B) No"
    follow_up_answer = ask_question(chat, chat_state, img_list, follow_up_question)
    print(f"Follow-up Answer: {follow_up_answer}")

    with open(f"../MiniGPT_BLINK_CoT/{idx}.txt", "a") as res:
        res.write(f"\n{follow_up_answer}\n")

