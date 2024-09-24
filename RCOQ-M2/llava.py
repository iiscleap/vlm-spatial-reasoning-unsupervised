from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
from accelerate import PartialState

def map_path(n):
    path = f"../data/BLINK/orig_images/val_Spatial_Relation_{n}.jpg"
    return path

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

distributed_state = PartialState()

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


processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-34b-hf", quantization_config=bnb_config, device_map=distributed_state.device)

num_list = list(range(1, 144))
path_list = list(map(map_path, num_list))

with distributed_state.split_between_processes(path_list) as path_new_list:
    for img_path in path_new_list:
        image = Image.open(img_path)
        idx = img_path.split("/")[-1].strip(".jpg")

        conversation = []
        questions_file = f"../data/BLINK/Ques_breakdown/GQ_options/{idx}.txt"
        questions = load_questions(questions_file)
        first_two_questions = "\n".join([questions[q] for q in ['Q1','Q2','Q3','Q4','Q5','Q6']])
        final_question = questions['Q7']

        # First stage: Pass the first six questions
        conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (f"You are a spatial reasoning bot that must answer all of the following 6 questions:\n{first_two_questions}\nYou must provide coherent explanations and answers to all the questions above. Format your answer in the following format: \nAns1: Verbose Answer \nAns2: Verbose Answer \nAns3: Verbose Answer \nAns4: Verbose Answer \nAns5: Verbose Answer \nAns6: Verbose Answer")
                },
                {"type": "image"}
            ]
        })

        # Generate response for the first six questions
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(prompt, image, return_tensors="pt").to(distributed_state.device)
        output = model.generate(**inputs, max_new_tokens=500)
        response = processor.decode(output[0], skip_special_tokens=True)

        clean_response = []
        for line in response.split('\n'):
            if line.startswith("Ans"):
                clean_response.append(line.strip())

        # Save the cleaned and formatted response for the first six answers
        formatted_output = "\n".join(clean_response)
        with open(f"../BLINK_CoT/{idx}.txt", "w") as all_answers_file:
            all_answers_file.write(f"{formatted_output}\n")

        # Second stage: Pass Q7 (final question) with previous answers as context
        conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"You are a spatial reasoning bot that must answer the following question:\n{final_question} \nYou must provide an explanation for your answer and finally your answer must be either (A) or (B)."
                }
            ]
        })

        # Generate response for the final question (Q7)
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(prompt, image, return_tensors="pt").to(distributed_state.device)
        output = model.generate(**inputs, max_new_tokens=500)
        response = processor.decode(output[0], skip_special_tokens=True)

        clean_response = []
        for line in response.split('\n'):
            if line.startswith("Ans"):
                clean_response.append(line.strip())

        # Save the cleaned and formatted response for the final question
        formatted_output = "\n".join(clean_response)
        with open(f"../BLINK_CoT/{idx}.txt", "a") as all_answers_file:
            all_answers_file.write(f"\n{formatted_output}\n")

print("Processing complete!")
