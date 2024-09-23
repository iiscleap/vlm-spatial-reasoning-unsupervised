# RCOQ-M1: Running Experiments with Multiple Vision-Language Models

This repository provides the scripts to run experiments with several popular Vision-Language Models. Follow the instructions below to set up the environment for each model and run the corresponding experiments.

## Running LLaVA Model

1. Set up the environment according to the instructions on the official LLaVA model page:  
   [LLaVA v1.6 34B on Hugging Face](https://huggingface.co/llava-hf/llava-v1.6-34b-hf)

2. To run our experiment:
    ```bash
    cd RCOQ-M1
    python llava.py
    ```

## Running InternVL Model

1. Set up the environment according to the instructions on the official InternVL model page:  
   [InternVL Chat V1.2 on Hugging Face](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2)

2. To run our experiment:
    ```bash
    cd RCOQ-M1
    python internvl.py
    ```

## Running LLaMA3v Model

1. Set up the environment according to the instructions on the official LLaMA3v model page:  
   [LLaVA-pp on GitHub](https://github.com/mbzuai-oryx/LLaVA-pp)

2. To run our experiment:

   Copy llama3v.py into RCOQ-M1/LLaVA-pp/LLaVA
    ```bash
    cd RCOQ-M1/LLaVA-pp/LLaVA
    python llama3v.py
    ```

## Running MiniGPT V2 Model

1. Set up the environment according to the instructions on the official MiniGPT-4 model page:  
   [MiniGPT-4 on GitHub](https://github.com/Vision-CAIR/MiniGPT-4)

2. To run our experiment:

   Copy minigpt.py into RCOQ-M1/MiniGPT-4
    ```bash
    cd RCOQ-M1/MiniGPT-4
    python minigpt.py
    ```

