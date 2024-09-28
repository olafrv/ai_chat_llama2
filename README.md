# AI Chat Llama2

ChatBot using [Meta AI Llama v2 LLM models](https://ai.meta.com/llama/) 
on your local PC (some without GPU but a bit slow if not enough RAM).

<a href="ai_chat_llama2.png"><img src="ai_chat_llama2.png"></a> 

```bash
(...)
MODEL_DEVICE: cuda:0
Model loaded.
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
PROMPTS_RAW: [{'author': 'sysdef', 'text': 'This is my default system prompt.'}, {'author': 'user', 'text': 'Show me emojies?'}, {'author': 'sys', 'text': ''}]
LAST_PROMPT: ---<s><<SYS>>This is my default system prompt.<</SYS>></s><s>[INST] Show me emojies? [/INST]---
/home/ubuntu/code/github/ai_chat_llama2/venv/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:601: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
/home/ubuntu/code/github/ai_chat_llama2/venv/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:606: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
PROMPTS_RAW: [{'author': 'sysdef', 'text': 'This is my default system prompt.'}, {'author': 'user', 'text': 'Show me emojies?'}, {'author': 'sys', 'text': ' Of course! Here are some emojis you can use:\n\n😊👍💬👀💕😍'}, {'author': 'user', 'text': 'Nice, what day is today?'}, {'author': 'sys', 'text': ''}]
LAST_PROMPT: ---<s><<SYS>>This is my default system prompt.<</SYS>></s><s>[INST] Show me emojies? [/INST] Of course! Here are some emojis you can use:

😊👍💬👀💕😍</s><s>[INST] Nice, what day is today? [/INST]---
PROMPTS_RAW: [{'author': 'sysdef', 'text': 'This is my default system prompt.'}, {'author': 'user', 'text': 'Show me emojies?'}, {'author': 'sys', 'text': ' Of course! Here are some emojis you can use:\n\n😊👍💬👀💕😍'}, {'author': 'user', 'text': 'Nice, what day is today?'}, {'author': 'sys', 'text': ' Today is March 28th! 🌞🌻🎉'}, {'author': 'user', 'text': 'White a simple hello world in python 3?'}, {'author': 'sys', 'text': ''}]
LAST_PROMPT: ---<s><<SYS>>This is my default system prompt.<</SYS>></s><s>[INST] Show me emojies? [/INST] Of course! Here are some emojis you can use:

😊👍💬👀💕😍</s><s>[INST] Nice, what day is today? [/INST] Today is March 28th! 🌞🌻🎉</s><s>[INST] White a simple hello world in python 3? [/INST]---
```

## ChatBot Usage

> The best model is the GPTQ [Quantized](https://huggingface.co/docs/optimum/concept_guides/quantization) but requires GPU, see [llama_models.yaml](llama_models.yaml).

> Register at https://huggingface.co to get a token, ask for download access to the models, and [later train them with Autotrain](https://huggingface.co/docs/autotrain/index)

```bash
git clone https://github.com/olafrv/ai_chat_llama.git
cd ai_chat_llama
sudo apt install make
# make help
make install  # If fails see NVIDIA section below
# export MODEL_STORE=./models
export HUGGINGFACE_TOKEN=***********
# Llama v2 models will be downloaded (10-20 GiB / each)
make run MODEL_INDEX=2 # gptq
# Navigate in your browser to 127.0.0.1:7860
```

## Model Training (Draft)

> **TODO:** Training passed but prompt require testing.

Train the base LLAMA v2 original model with custom data set:

```bash
make train-original  # meta-llama/Llama-2-7b-chat-hf
```

I expect that in a couple of months we can use AutoTrain:
```bash
# Llama2 is not supported by AutoTrain (Aug/2023)
# autotrain llm --help
# autotrain setup --update-torch  # Only if using Google Collab
# autotrain setup
# HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 autotrain llm --train \
#			--data_path datasets/olafrv/fmt \
#			--model meta-llama/Llama-2-7b-chat-hf \
#			--text_column text --learning_rate 2e-4 --num_train_epochs 3 \
#			--train_batch_size 12 --block_size 1024 --use_peft \
#			--project_name olafrv/Llama-2-7b-chat-hf-trained \
#			--use_int4 --trainer sft > logs/training.log &
# tail -f logs/training.log
```

## WSL v2 increasing RAM and Swap

To increase the RAM and SWAP memory on Windows Subsystem for Linux v2:
```powershell
# https://learn.microsoft.com/en-us/windows/wsl/wsl-config

# As Local User
Start-Process -File notepad.exe -ArgumentList "$env:userprofile/.wslconfig"

# Content of .wslconfig:
# [wsl2]
# memory=25GB
# swap=25GB

# Stop the VM
wsl --shutdown

# As Local Administrator
Restart-Service LxssManager
```

## NVIDIA GPU Driver and Utilities

### The Hardware

I will describe here the hard way of getting NVIDIA drivers,
pytorch, AutoGPTQ, urllib3 and many other stuff to work under
Windows Subsystem for Linux v2, where I was running tests.
But on bare metal or ML/GPU cloud intances gets easier.

My hardware was an ASUS ROG Strix G713RW laptop with:

* AMD Ryzen 9 6900HX 32GB DDR5 with Radeon Graphics.
* NVIDIA GeForce RTX 3070 Ti 8GB GDDR6 Laptop Edition. 

The complications are:

* Host OS Windows 11 Pro 64 bits (AMD):
  * Windows Virtulization Platform + WSL v2 features enabled.
  * Device Security -> Core Isolation -> Memory Integraty -> Off.
  * NVIDIA Driver Version 560.94 supports Direct 3D 12.1.
* Guest Operating System Ubuntu 22.04 x86-64 (not AMD-64):
  * CUDA Driver Version = 12.6 (Installed on Linux from NVIDIA site).

Before running `make install` of AI Chat Llama v2, and only 
if your are going to use GPU power, then this has to be
configured manually (I'm too lazy to *Makify* it).

### Pre-flight checks on the Linux Guest.

Check first what is already built-in in the WSL Linux image:

```bash
nvidia-smi
```

Output should be like this (python3.10 is the running the Chatbot):
```bash
Fri Sep 27 23:44:00 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.02              Driver Version: 560.94         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3070 ...    On  |   00000000:01:00.0  On |                  N/A |
| N/A   57C    P8             16W /  130W |    6053MiB /   8192MiB |      3%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A     75869      C   /python3.10                                 N/A      |
+-----------------------------------------------------------------------------------------+
```

(Optional) You can play a bit with the NVIDIA Container Toolkit (If you have docker):
```
sudo apt-get install -y nvidia-docker2
sudo docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```
Output should be like this:
```bash
(...)
GPU Device 0: "Ampere" with compute capability 8.6
> Compute 8.6 CUDA device: [NVIDIA GeForce RTX 3070 Ti Laptop GPU]
47104 bodies, total time for 10 iterations: 48.482 ms
= 457.649 billion interactions per second
= 9152.976 single-precision GFLOP/s at 20 flops per interaction
```

### Installation of NVIDIA CUDA Driver Libraries (Source Code)

This is needed so Python (pip) can compile the necesary ML packages for your CUDA Architecture:

```bash
###
# Downloads/Documentation:
# https://developer.nvidia.com/cuda-downloads (Linux > Installer Type > deb(network))
# https://developer.nvidia.com/cuda-toolkit-archive (For older version, incl. docs.)
# Tested:
# CUDA 12.6 - 
# CUDA 12.1 - Not supported by PyTorch (Aug/2023) breaks AutoGPTQ CUDA ext. compilation.
# CUDA 11.8 - Compiles with PyTorch / AutoGPTQ and my works with my RTX 3070.
###
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

```bash
## Test the CUDA code compilation
git clone https://github.com/nvidia/cuda-samples
cd cuda-samples/Samples/1_Utilities/deviceQuery
make  # It must compile for your GPU natively, no GCC flags
./deviceQuery
(...)
Device 0: "NVIDIA GeForce RTX 3070 Ti Laptop GPU"
  CUDA Driver Version / Runtime Version          12.6 / 12.6
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 8192 MBytes (8589410304 bytes)
  (046) Multiprocessors, (128) CUDA Cores/MP:    5888 CUDA Cores
(...)
```

FInally, you can `make install` the AI Chat Llama v2.

## References

### Meta AI Llama v2 LLM Model

* Llama v2 model code examples: 
  * https://huggingface.co/blog/llama2
  * https://github.com/facebookresearch/llama.git
* Llama v2 pre-trained model download (e-mail with signed link):
  * https://ai.meta.com/resources/models-and-libraries/llama-downloads/
* Llama v2 pre-trained models on Hugging Face: 
  * For GPUs: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
  * GPTQ Quantized: https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ
  * GGML Quantized: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
* Transformer Reinforcement Learning
  * https://huggingface.co/docs/trl/main/en/installation
* Supervised Fine-tuning Trainer
  * https://huggingface.co/docs/trl/main/en/sft_trainer

### HuggingFace

* https://huggingface.co
* https://huggingface.co/docs/huggingface_hub/quick-start
* https://huggingface.co/docs/autotrain/index

### GRadio
* https://www.gradio.app/guides/quickstart
* https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks

### NVIDIA CUDA on Windows Subsystem for Linux v2 (aka WSL2):

* https://developer.nvidia.com/cuda/wsl
* https://developer.nvidia.com/cuda-downloads
* https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
* https://documentation.ubuntu.com/wsl/en/latest/tutorials/gpu-cuda/
* https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl
* https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute
