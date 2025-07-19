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
sudo apt install make  # make help
# NVIDIA CUDA GPU support for Docker/WSL
# https://github.com/olafrv/nvidia-docker-wsl
make install  # If fails see NVIDIA section above
# export MODEL_STORE=./models
export HUGGINGFACE_TOKEN=***********
# Llama v2 models will be downloaded (10-20 GiB / each)
make run MODEL_INDEX=2 # gptq
# Navigate in your browser to 127.0.0.1:7860
```

## Model Training (Draft)

> **TODO:** Training requires RAM/GRAM also datasets are tricky.

Train the base LLAMA v2 original model with custom data set:

```bash
make train-*
make autotrain-*
```

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
