import os
import yaml
import gradio
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
try:
    from auto_gptq import AutoGPTQForCausalLM
except ModuleNotFoundError:
    print("Module 'auto_gptq' with CUDA extensions required for GPTQ models.")

from huggingface_hub import hf_hub_download
from huggingface_hub import login as hf_hub_login
from llama_cpp import Llama  # type: ignore
from llama_formatter import llama_formatter

try:
    import torch
    print("CUDA Available for Pytorch: " + str(torch.cuda.is_available()))
except ModuleNotFoundError:
    print("module 'torch' is not installed or CUDA is not available.")


def init_model_and_tokenizer(model_metadata):
    path = model_metadata["path"] + "/" + model_metadata["name"]
    if not os.path.exists(path):
        os.makedirs(path)

    filename = model_metadata["file"] \
        if 'file' in model_metadata.keys() else "config.json"

    hf_hub_login(token=os.environ.get("HUGGINGFACE_TOKEN"))
    file_path = hf_hub_download(
        repo_id=model_metadata["name"],
        filename=filename,
        cache_dir=path,
        local_dir=path)

    if model_metadata["format"] == "ggml":
        model = Llama(file_path, n_ctx=2048)  # 4096
        tokenizer = None

    elif model_metadata["format"] == "gptq":
        model = AutoGPTQForCausalLM.from_quantized(
                    model_metadata["name"], device_map="auto",
                    use_safetensors=True, use_triton=False)
        tokenizer = AutoTokenizer.from_pretrained(
                        model_metadata["name"])

    else:
        print("Loading model to GPU...")
        model = AutoModelForCausalLM.from_pretrained(
                    model_metadata["name"], device_map="auto",
                    token=True)
        tokenizer = AutoTokenizer.from_pretrained(
                        model_metadata["name"], token=True)

    return model, tokenizer


def bot_model_pipe(token, history, formatter):
    bloviated = False  # True if model got crazy (monologue)
    merged = history[-1][1] + token
    if (formatter.BOS not in merged
            and formatter.EOS not in merged):
        history[-1][1] += token  # Update chat history
        formatter.concat_last(token)  # Concat to last entry
    else:
        bloviated = True  # We need to cut the monologue part
        bos_pos = merged.find(formatter.BOS)
        eos_pos = merged.find(formatter.EOS)
        cut_pos = min(bos_pos, eos_pos)  # Assume is the 1st one
        if (cut_pos == -1):
            cut_pos = max(bos_pos, eos_pos)  # Change to the last one
        history[-1][1] = merged[:cut_pos]  # Cut and update chat hist.
        formatter.replace_last(history[-1][1])  # Replace last entry

    return bloviated


def ui(model_metadata: dict, model: Llama, tokenizer):

    with gradio.Blocks() as ui:
        chatbot = gradio.Chatbot()
        msg = gradio.Textbox()
        clear = gradio.Button("Clear")

        formatter = llama_formatter()
        
        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            # history:
            # - sequence:
            #   - instruction
            #   - response (None for the most recent sequence)
            formatter.add("user", history[-1][0])  # last user message
            history[-1][1] = ""  # reset placeholder for sys response
            prompt = formatter.format()  # format prompt for llama model
            formatter.add("sys", "")  # add placeholder for sys response
            kwargs = dict(temperature=0.6, top_p=0.9)

            print(f"PROMPTS_RAW: {formatter.prompts}")
            print(f"LAST_PROMPT: ---{prompt}---")

            if model_metadata["format"] == 'ggml':
                kwargs["max_tokens"] = 512
                # stream=False do not solve the broken emojies issue
                # https://github.com/abetlen/llama-cpp-python/issues/372
                for chunk in model(prompt=prompt, stream=True, **kwargs):
                    token = chunk["choices"][0]["text"]
                    bloviated = bot_model_pipe(token, history, formatter)
                    yield history
                    if (bloviated):
                        break  # Go back to wait for instructions
            else:

                streamer = TextIteratorStreamer(
                    tokenizer, skip_prompt=True, timeout=30)
                inputs = tokenizer(
                    prompt, return_tensors="pt").to(model.device)
                kwargs["max_new_tokens"] = 512
                kwargs["input_ids"] = inputs["input_ids"]
                kwargs["streamer"] = streamer

                thread = Thread(target=model.generate, kwargs=kwargs)
                thread.start()

                for token in streamer:
                    bloviated = bot_model_pipe(token, history, formatter)
                    yield history
                    if (bloviated):
                        break  # Go back to wait for instructions

        msg.submit(
                user, [msg, chatbot], [msg, chatbot], queue=False
             ).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

    ui.queue()
    ui.launch(share=False, debug=True)  # share=True is insecure!


def main():
    with open("./llama_models.yaml", "r") as f:
        MODELS_METADATA = yaml.safe_load(f)

    model_index = int(os.environ.get("AI_LLAMA2_CHAT_MODEL"))
    assert model_index in range(0, len(MODELS_METADATA)), \
        f"Invalid model index: {model_index}"

    model_metadata = MODELS_METADATA[model_index]
    print(f"Using model: {model_metadata['name']}")

    model_metadata["path"] = \
        os.environ.get("AI_LLAMA2_CHAT_STORE") or "./models"

    model, tokenizer = init_model_and_tokenizer(model_metadata)

    ui(model_metadata, model, tokenizer)


if __name__ == '__main__':
    main()
