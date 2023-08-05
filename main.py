import os
import gradio
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
# from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from llama_cpp import Llama  # type: ignore
from llama_formatter import llama_formatter

MODELS_METADATA = (
    {
        "format": "gpu",
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "file": None,
        "path": "./models"
    },
    {
        "format": "ggml",
        "name": "TheBloke/Llama-2-7B-Chat-GGML",
        "file": "llama-2-7b-chat.ggmlv3.q4_K_M.bin",
        "path": "./models"
    },
    {
        "format": "gptq",
        "name": "TheBloke/Llama-2-7b-Chat-GPTQ",
        "file": None,
        "path": "./models"
    }
)


def init_model_and_tokenizer(model_metadata):
    if not os.path.exists(model_metadata["path"]):
        os.makedirs(model_metadata["path"])

    if model_metadata["format"] == "ggml":
        file_path = hf_hub_download(
            repo_id=model_metadata["name"],
            filename=model_metadata["file"],
            local_dir=model_metadata["path"])
        model = Llama(file_path, n_ctx=2048)  # 4096
        tokenizer = None
    elif model_metadata["format"] == "gptq":
        # !!! from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(
            model_metadata["name"], device_map="auto",
            use_safetensors=True, use_triton=False)
        tokenizer = AutoTokenizer.from_pretrained(model_metadata["name"])
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_metadata["name"], device_map="auto", token=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_metadata["name"], token=True)

    return model, tokenizer


# https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks
def ui(model_metadata: dict, model: Llama, tokenizer):

    with gradio.Blocks() as ui:
        formatter = llama_formatter()
        chatbot = gradio.Chatbot()
        msg = gradio.Textbox()
        clear = gradio.Button("Clear")

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
            print(f"PROMPT: ---{prompt}---")

            if model_metadata["format"] == 'ggml':
                kwargs["max_tokens"] = 512
                # stream=False
                # https://github.com/abetlen/llama-cpp-python/issues/372
                for chunk in model(prompt=prompt, stream=True, **kwargs):
                    token = chunk["choices"][0]["text"]
                    together = history[-1][1] + token
                    print(f"STREAM: {together}")
                    if (formatter.BOS not in together
                            and formatter.EOS not in together):
                        history[-1][1] += token  # Update chatbot
                        formatter.concat_last(token)
                        yield history
                    else:
                        print("Model is getting crazy!")
                        together = together[:together.find(formatter.BOS)]
                        together = together[:together.find(formatter.EOS)]
                        history[-1][1] = together
                        formatter.update_last(history[-1][1])
                        yield history
                        break
            else:

                streamer = TextIteratorStreamer(
                    tokenizer, skip_prompt=True, Timeout=5)
                inputs = tokenizer(
                    prompt, return_tensors="pt").to(model.device)
                kwargs["max_new_tokens"] = 512
                kwargs["input_ids"] = inputs["input_ids"]
                kwargs["streamer"] = streamer
                thread = Thread(target=model.generate, kwargs=kwargs)
                thread.start()

                for token in streamer:
                    history[-1][1] += token  # Update chatbot
                    formatter.update_last(token)
                    yield history

        msg.submit(
                user, [msg, chatbot], [msg, chatbot], queue=False
             ).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

    ui.queue()
    ui.launch(share=True, debug=True)


def main():
    model_metadata = MODELS_METADATA[1]
    model, tokenizer = init_model_and_tokenizer(model_metadata)
    ui(model_metadata, model, tokenizer)


if __name__ == '__main__':
    main()
