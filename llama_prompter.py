"""
 File: llamma_prompter.py

 Provides a simple prompt-based conversation interface for a language model
 (Llama 2), acting as a stream handler and allowing the user to input commands
 and receive responses the model while preventing model monologues.
"""

from threading import Thread
from huggingface_hub import hf_hub_download
from huggingface_hub import login as hf_hub_login
from llama_cpp import Llama  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
try:
    import torch
    print("CUDA Available for Pytorch: " + str(torch.cuda.is_available()))
except ModuleNotFoundError:
    print("module 'torch' is not installed or CUDA is not available.")
try:
    from auto_gptq import AutoGPTQForCausalLM
except ModuleNotFoundError:
    print("Module 'auto_gptq' with CUDA extensions required for GPTQ models.")

from transformers import TextIteratorStreamer
from llama_formatter import llama_formatter


class llama_prompter:
    timeout = 60*60*24  # seconds
    model_metadata = None
    model = None
    tokenizer = None
    formatter = None

    """ Initialize the model and tokenizer from metadata specifications """
    def __init__(self, model_metadata: dict, huggingface_token: str):
        self.model_metadata = model_metadata
        self.formatter = llama_formatter()

        if model_metadata["online"]:
            print("Downloading the model...")
            hf_hub_login(token=huggingface_token)
            file_path = hf_hub_download(
                    repo_id=model_metadata["name"],
                    filename=model_metadata["file"],
                    local_dir=model_metadata["path"],
                    local_dir_use_symlinks=False
                )
        else:
            pass

        # https://huggingface.co/docs/transformers/index
        print("Loading the model...")
        if model_metadata["format"] == "ggml":
            self.model = Llama(file_path, n_ctx=2048)  # 4096
            self.tokenizer = None

        elif model_metadata["format"] == "gptq":
            self.model = AutoGPTQForCausalLM.from_quantized(
                            model_metadata["name"],
                            device_map="auto",
                            use_safetensors=True,
                            use_triton=False)
            self.tokenizer = AutoTokenizer.from_pretrained(
                            model_metadata["name"])

        elif model_metadata["format"] == "original":
            self.model = AutoModelForCausalLM.from_pretrained(
                        model_metadata["name"],
                        device_map="auto",
                        token=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                            model_metadata["name"],
                            token=True)

        elif model_metadata["format"] == "tlrsft":
            # https://github.com/huggingface/peft/issues/774 !!!
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                            model_metadata["path"],
                            device_map=None,
                            offload_folder="offload/",
                            token=True
                        )

            self.tokenizer = AutoTokenizer.from_pretrained(
                            model_metadata["path"],
                            device_map=None,
                            token=True)

        if hasattr(self.model, 'device'):
            print(f"MODEL_DEVICE: {self.model.device}")
        else:
            print("MODEL_DEVICE: Undefined")

        print("Model loaded.")

    """ Get the current prompt text in llama v2 format """
    def get_prompt(self) -> str:
        return self.formatter.format()

    """ Add a new text to the stack of prompts """
    def stack(self, role: str, text: str) -> None:
        self.formatter.add(role, text)

    def empty(self) -> None:
        self.formatter.empty()

    """ Submit a prompt to the model and return a streamer object """
    def submit(self, prompt: str):
        kwargs = dict(temperature=0.6, top_p=0.9)
        if self.model_metadata["format"] == 'ggml':
            kwargs["max_tokens"] = 512
            # stream=False do not solve the broken emojies issue
            # https://github.com/abetlen/llama-cpp-python/issues/372
            streamer = self.model(prompt=prompt, stream=True, **kwargs)
        else:
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, timeout=self.timeout)
            if self.model_metadata["format"] == "tlrsft":
                inputs = self.tokenizer(
                    prompt, return_tensors="pt")  # device_map = None
            else:
                inputs = self.tokenizer(
                    prompt, return_tensors="pt").to(self.model.device)
            kwargs["max_new_tokens"] = 512
            kwargs["input_ids"] = inputs["input_ids"]
            kwargs["streamer"] = streamer

            thread = Thread(target=self.model.generate, kwargs=kwargs)
            thread.start()
        return streamer

    """ Prevent model monologues by checking the chat history"""
    def check_history(self, new_text: str, history: list) -> bool:
        bloviated = False  # True if model got crazy (monologue)
        merged = history[-1][1] + new_text
        if (self.formatter.BOS not in merged
                and self.formatter.EOS not in merged):
            history[-1][1] += new_text  # Update chat history
            self.formatter.concat_last(new_text)  # Concat to last entry
        else:
            bloviated = True  # We need to cut the monologue part
            bos_pos = merged.find(self.formatter.BOS)
            eos_pos = merged.find(self.formatter.EOS)
            cut_pos = min(bos_pos, eos_pos)  # Assume is the 1st one
            if (cut_pos == -1):
                cut_pos = max(bos_pos, eos_pos)  # Change to the last one
            history[-1][1] = merged[:cut_pos]  # Cut and update chat hist.
            self.formatter.replace_last(history[-1][1])  # Replace last entry

        return bloviated
