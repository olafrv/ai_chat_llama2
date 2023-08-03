class llama_formatter:
    # https://github.com/facebookresearch/llama/tree/main
    # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212
    # https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ/discussions/5
    # https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py

    BOS, EOS = "<s>", "</s>"  # Sequence tokens
    B_INST, E_INST = "[INST]", "[/INST]"  # User's instruction tags
    B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"  # System's response tags
    DEFAULT_SYSTEM_PROMPT = """This is my default system prompt."""
    prompts = None
    prompts_output = ""
    prompts_index = 0

    def __init__(self):
        self.prompts = [
            {"author": "sysdef", "text": self.DEFAULT_SYSTEM_PROMPT}
        ]

    def add(self, author, text):
        assert author in ("user", "sys")
        prompt = {"author": author, "text": text}
        self.prompts.append(prompt)
        return format(prompt)

    def concat_last(self, text):
        self.prompts[-1]["text"] += text

    def update_last(self, text):
        self.prompts[-1]["text"] = text

    def format(self):

        while self.prompts_index < len(self.prompts):

            author = self.prompts[self.prompts_index]["author"]
            text = self.prompts[self.prompts_index]["text"]

            if author == "sysdef":
                # Tokenizer adds BOS at encoding, so BOS not added here?
                self.prompts_output += \
                    f"{self.B_SYS}{text}{self.E_SYS}{self.EOS}"
            elif author == "user":
                if (text and text.strip() != ""):
                    self.prompts_output += \
                        f"{self.BOS}{self.B_INST} {text.strip()} {self.E_INST}"
            elif author == "sys":
                self.prompts_output += f" {text.strip()}{self.EOS}"

            self.prompts_index += 1

        return self.prompts_output