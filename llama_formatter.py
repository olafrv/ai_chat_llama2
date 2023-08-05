"""
 This class transform the chat history into a prompt for llama model
 * https://github.com/facebookresearch/llama/tree/main
 * https://github.com/facebookresearch/llama/blob/main/llama/generation.py
 * https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py
 * https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ/discussions/5
"""


class llama_formatter:
    BOS, EOS = "<s>", "</s>"  # Sequence tokens
    B_INST, E_INST = "[INST]", "[/INST]"  # User's instruction tags
    B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"  # System's response tags
    DEFAULT_SYSTEM_PROMPT = """This is my default system prompt."""
    prompts = []         # List of prompts  (author, text)
    prompts_output = ""  # Output prompt incrementally cached
    prompts_index = 0    # Keep track of the last formatted prompt

    def __init__(self):
        self.prompts.append(
            {"author": "sysdef", "text": self.DEFAULT_SYSTEM_PROMPT}
        )

    def add(self, author, text) -> None:
        assert author in ("user", "sys")
        prompt = {"author": author, "text": text}
        self.prompts.append(prompt)

    def empty(self) -> None:
        self.prompts = []
        self.prompts_output = ""
        self.prompts_index = 0

    def concat_last(self, text) -> None:
        self.prompts[-1]["text"] += text

    def replace_last(self, text) -> None:
        self.prompts[-1]["text"] = text

    def format(self) -> str:

        while self.prompts_index < len(self.prompts):

            author = self.prompts[self.prompts_index]["author"]
            text = self.prompts[self.prompts_index]["text"]

            if author == "sysdef":
                self.prompts_output += \
                    f"{self.BOS}{self.B_SYS}{text}{self.E_SYS}{self.EOS}"
            elif author == "user":
                if (text and text.strip() != ""):
                    self.prompts_output += \
                        f"{self.BOS}{self.B_INST} {text.strip()} {self.E_INST}"
            elif author == "sys":
                self.prompts_output += f" {text.strip()}{self.EOS}"

            self.prompts_index += 1

        return self.prompts_output
