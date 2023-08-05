import os
import yaml
import gradio
from llama_prompter import llama_prompter


def ui(prompter: llama_prompter):
    # UI for Chatbot
    with gradio.Blocks() as ui:
        chatbot = gradio.Chatbot()
        msg = gradio.Textbox()
        clear = gradio.Button("Clear")

        def user(user_message: str, history: list):
            return "", history + [[user_message, None]]

        def bot(history: list):
            # history -> prompter.stack:
            # - sequence:
            #   - [0] user instruction
            #   - [1] sys response (None for the most recent sequence)
            prompter.stack("user", history[-1][0])  # last user message
            history[-1][1] = ""  # reset placeholder for sys response
            prompt = prompter.get_prompt()  # format prompt for llama model
            prompter.stack("sys", "")  # add placeholder for sys response

            print(f"PROMPTS_RAW: {prompter.formatter.prompts}")
            print(f"LAST_PROMPT: ---{prompt}---")

            if prompter.model_metadata["format"] == 'ggml':
                for chunk in prompter.submit(prompt):
                    token = chunk["choices"][0]["text"]
                    bloviated = prompter.check_history(token, history)
                    yield history
                    if (bloviated):
                        break  # Go back to wait for instructions
            else:
                for token in prompter.submit(prompt):
                    bloviated = prompter.check_history(token, history)
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
    # Set model metadata
    with open("./llama_models.yaml", "r") as f:
        MODELS_METADATA = yaml.safe_load(f)
    model_index = int(os.environ.get("AI_LLAMA2_CHAT_MODEL"))
    assert model_index in range(0, len(MODELS_METADATA)), \
        f"Invalid model index: {model_index}"
    model_metadata = MODELS_METADATA[model_index]
    print(f"Using model: {model_metadata['name']}")

    # Set model store path
    model_metadata["path"] = \
        os.environ.get("AI_LLAMA2_CHAT_STORE") or "./models"
    print(f"Store path: {model_metadata['path']}")

    # Create model prompter
    model_prompter = llama_prompter(model_metadata,
                                    os.environ.get("HUGGINGFACE_TOKEN"))
    # Start UI of Chatbot
    ui(model_prompter)


if __name__ == '__main__':
    main()
