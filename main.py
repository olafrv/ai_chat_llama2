import os
import yaml
import gradio
from time import sleep
from llama_prompter import llama_prompter
import argparse


def ui(prompter: llama_prompter):
    # Define the UI for Chatbot
    with gradio.Blocks() as ui:
        chatbot = gradio.Chatbot()  # return history
        textbox = gradio.Textbox()  # return user message
        clearbt = gradio.Button("Clear")  # clear message

        def send_msg(message: str, history: list):
            return "", history + [[message, None]]

        def clear_chat():
            print("Forgeting history...")
            prompter.empty()
            return []

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

            if prompter.model_metadata["architecture"] == 'ggml':
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

        textbox.submit(
                send_msg, [textbox, chatbot], [textbox, chatbot], queue=False
            ).then(bot, chatbot, chatbot)
        clearbt.click(clear_chat, None, chatbot, queue=False)

    # Start the UI for the Chatbot
    ui.queue()
    ui.launch(share=False, debug=True)  # share=True is insecure!


def main():
    # Set model metadata
    with open("./llama_models.yaml", "r") as f:
        MODELS_METADATA = yaml.safe_load(f)

    print("MODEL_INDEXES:")
    for i, model_metadata in enumerate(MODELS_METADATA):
        print(f"{i}: {model_metadata['architecture']}")

    # Arguments
    parser = argparse.ArgumentParser(
                    prog='AI Llama2 Chatbot',
                    description='Llama2 LLM Model Chatbot')
    parser.add_argument('integers', metavar='MODEL_INDEX', type=int,
                        help='model index')
    parser.parse_args()

    # Set model
    model_index = int(os.sys.argv[1])
    assert model_index in range(0, len(MODELS_METADATA)), \
        f"Invalid model index: {model_index}"

    model_metadata = MODELS_METADATA[model_index]
    print(f"MODEL_NAME: {model_metadata['name']}")

    # Set model store path
    if ('path' not in model_metadata):
        model_metadata["path"] = \
            os.environ.get("AI_LLAMA2_CHAT_STORE") or "./models"
    model_metadata["path"] += "/" + model_metadata["name"]
    if not os.path.exists(model_metadata["path"]):
        os.makedirs(model_metadata["path"])
    print(f"MODEL_PATH: {model_metadata['path']}")

    # Create model prompter
    print("Initializing model prompter...")
    model_prompter = llama_prompter(model_metadata,
                                    os.environ.get("HUGGINGFACE_TOKEN"))
    # Start UI of Chatbot
    ui(model_prompter)


if __name__ == '__main__':
    main()
