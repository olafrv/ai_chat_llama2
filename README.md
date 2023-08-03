# AI Chat Llama2

ChatBot using Meta AI Llama v2 LLM model.

<a href="ai_chat_llama2.png"><img src="ai_chat_llama2.png"></a> 

## ChatBot Running

> Only working for GGML model (hardcoded), I need time to test others.

```bash
git clone https://github.com/olafrv/ai_chat_llama.git
cd ai_chat_llama
make install
make run
```

## Model Training (Draft)

* Register at https://huggingface.co to use AutoTrain Advanced:
https://huggingface.co/docs/autotrain/index

* Now install python environment and setup autotrain:
```
# If you face an error with llama-cpp-python see requirements.txt
make install
autotrain setup
```

* Train the base LLAMA v2 model with custom data set:
```bash
# autotrain llm --help
# autotrain setup --update-torch  # Only if using Google Collab
autotrain llm --train \
--data_path . \
--model meta-llama/Llama-2-7b-hf \
--learning_rate 2e-4 \
--num_train_epochs 3 \
--train_batch_size 12 \ 
--block-size 2048
--use_peft \
--train_on_inputs \
--project_name llama_trained \
--use_int4 \
--trainer sft
```

## References

* Meta AI Llama v2 LLM Model:
  * Llama v2 model code examples: 
    * https://github.com/facebookresearch/llama.git
  * Llama v2 pre-trained model download (e-mail with signed link):
    * https://ai.meta.com/resources/models-and-libraries/llama-downloads/
  * Llama v2 pre-trained models on Hugging Face: 
    * For GPUs: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
    * GPTQ Quantized: https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ
    * GGML Quantized: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
* HuggingFace:
  * https://huggingface.co
  * https://huggingface.co/docs/huggingface_hub/quick-start
  * https://huggingface.co/docs/autotrain/index
* GRadio: https://www.gradio.app/guides/quickstart
* Tricky references in the [main.py](main.py) source code.
