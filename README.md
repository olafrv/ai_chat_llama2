# AI Chat using LLAMA v2 LLM Model

## Running

```bash
git clone https://github.com/olafrv/ai_chat_llama.git
cd ai_chat_llama
make install
make run
```

## Training (Draft)

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

* Llama v2 model code examples: 
  * https://github.com/facebookresearch/llama.git
* Llama v2 pre-trained model download (e-mail with signed link):
  * https://ai.meta.com/resources/models-and-libraries/llama-downloads/
* Llama v2 pre-trained models Hugging Face: 
  * For GPUs: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
  * GPTQ Quantized: https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ
  * GGML Quantized: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
* HuggingFace CLI:
  * https://huggingface.co
  * https://huggingface.co/docs/huggingface_hub/quick-start
  * https://huggingface.co/docs/autotrain/index
* GRadio: https://www.gradio.app/guides/quickstart