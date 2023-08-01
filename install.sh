#!/bin/bash

# Register at https://huggingface.co to use:
# https://huggingface.co/docs/autotrain/index

git clone https://github.com/facebookresearch/llama.git

# Use Llama v2 Download URL from authorization e-mail requested at:
# https://ai.meta.com/resources/models-and-libraries/llama-downloads/
# cd ./llama; ./download.sh; # Enter 7B-chat in the model list.
# This will create the directory ./llama/llama-2-7b-chat/
# Size to download is about 15GB considering what is listed at:
# https://huggingface.co/meta-llama/Llama-2-7b-chat/tree/main

pip install -r llama/requirements.txt
pip install -r requirements.txt
