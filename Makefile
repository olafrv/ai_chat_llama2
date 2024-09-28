#!/usr/bin/make

PYTHON_VENV_DIR?=./venv

.PHONY: help collect install install-dev install-venv install-base uninstall uninstall-venv clean run train-original

help:
	@echo 'ENVIRONMENT: ## PYTHON_VENV_DIR' | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'
	@grep -E '^[\.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'

collect:	## Collect and shows environment (sensitive values!)
	$(foreach v, $(filter-out .VARIABLES,$(.VARIABLES)), $(info $(v) = $($(v))))
	@ . ${PYTHON_VENV_DIR}/bin/activate \
		&& python3 -m torch.utils.collect_env

install: install-venv	## Install dependencies for production
	@ true \
		&& . ${PYTHON_VENV_DIR}/bin/activate \
		&& pip3 install -Ur requirements.txt

# customize!
install-dev: install-venv	## Install dependencies for development
	@ true \
		&& . ${PYTHON_VENV_DIR}/bin/activate \
		&& pip3 install -Ur requirements-dev.txt

# customize!
install-venv: install-base	## Install Linux packages and python (venv)
	@ true \
		&& sudo apt install -y patchelf ccache \
		&& sudo apt install -y zlib1g-dev libjpeg-dev libpq-dev git-lfs \
		&& test -d ${PYTHON_VENV_DIR} || python3 -m venv ${PYTHON_VENV_DIR} \
		&& . ${PYTHON_VENV_DIR}/bin/activate \
    	&& pip3 install --upgrade pip \
		&& pip3 install -Ur requirements.txt \
		&& test -d tmp/AutoGPTQ || git clone https://github.com/PanQiWei/AutoGPTQ.git tmp/AutoGPTQ \
		&& cd tmp/AutoGPTQ && BUILD_CUDA_EXT=1 pip3 install -vvv --no-build-isolation -e .

install-base:	## Install base Linux packages and python
	@ true \
		&& sudo apt update \
		&& sudo apt install -y python3 \
    	&& sudo apt install -y python3.10-venv python3-dev python3-setuptools \
    	&& sudo apt install -y --no-install-recommends build-essential gcc \
    	&& sudo apt install -y python3-pip \
		&& sudo apt clean

# customize!
clean:	## Delete virtual environment and temporal folders
	rm -rf __pycache__ logs offload tmp "${PYTHON_VENV_DIR}"

# customize!
run:	## Run the python application in virtual environment
	@ . ${PYTHON_VENV_DIR}/bin/activate \
		&& mkdir -p logs \
		&& python3 main.py
		
# - https://huggingface.co/docs/trl/main/en/index
# - https://huggingface.co/docs/trl/main/en/sft_trainer
# - https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
# - https://huggingface.co/docs/trl/main/en/clis
# train-original: ## Train 'original' model
#	. ${PYTHON_VENV_DIR}/bin/activate \
#		&& mkdir -p datasets \
#		&& python3 dataset_format.py \
#		&& trl sft \
#		--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
#		--dataset_name datasets/olafrv-trl \
#		--dataset_text_field "instruction" \
#		--output_dir models/olafrv/Llama-2-7b-chat-hf-trained

# https://huggingface.co/docs/transformers/main/en//quantization
# Only 4-bit models are supported, and we recommend deactivating 
# the ExLlama kernels if you’re finetuning a quantized model with PEFT.
# train-gptq: ## Train 'tlrsft' model
#	. ${PYTHON_VENV_DIR}/bin/activate \
#		&& mkdir -p datasets \
#		&& python3 dataset_format.py \
#		&& trl sft \
#		--model_name_or_path TheBloke/Llama-2-7b-Chat-GPTQ \
#		--dataset_name datasets/olafrv-trl \
#		--dataset_text_field "instruction" \
#		--output_dir models/olafrv/Llama-2-7b-Chat-GPTQ-trained

# https://huggingface.co/docs/autotrain/v0.8.21/llm_finetuning
# Llama2 is now supported by AutoTrain, but CUDA out of memory.
# autotrain llm --help
# autotrain setup --update-torch  # Only if using Google Collab
# autotrain setup
# autotrain-original: ## Train 'original' model
#	# HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 
#	cd models/olafrv \
#	&& autotrain llm --train --trainer sft \
#		--token ${HUGGINGFACE_TOKEN} \
#		--data_path ../../datasets/olafrv-autotrain \
#		--model meta-llama/Llama-2-7b-chat-hf \
#		--text_column text \
#		--project_name "Llama-2-7b-chat-hf-autotrain"
# > logs/training.log &
# tail -f logs/training.log