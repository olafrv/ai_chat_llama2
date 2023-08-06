#!/usr/bin/make

# Environment Variables
# - GITHUB_USER
# - GITHUB_TOKEN
# - PYTHON_VENV_DIR

NAME=$(shell cat METADATA | grep NAME | cut -d"=" -f2)
VERSION:=$(shell cat METADATA | grep VERSION | cut -d"=" -f2)
REPOSITORY="${NAME}"
IMAGE_NAME="ghcr.io/${GITHUB_USER}/${REPOSITORY}"
IMAGE_APP_DIR="/opt/${REPOSITORY}"
GITHUB_API="https://api.github.com/repos/${GITHUB_USER}/${REPOSITORY}"
GITHUB_API_JSON:=$(shell printf '{"tag_name": "%s","target_commitish": "main","name": "%s","body": "Version %s","draft": false,"prerelease": false}' ${VERSION} ${VERSION} ${VERSION})
CPUS=2
PYTHON_VENV_DIR?=./venv

# CAUTION: sensitive environment variables!
collect:
	$(foreach v, $(filter-out .VARIABLES,$(.VARIABLES)), $(info $(v) = $($(v))))
	python3 -m torch.utils.collect_env

install: install.venv
	@ . ${PYTHON_VENV_DIR}/bin/activate \
		&& pip3 install -Ur requirements.txt

# customize!
install.dev: install.venv
	@ true \
		&& sudo apt install -y patchelf ccache \
		&& . ${PYTHON_VENV_DIR}/bin/activate \
		&& pip3 install -Ur requirements-dev.txt

# customize!
install.venv: install.base
	@ true \
		&& sudo apt install -y zlib1g-dev libjpeg-dev libpq-dev git-lfs \
		&& test -d ${PYTHON_VENV_DIR} \
		|| python3 -m venv ${PYTHON_VENV_DIR} \
		&& . ${PYTHON_VENV_DIR}/bin/activate \
    	&& pip3 install --upgrade pip \
		&& pip3 install -Ur requirements.txt \
		&& CMAKE_ARGS='-DLLAMA_CUBLAS=on' NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-11' \
			FORCE_CMAKE=1 CXX=g++-11 CC=gcc-11 pip install llama-cpp-python --no-cache-dir \
		&& cd tmp \
		&& test -d AutoGPTQ || git clone https://github.com/PanQiWei/AutoGPTQ.git \
		&& cd AutoGPTQ && BUILD_CUDA_EXT=1 pip3 install .

install.base:
	@ sudo apt update \
		&& sudo apt install -y python3 \
    	&& sudo apt install -y python3.10-venv python3-dev python3-setuptools \
    	&& sudo apt install -y --no-install-recommends build-essential gcc \
    	&& sudo apt install -y python3-pip \
		&& sudo apt clean

uninstall: uninstall.venv clean

uninstall.venv: 
	#pip3 list --user --format=freeze | sed 's/=.*$//' | xargs pip3 uninstall --yes
	#@ test ! -d env \
	#	|| . ${PYTHON_VENV_DIR}/bin/activate \
	#	&& pip3 uninstall --yes -r requirements.txt \
	#	&& pip3 uninstall --yes -r requirements-dev.txt
	rm -rf ${PYTHON_VENV_DIR}

# customize!
clean:
	@ rm -rf build logs

# customize!
check-config:
	@ . ${PYTHON_VENV_DIR}/bin/activate \
		&& python3 main.py --check-config

build: install.dev
	# https://nuitka.net/doc/user-manual.html
	# https://nuitka.net/info/debian-dist-packages.html (Work in ubuntu!)
	# python3 -m nuitka --standalone --onefile --enable-plugin=numpy -o ${NAME}.bin main.py
	@ . ${PYTHON_VENV_DIR}/bin/activate \
		&& python3 -m nuitka --include-package=${NAME} --output-dir=./build \
			--show-progress --report=./build/main.py.xml -j6 \
			main.py
	@ chmod +x build/main.py.bin

# customize!
run:
	@ mkdir -p logs \
		&& . ${PYTHON_VENV_DIR}/bin/activate \
		&& python3 main.py

# fix me!
#
# It trains the Llama v2 HF model but the result is unusable on llama_prompter.py
#
# I mean here: tmp/trl/examples/scripts/sft_trainer.py even tried this:
#
# Step 7: Olaf trying to fix the config saving
# quantization_config.to_json_file(script_args.output_dir + "/quantize_config.json")
# model.config.save_pretrained(script_args.output_dir)
#
train:
	@ mkdir -p datasets \
		&& . ${PYTHON_VENV_DIR}/bin/activate \
		&& python3 llama_train.py \
		&& python3 tmp/trl/examples/scripts/sft_trainer.py \
    		--model_name meta-llama/Llama-2-7b-chat-hf \
    		--dataset_name datasets/olafrv \
			--output_dir models/olafrv/Llama-2-7b-chat-hf-trained \
    		--load_in_4bit \
    		--use_peft \
    		--batch_size 4 \
    		--gradient_accumulation_steps 2

run.bin:
	@ ./build/main.py.bin

test:
	# https://docs.pytest.org/
	@ . ${PYTHON_VENV_DIR}/bin/activate \
		&& pytest -s -s --disable-warnings ${NAME}/tests/

# customize!
test.coverage:
	# https://coverage.readthedocs.io
	@ . ${PYTHON_VENV_DIR}/bin/activate \
		&& coverage run main.py \
		&& coverage report --show-missing ${NAME}/*.py

# customize!
test.coverage.report:
	@ . ${PYTHON_VENV_DIR}/bin/activate \
		&& coverage run -m pytest -s --disable-warnings ${NAME}/tests/ \
		&& coverage report --show-missing ${NAME}/*.py

profile: install.dev
	# https://docs.python.org/3/library/profile.html
	@ mkdir -p profile \
		&& . ${PYTHON_VENV_DIR}/bin/activate \
		&& python3 -m cProfile -o profile/main.py.prof main.py

profile.view: install.dev
	@ . ${PYTHON_VENV_DIR}/bin/activate && snakeviz profile/main.py.prof

docker.build:
	@ docker build -t ${IMAGE_NAME}:${VERSION} .
	@ docker tag ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:latest 

docker.clean:
	@ docker images | grep ${IMAGE_NAME} | awk '{print $$1":"$$2}' | sort | xargs --no-run-if-empty -n1 docker image rm

# customize!
docker.run:
	@ docker run --rm --cpus ${CPUS} \
		-v "${PWD}/config.yaml:${IMAGE_APP_DIR}/config.yaml:ro" \
    	-v "${PWD}/logs:${IMAGE_APP_DIR}/logs" \
		${IMAGE_NAME}:${VERSION}

# customize!
docker.exec:
	@ docker run --rm -it --cpus ${CPUS} \
		-v "${PWD}/config.yaml:${IMAGE_APP_DIR}/config.yaml:ro" \
    	-v "${PWD}/logs:${IMAGE_APP_DIR}/logs" \
		--entrypoint /bin/bash ${IMAGE_NAME}:${VERSION}

github.push: docker.build
	# https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry
	# https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
	# https://docs.github.com/en/actions/security-guides/automatic-token-authentication#about-the-github_token-secret
	echo ${GITHUB_TOKEN} | docker login ghcr.io --username ${GITHUB_USER} --password-stdin
	@ docker push ${IMAGE_NAME}:${VERSION}
	@ docker push ${IMAGE_NAME}:latest

github.release: github.push
	# Fail if uncommited changes
	git diff --exit-code
	git diff --cached --exit-code
	# Create and push tag
	git tag -d ${VERSION} && git push --delete origin ${VERSION} || /bin/true
	git tag ${VERSION} && git push origin ${VERSION}
	# https://docs.github.com/rest/reference/repos#create-a-release
	@echo '${GITHUB_API_JSON}' | curl \
		-H 'Accept: application/vnd.github+json' \
		-H 'Authorization: token ${GITHUB_TOKEN}' \
		-d @- ${GITHUB_API}/releases
