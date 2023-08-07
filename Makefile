#!/usr/bin/make

NAME=$(shell cat METADATA | grep NAME | cut -d"=" -f2)
VERSION:=$(shell cat METADATA | grep VERSION | cut -d"=" -f2)
REPOSITORY="${NAME}"
IMAGE_NAME="ghcr.io/${GITHUB_USER}/${REPOSITORY}"
IMAGE_APP_DIR="/opt/${REPOSITORY}"
GITHUB_API="https://api.github.com/repos/${GITHUB_USER}/${REPOSITORY}"
GITHUB_API_JSON:=$(shell printf '{"tag_name": "%s","target_commitish": "main","name": "%s","body": "Version %s","draft": false,"prerelease": false}' ${VERSION} ${VERSION} ${VERSION})
CPUS=2
PYTHON_VENV_DIR?=./venv

.PHONY: help collect freeze install install-dev install-venv install-base uninstall uninstall-venv clean check-config build run train-original

help:
	@echo 'ENVIRONMENT: ## GITHUB_USER, GITHUB_TOKEN' | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'
	@echo ': ## PYTHON_VENV_DIR' | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'
	@grep -E '^[\.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'

collect:	## Collect and shows environment (sensitive values!)
	$(foreach v, $(filter-out .VARIABLES,$(.VARIABLES)), $(info $(v) = $($(v))))
	@ python3 -m torch.utils.collect_env

freeze:		## Save dpkg and pip freeze to freeze-*.txt
	dpkg-query --list > freeze-dpkg.txt
	pip3 freeze > freeze-pip.txt

install: install-venv	## Install dependencies for production
	@ . ${PYTHON_VENV_DIR}/bin/activate \
		&& pip3 install -Ur requirements.txt

# customize!
install-dev: install-venv	## Install dependencies for development
	@ true \
		&& sudo apt install -y patchelf ccache \
		&& . ${PYTHON_VENV_DIR}/bin/activate \
		&& pip3 install -Ur requirements-dev.txt

# customize!
install-venv: install-base	## Install base Linux packages and python
	@ true \
		&& sudo apt install -y zlib1g-dev libjpeg-dev libpq-dev git-lfs \
		&& test -d ${PYTHON_VENV_DIR} || python3 -m venv ${PYTHON_VENV_DIR} \
		&& . ${PYTHON_VENV_DIR}/bin/activate \
    	&& pip3 install --upgrade pip \
		&& pip3 install -Ur requirements.txt \
		&& CMAKE_ARGS='-DLLAMA_CUBLAS=on' NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-11' \
			FORCE_CMAKE=1 CXX=g++-11 CC=gcc-11 pip install llama-cpp-python --no-cache-dir \
		&& test -d tmp/AutoGPTQ || git clone https://github.com/PanQiWei/AutoGPTQ.git tmp/AutoGPTQ \
		&& cd tmp/AutoGPTQ && BUILD_CUDA_EXT=1 pip3 install .

install-base:	## Install base Linux packages and python
	@ true \
		&& sudo apt update \
		&& sudo apt install -y python3 \
    	&& sudo apt install -y python3.10-venv python3-dev python3-setuptools \
    	&& sudo apt install -y --no-install-recommends build-essential gcc \
    	&& sudo apt install -y python3-pip \
		&& sudo apt clean

uninstall: clean-venv clean	## Uninstall virtual environment and clean build files
	@ true

clean:	## Delete build/ and logs/ folders
	@ rm -rf build

# customize!
clean-venv:	## Delete virtual environment and its related folders
	rm -rf __pycache__ logs offload tmp "${PYTHON_VENV_DIR}"

# customize!
check-config:	## Check configuration
	@ . ${PYTHON_VENV_DIR}/bin/activate \
		&& python3 main.py --check-config

# customize!
# https://nuitka.net/doc/user-manual.html
# https://nuitka.net/info/debian-dist-packages.html (Works in Ubuntu!)
# python3 -m nuitka --standalone --onefile --enable-plugin=numpy -o ${NAME}.bin main.py
# python3 -m nuitka --include-package=${NAME} --output-dir=./build --show-progress --report=./build/main.py.xml -j6 main.py
build:	## Build binary (main.bin)
	@ . ${PYTHON_VENV_DIR}/bin/activate \
		&& mkdir -p build \
		&& python3 -m nuitka --output-dir=./build \
			--show-progress --report=./build/main.py.xml -j6 \
			main.py \
		&& chmod +x build/main.bin

# customize!
run:	## Run the python application in virtual environment
	@ . ${PYTHON_VENV_DIR}/bin/activate \
		&& mkdir -p logs \
		&& python3 main.py
		
# References:
# - https://huggingface.co/docs/trl/main/en/index
# - https://huggingface.co/docs/trl/main/en/sft_trainer
train-original: train-trl	## Train 'original' model (install TRL dependencies)
	. ${PYTHON_VENV_DIR}/bin/activate \
		&& mkdir -p datasets \
		&& python3 dataset_format.py \
		&& test -d tmp/trl || git clone https://github.com/lvwerra/trl tmp/trl \
		&& python3 tmp/trl/examples/scripts/sft_trainer.py \
		--model_name meta-llama/Llama-2-7b-chat-hf \
		--dataset_name datasets/olafrv \
		--output_dir models/olafrv/Llama-2-7b-chat-hf-trained \
		--load_in_4bit \
		--use_peft \
		--batch_size 4 \
		--gradient_accumulation_steps 2

# customize!
run-bin:	## Run compiled binary (main.bin)
	PYTHONPATH=. ./build/main.bin

# customize!
# https://docs.pytest.org/
test:	## Run tests (if any on tests/ directory)
	. ${PYTHON_VENV_DIR}/bin/activate \
		&& pytest -s -s --disable-warnings ${NAME}/tests/

# customize!
# https://coverage.readthedocs.io
coverage-code:	## Runs code's coverage and generates report HTML report
	. ${PYTHON_VENV_DIR}/bin/activate \
		&& coverage run main.py \
		&& coverage report --show-missing *.py

# customize!
# https://coverage.readthedocs.io
coverage-test:	## Runs test's coverage and generates HTML report
	. ${PYTHON_VENV_DIR}/bin/activate \
		&& coverage run -m pytest -s --disable-warnings ${NAME}/tests/ \
		&& coverage report --show-missing ${NAME}/*.py

# https://docs.python.org/3/library/profile.html
profile:	## Run profiler on main.py
	. ${PYTHON_VENV_DIR}/bin/activate \
		&& mkdir -p profile \
		&& python3 -m cProfile -o profile/main.py.prof main.py

profile-view:	## View profiler results with snakeviz
	. ${PYTHON_VENV_DIR}/bin/activate \
		&& snakeviz profile/main.py.prof

docker-build:	## Build docker image from Dockerfile
	test -d Dockerfile
	docker build -t ${IMAGE_NAME}:${VERSION} .
	docker tag ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:latest 

docker-clean:	## Delete docker image
	test -d Dockerfile
	docker images | grep ${IMAGE_NAME} | awk '{print $$1":"$$2}' | sort | xargs --no-run-if-empty -n1 docker image rm

# customize!
docker-run:		## Run docker image
	test -d Dockerfile
	docker run --rm --cpus ${CPUS} \
		-v "${PWD}/config.yaml:${IMAGE_APP_DIR}/config.yaml:ro" \
    	-v "${PWD}/logs:${IMAGE_APP_DIR}/logs" \
		${IMAGE_NAME}:${VERSION}

# customize!
docker-exec:	## Executes /bin/bash inside docker image
	test -d Dockerfile
	docker run --rm -it --cpus ${CPUS} \
		-v "${PWD}/config.yaml:${IMAGE_APP_DIR}/config.yaml:ro" \
    	-v "${PWD}/logs:${IMAGE_APP_DIR}/logs" \
		--entrypoint /bin/bash ${IMAGE_NAME}:${VERSION}

github-push: docker-build	## Push docker image to github
	# https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry
	# https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
	# https://docs.github.com/en/actions/security-guides/automatic-token-authentication#about-the-github_token-secret
	echo ${GITHUB_TOKEN} | docker login ghcr.io --username ${GITHUB_USER} --password-stdin
	docker push ${IMAGE_NAME}:${VERSION}
	docker push ${IMAGE_NAME}:latest

github-release: github-push	## Create github release
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
