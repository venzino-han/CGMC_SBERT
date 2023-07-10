default: build

help:
	@echo 'Management commands for cgmc:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the cgmc project.'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t cgmc 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=0"' --ipc=host --name cgmc -v `pwd`:/workspace/cgmc cgmc:latest /bin/bash

up: build run

rm: 
	@docker rm cgmc

stop:
	@docker stop cgmc

reset: stop rm

bert-build:
	@docker build -t bert -f bert_docker.yml .

bert-run:
	@echo "Booting up BERT Docker Container"
	@docker run -it --gpus '"device=1"' --ipc=host --name cgmc_bert -v `pwd`:/workspace/cgmc bert:latest /bin/bash
