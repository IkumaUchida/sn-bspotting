.PHONY: black-check
black-check:
	poetry run black --check src tests

.PHONY: black
black:
	poetry run black src tests

.PHONY: flake8
flake8:
	poetry run flake8 src tests

.PHONY: isort-check
isort-check:
	poetry run isort --check-only src tests

.PHONY: isort
isort:
	poetry run isort src tests

.PHONY: mdformat
mdformat:
	poetry run mdformat *.md

.PHONY: mdformat-check
mdformat-check:
	poetry run mdformat --check *.md

.PHONY: mypy
mypy:
	poetry run mypy src

.PHONY: test
test:
	poetry run pytest tests --cov=src --cov-report term-missing --durations 5

.PHONY: format
format:
	$(MAKE) black
	$(MAKE) isort
	$(MAKE) mdformat

.PHONY: lint
lint:
	$(MAKE) black-check
	$(MAKE) isort-check
	$(MAKE) mdformat-check
	$(MAKE) flake8
	$(MAKE) mypy

.PHONY: test-all
test-all:
	$(MAKE) lint
	$(MAKE) test


.PHONY: docker-run-conda
docker-run-conda:

	docker run \
	--name bspotting_cuconda \
	--gpus all \
	-d \
	-u $(id -u):$(id -g) \
	--shm-size=516G \
	-it \
	-v $(PWD):/workspace \
	jptman/cuconda:v1 bash

	docker attach bspotting_cuconda


.PHONY: docker
docker:
	docker build -t ikumauchida/sn-bspotting:latest . 
	docker run -t ikumauchida/sn-bspotting:latest echo "ikumauchida/sn-bspotting done"
# docker run --platform linux/amd64 -t atomscott/soccertrack:latest echo "atomscott/soccertrack done"
# if cpu dont user --gpus all
# if m1 mac add --platform linux/amd64 before the image name

.PHONY: docker-cpu
docker-check-gpu:
	docker run --gpus all -t atomscott/soccertrack:latest  nvidia-smi


.PHONY: docker-push
docker-push:
	docker login
	docker push atomscott/soccertrack:latest

.PHONY: docker-run
docker-run:
	docker run --rm --gpus all -t -it -v $(PWD):/workspace ikumauchida/sn-bspotting:latest bash

#################################################################################
# Singularity                                                     #
#################################################################################
.PHONY: singularity-pull
singularity-pull:
	singularity pull docker://atomscott/soccertrack:latest