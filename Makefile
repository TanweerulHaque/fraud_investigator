#################################################################################
# GLOBALS                                                                       #
#################################################################################

SHELL:=/bin/bash

# Load environment variables if an .env file exists
-include .env

#################################################################################
# Commands                                                                      #
#################################################################################

## Set up local development environment and install requirements
local-setup: create-environment requirements

## Set up python interpreter environment using venv
create-environment:
	@echo ">>> Creating virtual environment in ~/fidsenv"
	python3 -m venv ~/fidsenv
	@echo ">>> Run 'source ~/fidsenv/bin/activate' to use this environment"

## Install Python Dependencies
requirements:
	@echo ">>> Installing dependencies."
	pip install --upgrade pip
	pip install -r requirements.txt

## Run the demo case from data/cases/sample_case.json
demo:
	@echo ">>> Running the local investigation pipeline demo..."
	python3 main.py demo

## Seed long-term memory from data/seed_cases.json
seed:
	@echo ">>> Seeding historical memory database..."
	python3 main.py seed

## Run the evaluation harness on data/eval_cases.json
evaluate:
	@echo ">>> Running evaluation harness..."
	python3 main.py evaluate

## Run evaluation harness with verbose output
evaluate-verbose:
	@echo ">>> Running evaluation harness (verbose)..."
	python3 main.py evaluate --verbose

## Export LangGraph structure and save Mermaid/ASCII artifacts
graph:
	@echo ">>> Generating LangGraph structure artifacts..."
	python3 main.py graph

## Launch the Streamlit web dashboard dashboard
ui:
	@echo ">>> Starting Streamlit app dashboard..."
	streamlit run streamlit_app.py

#################################################################################
# Cleaning operations                                                           #
#################################################################################

## Remove Python file artifacts and caches
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .mypy_cache

## Remove generated data artifacts, evaluation reports, databases, and vector stores
clean-artifacts:
	@echo ">>> Removing local databases and execution artifacts..."
	rm -rf data/chroma
	rm -f data/fraud_twin.db
	rm -rf data/artifacts/*
	rm -f data/demo_sample_case.json
	rm -f data/eval_report.json

## Delete all temporary files and reset the local runtime data
clean: clean-pyc clean-artifacts
	@echo ">>> Project environment cleaned up."
