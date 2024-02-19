# semantic-kernel.openvino

This experimental sample shows how to integrate Semantic-Kernel with OpenVINO

## Requirements

- Linux, Windows
- Python >= 3.9.0
- CPU or GPU compatible with OpenVINO.

## Install the requirements

    $ python3 -m venv openvino_env

    $ source openvino_env/bin/activate

    $ python3 -m pip install --upgrade pip
    
    $ pip install wheel setuptools
    
    $ pip install -r requirements.txt


## Export models

Export the Text-generation IR model from HuggingFace:

    $ optimum-cli export openvino --model gpt2 llm_model

see more [usage examples](https://github.com/huggingface/optimum-intel?tab=readme-ov-file#export)   

Export the Embedding IR model from HuggingFace:

    $ python3 export_embedding.py

## Launch notebook sample

    $ jupyter lab sample.ipynb
