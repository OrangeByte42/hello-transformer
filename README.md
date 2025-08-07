<img src="assets/header_img.png" style="width: 100%; height: auto; margin-bottom: 30px;">

<h1 align="center" style="font-weight: bold;">hello-transformer 👋🤖</h1>

<p align="center">
    <a href="#-getting-started">🚀 Getting Started</a> -
    <a href="#-usage">🧑‍💻 Usage</a> -
    <a href="./CHANGELOG.md">📙 Changelog</a> -
    <a href="#-maintainers">👥 Maintainers</a> -
    <a href="#-contributing">🤝 Contributing</a> -
    <a href="#-license">📄 License</a>
</p>

<p align="center">
    <!-- Project maintenance status -->
    <img src="https://img.shields.io/badge/Build-Passed-gre">
    <img src="https://img.shields.io/badge/Unit Test-No-red">
    <img src="https://img.shields.io/badge/Maintained-Yes-gre">
    <img src="https://img.shields.io/badge/Latest Release-No-red">
    </br>
    <!-- Project development environment -->
    <img src="https://img.shields.io/badge/Ubuntu-20.04.6 LTS-E95420?logo=ubuntu">
    <img src="https://img.shields.io/badge/Intel-Xeon Silver 4210R-0071C5?logo=intel">
    <img src="https://img.shields.io/badge/Nvidia-4 x RTX 3070 8GB-76B900?logo=nvidia">
    </br>
    <img src="https://img.shields.io/badge/Python-3.9.23-3776AB?logo=python">
    <img src="https://img.shields.io/badge/PyTorch-2.7.1+cu128-EE4C2C?logo=pytorch">
    <img src="https://img.shields.io/badge/Transformers-4.54.1-FFD21E?logo=huggingface">
    <img src="https://img.shields.io/badge/SpaCy-3.7.0-09A3D5?logo=spacy">
    </br>
    <!-- Documentation -->
    <img src="https://img.shields.io/badge/License-GPL3-238636">
    <img src="https://img.shields.io/badge/README Style-Standard-gre">
    <img src="https://img.shields.io/badge/CHANGELOG-Keep A Changelog-f25d30">
</p>


## 📦 About The Project

> 📃 Original Paper: [《Attention Is All You Need》](https://arxiv.org/abs/1706.03762)

### Introduction

This project reproduces **Basic Transformer Architecture** in 《Attention Is All You Need》. Train transformer model in Multi30k dataset (DE2EN), and archieved a maximum BLEU score of 27-28.
- *Tokenizer*: BERT / SpaCy (configurable);
- *Training Mode*: DDP / Non-DDP (configurable);
- *Training Method*: Teacher Forcing;
- *Evaluate Method*: Auto-regressive;
- *Evaluate Metric*: Corpus BLEU Score;

### Training Trace (Loss/PPL/BLEU)

TBD

### Prediction Trace

TBD


## 🚀 Getting Started

### Device Requirements

- At least one Nvidia GPU (>= 8GB GPU Memory);

### Install Dependencies

Create a Python 3.9.23 environment (use venv or conda), and then install dependencies:
- Install PyTorch: https://pytorch.org/
- Install SpaCy tokenizers:
    ```bash
    pip index versions spacy
    python install spacy==3.7.0

    python -m spacy download de_core_news_sm
    python -m spacy download en_core_web_sm
    ```
- Install other dependencies:
    ```bash
    pip install -r requirements-py3.9.23.txt
    ```

### Check If Runnable

```bash
bash ./scripts/test_runnable.sh
```


## 🧑‍💻 Usage

### Pre-Training

Modify `./src/configs.py` or `./src/train.py` for your data / model / training / save configurations.

Then, train model:
- **Single-node single-GPU (no DDP)**:
    ```bash
    python -u -m src.train
    ```
- **Single-node single-GPU Training (DDP)**:
    ```bash
    torchrun --nnodes=1 --node_rank=0 --nproc_per_node=4 -m src.train
    ```

Also, you can use scripts to train transformer:
```bash
bash ./scripts/train.sh
```

### Inferencing

TBD


## 👥 Maintainers

[@OrangeByte42](https://github.com/OrangeByte42).


## 🤝 Contributing

<a href="https://github.com/OrangeByte42/photo-archiver/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OrangeByte42/photo-archiver" alt="contrib.rocks image" />
</a>


## 📄 License

[GPL3](./LICENSE) © OrangeByte42


## 🙏 Acknowledgments

Thanks to:
- [hyunwoongko/transformer](https://github.com/hyunwoongko/transformer)

