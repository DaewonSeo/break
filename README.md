### BREAK-Refactor

##### [Different]
  - .gitignore: data/politic/politic\_checkpoint.pt

##### [Requirements]

: torch, python version

  - python Version: 3.10.15
  - Torch CUDA available: True
  - CUDA version (compiled): 12.4
  - CUDA device: NVIDIA GeForce RTX 4060 Ti
  - CUDA device capability: (8, 9)

-----

### How to Run

모델을 실행하려면 터미널에 아래 명령어를 입력하세요. `--dataset_name` 인자로 'politic'과 같은 데이터셋 이름을 지정해야 합니다.

```bash
python main.py --dataset_name politic
```

-----

### Original Text

This is the official code of paper "Graph with Sequence: Broad-Range Semantic Modeling for Fake News Detection"

First, setting parameters on config.py (dataset name, data path, and so on)

Then, run main.py

Datasets we used are incorporated in '/data' direction (Only text content since images are too large, we uploaded images to Google Drive, the link is as follows：[https://drive.google.com/drive/folders/1fjX5B6BQoi5cETKkDRiRxIT6UQB5gbzU?usp=drive\_link](https://drive.google.com/drive/folders/1fjX5B6BQoi5cETKkDRiRxIT6UQB5gbzU?usp=drive_link))