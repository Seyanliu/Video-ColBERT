# VideoColBERT for Text-Video Retrieval

This project implements VideoColBERT, a model for efficient and effective text-video retrieval, based on the ColBERT late-interaction architecture. The codebase is adapted from the official implementation of "Disentangled Representation Learning for Text-Video Retrieval" (DRL).

## Introduction

VideoColBERT adapts the ColBERT strategy for the video domain. It computes fine-grained similarity between each token of a text query and all frames of a video, enabling a more accurate and robust retrieval by capturing nuanced interactions. This late-interaction mechanism is both powerful and efficient for large-scale retrieval tasks.

This implementation is built on PyTorch and leverages pre-trained CLIP models for feature extraction.

**Original Paper:**  
[Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval](https://arxiv.org/pdf/2503.19009v1)

## Features

-   **VideoColBERT Model**: An implementation of the ColBERT architecture for end-to-end text-video retrieval.
-   **CLIP-based Encoders**: Uses powerful pre-trained ViT models from CLIP for encoding text and video frames.
-   **Distributed Training**: Supports multi-GPU training using `torch.distributed`.
-   **Standard Datasets**: Includes dataloaders for common text-video retrieval benchmarks like MSR-VTT.
-   **Modular Design**: The code is organized into modules for data loading, modeling, and utilities, making it easy to extend.

## Requirements

The project is built with Python 3 and PyTorch. Key dependencies include:

-   `torch`
-   `decord` (for efficient video loading)
-   `numpy`
-   `tqdm`
-   `opencv-python`

You can install all the required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1.  Download the MSR-VTT dataset (or any other dataset you wish to use).
2.  Update the `anno_path` and `video_path` arguments in the training script to point to your dataset's annotations and video files.

### Training

The `train.sh` script provides an example of how to launch a distributed training job.

```bash
#!/bin/bash

# Example training script for MSR-VTT
python -m torch.distributed.launch --nproc_per_node=4 main.py \
  --do_train 1 \
  --workers 8 \
  --batch_size 128 \
  --batch_size_val 128 \
  --epochs 5 \
  --lr 1e-4 \
  --max_words 32 \
  --max_frames 12 \
  --video_framerate 1 \
  --output_dir ckpts/msrvtt_videocolbert \
  --base_encoder "ViT-B/32" \
  --datatype "msrvtt" \
  --anno_path "/path/to/your/msrvtt/annotations" \
  --video_path "/path/to/your/msrvtt/videos"
```

To start training, modify the paths and hyperparameters in `train.sh` as needed and run:

```bash
bash train.sh
```

### Evaluation

To evaluate a trained model, set `--do_train 0`, `--do_eval 1`, and provide the path to your checkpoint with the `--init_model` argument.

```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py \
  --do_eval 1 \
  --workers 8 \
  --batch_size_val 128 \
  --max_words 32 \
  --max_frames 12 \
  --output_dir ckpts/msrvtt_videocolbert \
  --base_encoder "ViT-B/32" \
  --datatype "msrvtt" \
  --anno_path "/path/to/your/msrvtt/annotations" \
  --video_path "/path/to/your/msrvtt/videos" \
  --init_model "path/to/your/pytorch_model.bin"
```

## Code Structure

-   `main.py`: The main script for training and evaluation.
-   `train.sh`: Example script for launching a training job.
-   `requirements.txt`: A list of dependencies.
-   `tvr/`: Main source code directory.
    -   `dataloaders/`: Contains data loading and preprocessing logic for different datasets.
    -   `models/`: Defines the model architecture (`VideoColBERT`), loss functions, and optimization components.
    -   `utils/`: Includes utility functions for metrics, logging, and distributed communication.

## Acknowledgements

This code is built upon the foundation of the [DRL for Text-Video Retrieval](https://github.com/foolwood/DRL) project. We thank the original authors for their contribution to the community.
