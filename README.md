# GeoAnchor3D

**GeoAnchor3D: Hierarchical Spatial Regulation for Task-Adaptive Multimodal Scene Understanding**

GeoAnchor3D is a unified multimodal large language model (MLLM) for 3D scene understanding, built upon [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene). It supports cross-modal grounding, dense captioning, and question answering.

## Motivation

Current MLLMs for 3D scene understanding face two key challenges:

1. **Inflexible Modality Interaction.** Static fusion strategies combine geometric and semantic representations with fixed weights, failing to adapt across tasks with heterogeneous spatial demands.
2. **Deep-Layer Modality Degradation.** Geometric cues are progressively suppressed by dominant semantic features in deeper LLM layers, weakening spatial reasoning.

## Method

GeoAnchor3D addresses these challenges via **hierarchical spatial regulation** with two complementary modules:

- **IGGA (Instruction-Aware Geometric Gating Attention).** Performs per-head gating conditioned on instruction semantics at the input level, dynamically routing geometric information according to task demands — activating spatial context for grounding tasks while suppressing it for semantic-oriented tasks.
- **GATH (Geometry-Aware Auxiliary Task Head).** Preserves spatial topology in intermediate LLM hidden states through coordinate regression of 3D bounding boxes. Active only during training; introduces zero inference overhead.

## Results

| Task         | Benchmark      | Metric      | Chat-Scene (Baseline) | GeoAnchor3D (Ours) |
|-------------|---------------|-------------|----------------------|-------------------|
| Grounding    | ScanRefer      | Acc@0.25 / Acc@0.5 | 55.5 / 50.2         | **56.5 / 51.1**   |
| Grounding    | Multi3DRefer   | F1@0.25 / F1@0.5   | 57.1 / 52.4         | **59.2 / 54.6**   |
| QA           | ScanQA         | CIDEr               | 87.7                | **89.9**          |
| Captioning   | Scan2Cap       | CIDEr@0.5           | 77.1                | **77.2**          |

GeoAnchor3D consistently outperforms Chat-Scene across all primary metrics, with particularly notable gains on geometry-intensive tasks (Multi3DRefer: +2.1 F1@0.25, +2.2 F1@0.5).

## Architecture

Object proposals are extracted from 3D scenes via Mask3D. 3D geometric features (Uni3D) and 2D appearance features (DINOv2 from multi-view images) are projected into the LLM token space. IGGA dynamically fuses spatial masks with semantic attention before the LLM layers, while GATH provides auxiliary geometric supervision from intermediate layers during training. The model uses Vicuna-7B-v1.5 as the LLM backbone with LoRA fine-tuning.

## Getting Started

### Environment

```shell
conda create -n geoanchor3d python=3.9.17
conda activate geoanchor3d
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Data Preparation

Follow the instructions in [preprocess/](preprocess/) to prepare annotations and extracted features.

### Training

Modify and run:
```shell
bash scripts/run.sh
```

### Inference

Set `evaluate=True` and `pretrained_path=/path/to/checkpoint.pth` in `scripts/run.sh`, then run the same script.

## Citation

If you find this work useful, please consider citing:

```BibTeX
@inproceedings{zhang2025geoanchor3d,
  title={GeoAnchor3D: Hierarchical Spatial Regulation for Task-Adaptive Multimodal Scene Understanding},
  author={Zhang, Xue and Liu, Chenxu and Zou, Minghao and Hao, Xiaoshuai and Zhou, Wei and Zhao, Yao},
  booktitle={IEEE Transactions on Multimedia},
  year={2025}
}
```

## Acknowledgement

This project is built upon [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene). Thanks to the open-source contributions of LLaMA, Vicuna, Mask3D, Uni3D, DINOv2, and the ScanNet benchmark suite.
