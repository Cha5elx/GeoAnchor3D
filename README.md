# GeoAnchor3D

**GeoAnchor3D: Hierarchical Spatial Regulation for Task-Adaptive Multimodal Scene Understanding**

**GeoAnchor3D：面向任务自适应多模态场景理解的层次化空间调控**

---

*English | [中文](#chinese)*

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

| Task       | Benchmark    | Metric              | Chat-Scene (Baseline) | GeoAnchor3D (Ours) |
|------------|-------------|---------------------|----------------------|--------------------|
| Grounding  | ScanRefer    | Acc@0.25 / Acc@0.5  | 55.5 / 50.2         | **56.5 / 51.1**    |
| Grounding  | Multi3DRefer | F1@0.25 / F1@0.5    | 57.1 / 52.4         | **59.2 / 54.6**    |
| QA         | ScanQA       | CIDEr               | 87.7                | **89.9**           |
| Captioning | Scan2Cap     | CIDEr@0.5           | 77.1                | **77.2**           |

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

---

## <span id="chinese">中文</span>

GeoAnchor3D 是一个统一的面向 3D 场景理解的多模态大语言模型（MLLM），基于 [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene) 构建，支持跨模态定位、稠密描述生成和问答任务。

### 研究动机

当前面向 3D 场景理解的 MLLM 面临两个关键挑战：

1. **模态交互不灵活。** 静态融合策略以固定权重组合几何与语义表征，难以适应不同任务对空间信息的差异化需求。
2. **深层模态退化。** 在 LLM 深层，几何线索逐渐被占主导地位的语义特征所抑制，削弱了空间推理能力。

### 方法

GeoAnchor3D 通过**层次化空间调控**解决上述问题，包含两个互补模块：

- **IGGA（指令感知几何门控注意力）。** 在输入端根据指令语义进行逐头门控，动态路由几何信息——对定位任务激活空间上下文，对语义任务则抑制空间信息注入，避免干扰。
- **GATH（几何感知辅助任务头）。** 在 LLM 中间层通过 3D 边界框坐标回归保留空间拓扑结构。仅训练时生效，推理阶段无额外计算开销。

### 实验结果

| 任务   | 数据集       | 指标                 | Chat-Scene (Baseline) | GeoAnchor3D (Ours) |
|--------|-------------|---------------------|----------------------|--------------------|
| 定位   | ScanRefer    | Acc@0.25 / Acc@0.5  | 55.5 / 50.2         | **56.5 / 51.1**    |
| 定位   | Multi3DRefer | F1@0.25 / F1@0.5    | 57.1 / 52.4         | **59.2 / 54.6**    |
| 问答   | ScanQA       | CIDEr               | 87.7                | **89.9**           |
| 描述   | Scan2Cap     | CIDEr@0.5           | 77.1                | **77.2**           |

GeoAnchor3D 在所有主要指标上均优于 Chat-Scene，尤其在几何密集型任务上提升显著（Multi3DRefer: +2.1 F1@0.25, +2.2 F1@0.5）。

### 模型架构

使用 Mask3D 从 3D 场景中提取物体候选区域，通过 Uni3D 编码 3D 几何特征，DINOv2 从多视图图像中提取 2D 外观特征，经 MLP 投影至 LLM 词元空间。IGGA 在 LLM 层之前动态融合空间掩码与语义注意力，GATH 在训练期间从中间层提供辅助几何监督。模型以 Vicuna-7B-v1.5 为 LLM 骨干，采用 LoRA 微调。

### 快速开始

**环境配置：**
```shell
conda create -n geoanchor3d python=3.9.17
conda activate geoanchor3d
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

**数据准备：** 参照 [preprocess/](preprocess/) 中的说明准备标注和提取特征。

**训练：** 修改 `scripts/run.sh` 后执行 `bash scripts/run.sh`。

**推理：** 在 `scripts/run.sh` 中设置 `evaluate=True` 和 `pretrained_path=/path/to/checkpoint.pth`，运行即可。

### 引用

```BibTeX
@inproceedings{zhang2025geoanchor3d,
  title={GeoAnchor3D: Hierarchical Spatial Regulation for Task-Adaptive Multimodal Scene Understanding},
  author={Zhang, Xue and Liu, Chenxu and Zou, Minghao and Hao, Xiaoshuai and Zhou, Wei and Zhao, Yao},
  booktitle={IEEE Transactions on Multimedia},
  year={2025}
}
```

### 致谢

本项目基于 [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene) 构建，感谢 LLaMA、Vicuna、Mask3D、Uni3D、DINOv2 以及 ScanNet 基准套件等开源项目的贡献。
