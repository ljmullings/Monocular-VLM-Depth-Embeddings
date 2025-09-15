# Depth-Enhanced VLM (working title: DistVLM)

A research framework for distance-augmented Vision–Language Models

## Overview

Vision–Language Models (VLMs) excel at semantic reasoning but lack explicit geometric grounding. This repository explores a simple but novel idea:

**Attach approximate per-object distance vectors to VLM embeddings.**

We build on [NVIDIA VILA-HD-8B-PS3-1.5K-SigLIP2](https://github.com/Efficient-Large-Model/VILA), adding a distance head that estimates how far each object is from the camera. These distance tokens are fused with the vision embeddings, enabling downstream reasoning that considers both semantics and geometry.

## Features

- Wrapper around VILA-HD 1.5K (high-resolution VLM).
- Integration with monocular depth estimators (MiDaS / ZoeDepth).
- Object-level embedding augmentation: `[vision embedding ⊕ distance token]`.
- Scripts for inference, adapter/LoRA training, and embedding visualization.
- Example configs for PS3 patch budgets (`NUM_LOOK_CLOSE`, `NUM_TOKEN_LOOK_CLOSE`).

## Installation

```bash
git clone 
cd depth-augmented-vlm

# create env
conda create -n distvlm python=3.10 -y
conda activate distvlm

# install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage

### Run inference

```bash
# set PS3 environment variables
export NUM_LOOK_CLOSE=2
export NUM_TOKEN_LOOK_CLOSE=2048
export SELECT_NUM_EACH_SCALE=256+512

# run a sample
python scripts/run_infer.py --config configs/base.yaml \
    --image assets/sample.jpg \
    --text "How far is the red car?"
```

### Train a distance head

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train_head.py --config configs/base.yaml
```

### Visualize embeddings

```bash
python scripts/viz_embeddings.py --run runs/exp01
```

## 📂 Repo Structure

```
src/
  vila_adapter/     # wrapper for VILA-HD calls
  depth/            # depth estimator integration
  head/             # distance MLP / LoRA head
  pipeline.py       # glue code
configs/
  base.yaml
scripts/
  run_infer.py
  train_head.py
  viz_embeddings.py
assets/
  sample.jpg
```

## 🧪 Research Context

This repository explores an under-studied space between multimodal semantics and geometric reasoning.

- **VLM research** → focuses on semantic alignment (objects, captions).
- **Depth research** → focuses on metric geometry (monocular/stereo).
- **This project** → bridges the two with distance-augmented embeddings.

If you use this work, please cite it as:

```bibtex
@misc{mullings2025distvlm,
  title={Distance-Augmented Vision-Language Models},
  author={Mullings, Laura},
  year={2025},
  howpublished={\url{https://github.com/yourusername/depth-augmented-vlm}},
  note={Independent Research}
}
```

## 📜 License

This repository is licensed under [CC-BY-NC-SA-4.0](LICENSE).
It builds on [NVIDIA VILA-HD](https://github.com/Efficient-Large-Model/VILA), also licensed under CC-BY-NC-SA-4.0.
See [LICENSE](LICENSE) and `NOTICE.md` for details.

## 🙏 Acknowledgements

- [NVIDIA VILA-HD](https://github.com/Efficient-Large-Model/VILA) for the base VLM.
- [MiDaS](https://github.com/isl-org/MiDaS), [ZoeDepth](https://github.com/isl-org/ZoeDepth) for monocular depth estimation.
- [KITTI-DEPTH] (https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) for benchmarking dataset.
  - @inproceedings{Uhrig2017THREEDV,
  author = {Jonas Uhrig and Nick Schneider and Lukas Schneider and Uwe Franke and Thomas Brox and Andreas Geiger},
  title = {Sparsity Invariant CNNs},
  booktitle = {International Conference on 3D Vision (3DV)},
  year = {2017} 
- Independent research project maintained by Laura Mullings.
