# DCW ComfyUI Wrapper

A ComfyUI node implementation of the **Dynamic Consistency Weighting (DCW)** technique that enables high-quality image generation with drastically reduced inference steps.

## What is DCW?

DCW is a post-processing technique that improves the quality of diffusion model outputs by leveraging wavelet-domain analysis. It allows you to generate images in **as few as 8 steps** without requiring model distillation or retraining.

### Key Features
- **Ultra-fast generation**: Generate images in 8 steps instead of the typical 20-50
- **No distillation needed**: Works with standard full models (tested with Anima, and others)
- **Wavelet-based refinement**: Uses discrete wavelet transform (DWT) to intelligently blend low and high-frequency components
- **Flexible control**: Adjustable strength parameter to balance generation speed vs quality
- **Multi-format support**: Works with both image (4D) and video (5D) latents

## Quick Start

### Installation

1. Clone this repository into your ComfyUI `custom_nodes` folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DawnW0lf/ComfyUI-DCW-Diffusion-Color-Wavelets-Node.git
```

2. Install dependencies:
```bash
cd comfyui-dcw
pip install -r requirements.txt
```

3. Restart ComfyUI

### Usage

1. Load your favorite model (e.g., Anima)
2. Add the **"Apply DCW (Wavelet Patch)"** node to your workflow
3. Connect your model to the node's input
4. Set parameters:
   - **Strength**: `0.3` (recommended for 8-step generation)
   - **Wavelet**: `haar` (default, good balance of speed and quality)
5. Use the patched model with very low step counts (8-15 steps)

### Recommended Settings

For **Anima** and similar models:
```
Steps: 8
Sampler: DPM++ 2M
Strength: 0.3
Wavelet: haar
CFG Scale: 4.0
```

## How It Works

DCW applies a wavelet-domain correction to the diffusion model's predictions:

1. **Decompose**: Split the input and output into wavelet components (low and high frequency)
2. **Correct**: Blend the low-frequency components based on the current timestep
3. **Reconstruct**: Inverse wavelet transform to get the corrected output
4. **Scale**: Apply adaptive scaling based on the diffusion timestep

The technique is particularly effective because it:
- Preserves high-frequency details (textures, edges) from the model's natural output
- Intelligently refines low-frequency components (color, composition) using temporal information
- Adapts the correction strength based on where you are in the diffusion process

## Parameters

- **Model**: The diffusion model to patch (required)
- **Strength**: Controls the intensity of DCW correction (0.0-4.0)
  - `0.0`: No correction, standard generation
  - `0.3`: Recommended for ultra-fast 8-step generation
  - `1.0`: Medium correction
  - `2.0+`: Heavy correction (may introduce artifacts at very low steps)
- **Wavelet**: The wavelet basis to use
  - `haar`: Fastest, good general-purpose choice
  - `db2`: Finer detail preservation
  - `db3`: Maximum detail (slightly slower)

### Latent Shape Handling
- **Images**: `[batch, channels, height, width]` (4D)
- **Videos**: `[batch, channels, frames, height, width]` (5D)

The implementation automatically detects and handles both formats.

## Credits

This implementation is based on the research from:

**Original Paper & Code**: [AMAP-ML/DCW](https://github.com/AMAP-ML/DCW)
- Licensed under CC-BY-NC-SA 4.0

**ComfyUI Wrapper**: Independent implementation for ComfyUI integration

The DCW technique was developed by the AMAP research team at NVIDIA.

## License

This project is licensed under **CC-BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International).

This work is based on the DCW technique from [AMAP-ML/DCW](https://github.com/AMAP-ML/DCW), which is also licensed under CC-BY-NC-SA 4.0.

### What this means:
- ✅ You can use, modify, and distribute this code
- ✅ You must attribute the original DCW work
- ❌ You cannot use this for commercial purposes without permission from NVIDIA/AMAP-ML
- ✅ Any modifications must also be shared under CC-BY-NC-SA 4.0

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest improvements
- Submit pull requests for enhancements

## Citation

If you use this in research, please cite the original DCW paper:

```bibtex
@article{yu2026eluci,
  title={Elucidating the SNR-t Bias of Diffusion Probabilistic Models},
  author={Meng Yu and Lei Sun and Jianhao Zeng and Xiangxiang Chu and Kun Zhan},
  journal={arXiv preprint arXiv:2604.16044},
  year={2026}
}
```

## Acknowledgments

- [AMAP-ML](https://github.com/AMAP-ML) and NVIDIA for the DCW technique
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for the amazing node framework
- [PyTorch Wavelets](https://github.com/fbcorlib/pytorch_wavelets) for wavelet implementations
- [Claude.ai](https://claude.ai) for helping me write this
---
