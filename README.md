Resemble Denoiser in ONNX
=======================

Denoiser from
https://github.com/resemble-ai/resemble-enhance

Resemble released a nice open speech enhancer model. It is in 2 parts, denoiser and an enhancer. The enhancer is much larger model and the quality is not much better (and sometimes it adds weird artifacts).

All PyTorch dependencies are removed. Allows for leaner distribution.


## Acknowledgments

Special thanks to:
- **Resemble AI** for releasing the original [resemble-enhance](https://github.com/resemble-ai/resemble-enhance) model
- **[skeskinen](https://github.com/skeskinen)** for the original ONNX conversion and implementation at [resemble-denoise-onnx-inference](https://github.com/skeskinen/resemble-denoise-onnx-inference)


Installation
===
This project has been tested on Python 3.13.

### Install from Git

**CPU version:**
```bash
uv add git+https://github.com/Brakanier/resemble-denoise-onnx-inference.git
# or with pip
pip install git+https://github.com/Brakanier/resemble-denoise-onnx-inference.git
```

**GPU version:**

For CUDA setup: [CUDA and cuDNN Setup for WSL2](WSL.md)

```bash
uv add git+https://github.com/Brakanier/resemble-denoise-onnx-inference.git[gpu]
# or with pip
pip install git+https://github.com/Brakanier/resemble-denoise-onnx-inference.git[gpu]
```

### Development Installation

```bash
git clone https://github.com/Brakanier/resemble-denoise-onnx-inference.git
cd resemble-denoise-onnx-inference
uv add .  # for CPU version
uv add .[gpu]  # for GPU version
```



Usage
===

### Programmatic Usage

```python
import librosa
import onnxruntime
from resemble_denoise import run, get_model_path
import scipy.io.wavfile

# Load audio
wav, sr = librosa.load("input_audio.wav", mono=True)

# Setup ONNX session
opts = onnxruntime.SessionOptions()
opts.inter_op_num_threads = 4
opts.intra_op_num_threads = 4

# Check for CUDA and setup providers
available_providers = onnxruntime.get_available_providers()
providers = []

if "CUDAExecutionProvider" in available_providers:
    providers.append("CUDAExecutionProvider")
    print("Using GPU acceleration")

providers.append("CPUExecutionProvider")

# Get model path from package
model_path = get_model_path()

# Create session and run denoising
session = onnxruntime.InferenceSession(str(model_path), providers=providers, sess_options=opts)
denoised_wav, new_sr = run(session, wav, sr)

# Save result
scipy.io.wavfile.write("denoised_output.wav", new_sr, denoised_wav)
```

### Script Usage

The `run.py` script is designed to denoise an audio file:

1. Place your test file as `test_audio.wav` or update the path in the script
2. Run: `python -m resemble_denoise.run`

The script will process the audio file and save the output to `denoiser_output.wav`.

