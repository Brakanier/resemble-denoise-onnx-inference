# CUDA and cuDNN Setup for WSL2

This guide explains how to properly set up CUDA and cuDNN in WSL2 for GPU acceleration (specifically for ONNX GPU Runtime).

[NVIDIA guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

## Prerequisites

- Windows 10/11 with WSL2 enabled
- NVIDIA GPU
- WSL2 Ubuntu distribution

## Step 1: Install NVIDIA Drivers on Windows

1. Download and install the latest NVIDIA drivers for Windows from the [NVIDIA website](https://www.nvidia.com/drivers/)
2. Make sure to install drivers that support CUDA
3. Restart Windows after installation

The Windows NVIDIA drivers will automatically be accessible from WSL2 - no additional driver installation is needed in WSL.

## Step 2: Install CUDA Toolkit in WSL2

**Important**: Install `cuda-toolkit` specifically, NOT the regular `cuda` package. Installing the regular `cuda` package will overwrite the Windows CUDA installation and cause GPU detection issues in WSL.

Use:
[Download page](https://developer.nvidia.com/cuda-12-9-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)

or

```bash
# Add repo
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
# Install CUDA Toolkit (NOT cuda package)
sudo apt-get -y install cuda-toolkit-12-9
```

## Step 3: Set Up CUDA Environment Variables

Add CUDA to your library path by adding this line to your `~/.bashrc` or `~/.profile`:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Apply the changes:
```bash
source ~/.bashrc
```

## Step 4: Install cuDNN (Required for ONNX Runtime)

### Add NVIDIA Package Repository

[NVIDIA guide](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html#)

First, add the NVIDIA package repository:

How to add repo: [here](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html#ubuntu-and-debian-network-installation) and [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-ubuntu)

or

```bash
sudo apt-get install zlib1g

# Download and install the repository key
# ONLY for Ubuntu 2404/x86_64, for other distro see links
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb

dpkg -i cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update package list
sudo apt update
```

### Install cuDNN

```bash
sudo apt install cudnn9-cuda-12
```

**Note**: cuDNN installs to `/usr/lib/x86_64-linux-gnu`, not to the CUDA directory.

## Step 5: Add cuDNN to Library Path

Add cuDNN library path to your environment variables. Add this line to your `~/.bashrc` or `~/.profile`:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

Your complete library path should look like:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

Apply the changes:
```bash
source ~/.bashrc
```

## Verification

To verify your installation:

```bash
# Check CUDA version
nvcc --version

# Check if GPU is detected
nvidia-smi
```

If your **PATH** doesn`t contain **/usr/local/cuda/bin**, add this line to your ~/.bashrc or ~/.profile:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

Apply the changes:
```bash
source ~/.bashrc
```

## Troubleshooting

- If `nvidia-smi` doesn't work, ensure you have the latest Windows NVIDIA drivers installed
- If CUDA libraries are not found, double-check your `LD_LIBRARY_PATH` environment variable
- Make sure you installed `nvidia-cuda-toolkit` and not `cuda` package
- Restart your WSL session after making environment variable changes

## Final Environment Setup

Your `~/.bashrc` should include:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

This setup ensures that both CUDA and cuDNN libraries are properly accessible for applications like ONNX Runtime that require GPU acceleration.
