"""Example of using resemble-denoise-onnx-inference package."""

import librosa
import onnxruntime
from resemble_denoise import run, get_model_path
import scipy.io.wavfile


def denoise_audio(input_path: str, output_path: str):
    """
    Denoise audio file using Resemble AI model.

    Args:
        input_path: Path to input audio file
        output_path: Path to save denoised audio
    """
    # Load audio
    wav, sr = librosa.load(input_path, mono=True)

    # Setup ONNX session
    opts = onnxruntime.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    opts.log_severity_level = 4

    # Check for CUDA availability
    available_providers = onnxruntime.get_available_providers()
    providers = []

    if "CUDAExecutionProvider" in available_providers:
        cuda_provider_options = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB limit
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }
        providers.append(("CUDAExecutionProvider", cuda_provider_options))
        print("Using CUDA for GPU acceleration")
    else:
        print("CUDA provider not available, using CPU")

    providers.append("CPUExecutionProvider")

    # Get model path from package
    model_path = get_model_path()

    # Create ONNX session
    session = onnxruntime.InferenceSession(
        str(model_path),
        providers=providers,
        sess_options=opts,
    )

    # Run denoising
    denoised_wav, new_sr = run(session, wav, sr, batch_process_chunks=False)

    # Save result
    scipy.io.wavfile.write(output_path, new_sr, denoised_wav)
    print(f"Denoised audio saved to {output_path}")


if __name__ == "__main__":
    denoise_audio("test_audio.wav", "denoised_output.wav")
