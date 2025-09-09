import time
from .denoiser import run, get_model_path
import onnxruntime
import librosa
import scipy

path = "./test_audio.wav"

wav, sr = librosa.load(path, mono=True)

opts = onnxruntime.SessionOptions()
opts.inter_op_num_threads = 4
opts.intra_op_num_threads = 4
opts.log_severity_level = 4

# Проверяем доступность CUDA провайдера
available_providers = onnxruntime.get_available_providers()
print(f"Доступные провайдеры ONNX Runtime: {available_providers}")

# Настраиваем провайдеры в порядке приоритета
providers = []
if "CUDAExecutionProvider" in available_providers:
    # Настройки для CUDA провайдера
    cuda_provider_options = {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB лимит
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }
    providers.append(("CUDAExecutionProvider", cuda_provider_options))
    print("✅ Используем CUDA для ускорения на GPU")
else:
    print("⚠️  CUDA провайдер недоступен, используем CPU")

# Добавляем CPU как резервный вариант
providers.append("CPUExecutionProvider")

# Получаем путь к модели через функцию пакета
model_path = get_model_path()

session = onnxruntime.InferenceSession(
    str(model_path),
    providers=providers,
    sess_options=opts,
)

start = time.time()
wav_onnx, new_sr = run(session, wav, sr, batch_process_chunks=False)
print(f"Ran in {time.time() - start}s")

scipy.io.wavfile.write("denoiser_output.wav", new_sr, wav_onnx)
