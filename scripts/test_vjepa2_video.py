import torch
from tqdm import tqdm
from transformers import AutoModel, AutoVideoProcessor
from torchcodec.decoders import VideoDecoder
import numpy as np

# -----------------------------
# 配置
# -----------------------------
model_path = "/data3/lg2/human_wm/vjepa2/weights"
video_path = "/data3/lg2/human_wm/Ego4d/exo_data/takes/cmu_bike08_5/frame_aligned_videos/downscaled/448/cam01.mp4"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8  # 根据显存调整

# -----------------------------
# 加载模型和 processor
# -----------------------------
print("Loading model and processor...")
model = AutoModel.from_pretrained(model_path).to(device)
processor = AutoVideoProcessor.from_pretrained(model_path)

# -----------------------------
# 使用 TorchCodec 读取视频
# -----------------------------
print("Reading video frames with TorchCodec...")
decoder = VideoDecoder(video_path, device="cpu")  # 解码在 CPU 通常更高效
num_frames = decoder.metadata.num_frames
print(f"Total frames: {num_frames}")

# 读取所有帧: shape (T, C, H, W), dtype=torch.uint8
video_tensor = decoder[:]  # [num_frames, C, H, W]

# 转为 list of (H, W, C) numpy arrays (uint8, 0-255)
video_frames_np = video_tensor.permute(0, 2, 3, 1).numpy()  # (T, H, W, C)
video_list = [frame for frame in video_frames_np]  # 每帧是 (H, W, 3)

# -----------------------------
# 使用 processor 处理视频帧
# -----------------------------
print("Processing video with processor...")
inputs = processor(video_list, return_tensors="pt")

# 安全获取视频张量（兼容不同模型）
print("Available keys:", list(inputs.keys()))
if "pixel_values_videos" in inputs:
    video_tensor_processed = inputs["pixel_values_videos"].to(device)
elif "pixel_values" in inputs:
    video_tensor_processed = inputs["pixel_values"].to(device)
else:
    raise KeyError(f"Unsupported processor output keys: {list(inputs.keys())}")

print(f"Processed video tensor shape: {video_tensor_processed.shape}")  # 应为 (1, T, C, H, W)

# -----------------------------
# 模型推理（分批处理）
# -----------------------------
print("Running inference...")
num_frames_proc = video_tensor_processed.shape[1]
outputs_list = []

for i in tqdm(range(0, num_frames_proc, batch_size), desc="Inference"):
    batch = video_tensor_processed[:, i:i + batch_size]  # (1, B, C, H, W)
    with torch.no_grad():
        outputs = model(batch)
    outputs_list.append(outputs)

print("Inference done!")

# -----------------------------
# 保存特征到磁盘
# -----------------------------
print("\nSaving features to disk...")

# 合并所有 batch 的输出（假设模型返回的是 BaseModelOutput）
all_hidden_states = []
for out in outputs_list:
    # V-JEPA v2 的 AutoModel 返回的是 BaseModelOutput，包含 last_hidden_state
    if hasattr(out, 'last_hidden_state'):
        all_hidden_states.append(out.last_hidden_state.cpu())
    else:
        # 如果直接返回 tensor（某些模型可能如此）
        all_hidden_states.append(out.cpu())

# 拼接成一个完整的 tensor: shape (1, total_tokens, hidden_size)
# 注意：total_tokens = num_frames × num_patches_per_frame
final_features = torch.cat(all_hidden_states, dim=1)

# 推荐：生成一个有意义的文件名
import os
video_name = os.path.splitext(os.path.basename(video_path))[0]  # 例如 "cam01"
output_dir = "/data3/lg2/human_wm/vjepa2/features"  # 推荐的保存目录
os.makedirs(output_dir, exist_ok=True)
feature_path = os.path.join(output_dir, f"{video_name}_vjepa2_features.pt")

# 保存
torch.save(final_features, feature_path)
print(f"✅ Features saved to: {feature_path}")
print(f"   Feature shape: {final_features.shape}")  # 例如 [1, 479*768, 1408]（ViT-G hidden_size=1408）