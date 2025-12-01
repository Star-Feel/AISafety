from huggingface_hub import hf_hub_download
import os


# 下载指定文件（如pytorch_model.bin）
hf_hub_download(
    repo_id="Manojb/stable-diffusion-2-base",
    local_dir="./pretrained/sd2",
    resume_download=True,  # 关键：开启断点续传
)