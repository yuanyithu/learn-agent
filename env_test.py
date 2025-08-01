import torch
import transformers

print("--- 环境诊断开始 ---")

# 1. 检查 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 2. 检查 Transformers 版本
print(f"Transformers 版本: {transformers.__version__}")

# 3. 核心测试：检查GPU是否可用
is_cuda_available = torch.cuda.is_available()
print(f"\nGPU 是否可用: {is_cuda_available}")

if is_cuda_available:
    # 如果GPU可用，打印更多信息
    gpu_count = torch.cuda.device_count()
    current_gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu_index)
    
    print(f"检测到 {gpu_count} 个GPU。")
    print(f"当前使用的GPU索引: {current_gpu_index}")
    print(f"当前GPU名称: {gpu_name}")
    
    # 简单的张量运算测试，确认GPU可以工作
    try:
        x = torch.tensor([1.0, 2.0, 3.0]).to("cuda")
        print(f"成功将张量移动到GPU: {x}")
        print("GPU工作正常！")
    except Exception as e:
        print(f"GPU运算时出错: {e}")

else:
    print("未检测到可用的NVIDIA GPU。模型将会在CPU上运行。")
    print("（注意：在CPU上运行大模型会非常慢。）")

print("\n--- 环境诊断结束 ---")