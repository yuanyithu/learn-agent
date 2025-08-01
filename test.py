# 创建测试脚本 verify_install.py
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-7B-Instruct"  # 示例模型（实际替换为你要部署的Qwen3模型）

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto"
    )
    print("✅ 环境配置成功！")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
except Exception as e:
    print(f"❌ 安装失败: {str(e)}")
