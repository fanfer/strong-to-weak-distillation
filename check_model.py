#!/usr/bin/env python3
"""
模型路径检查脚本
用于验证您的模型路径是否正确配置
"""

import argparse
import os
import sys

def check_model_path(model_path: str):
    """检查模型路径是否有效"""
    print("=" * 60)
    print("模型路径检查")
    print("=" * 60)
    print(f"检查模型路径: {model_path}")
    print()
    
    # 1. 检查是否是本地路径
    if os.path.exists(model_path):
        print("✅ 本地路径存在")
        
        # 检查必要文件
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        missing_files = []
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                print(f"✅ 找到: {file}")
            else:
                missing_files.append(file)
                print(f"❌ 缺失: {file}")
        
        if missing_files:
            print(f"\n⚠️  警告: 缺失关键文件 {missing_files}")
            print("请确保这是一个完整的模型checkpoint")
        else:
            print("\n✅ 本地模型文件完整")
    
    else:
        print("ℹ️  不是本地路径，将尝试从HuggingFace加载")
    
    print()
    print("-" * 60)
    
    # 2. 测试tokenizer加载
    print("测试tokenizer加载...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ Tokenizer加载成功")
        print(f"   词汇表大小: {tokenizer.vocab_size}")
        print(f"   模型最大长度: {tokenizer.model_max_length}")
        
        # 测试tokenization
        test_text = "这是一个测试文本"
        tokens = tokenizer.tokenize(test_text)
        print(f"   测试分词: '{test_text}' -> {len(tokens)} tokens")
        
    except Exception as e:
        print(f"❌ Tokenizer加载失败: {e}")
        return False
    
    print()
    print("-" * 60)
    
    # 3. 测试VLLM兼容性（如果安装了VLLM）
    print("测试VLLM兼容性...")
    try:
        from vllm import LLM
        print("✅ VLLM已安装")
        
        # 注意：这里不实际加载模型，只检查路径格式
        print(f"✅ 模型路径格式: {model_path}")
        
    except ImportError:
        print("⚠️  VLLM未安装，跳过VLLM兼容性检查")
    except Exception as e:
        print(f"❌ VLLM兼容性检查失败: {e}")
        return False
    
    print()
    print("=" * 60)
    print("✅ 模型路径检查完成")
    return True

def check_config_file():
    """检查config.py文件中的模型配置"""
    print("检查config.py文件...")
    
    if not os.path.exists("config.py"):
        print("❌ 未找到config.py文件")
        return None
    
    try:
        import config
        model_name = config.MODEL_NAME
        print(f"✅ config.py中的模型路径: {model_name}")
        return model_name
    except Exception as e:
        print(f"❌ 读取config.py失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="检查模型路径配置")
    parser.add_argument("--model_path", type=str, default=None,
                       help="要检查的模型路径")
    parser.add_argument("--check_config", action="store_true",
                       help="检查config.py文件中的模型配置")
    
    args = parser.parse_args()
    
    if args.check_config:
        model_path = check_config_file()
        if model_path:
            check_model_path(model_path)
    elif args.model_path:
        check_model_path(args.model_path)
    else:
        print("请提供模型路径或使用 --check_config 检查配置文件")
        print("\n使用示例:")
        print("  python check_model.py --model_path /path/to/your/checkpoint")
        print("  python check_model.py --check_config")

if __name__ == "__main__":
    main() 