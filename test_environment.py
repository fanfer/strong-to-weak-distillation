#!/usr/bin/env python3
"""
环境测试脚本

用于验证 Qwen3 Strong-to-Weak Distillation 环境是否正确配置

用法:
    python test_environment.py
"""

import sys
import importlib
import logging
from typing import List, Tuple
import warnings

# 抑制一些不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_python_version() -> bool:
    """测试Python版本"""
    logger.info("=== 测试Python版本 ===")
    version = sys.version_info
    logger.info(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        logger.info("✓ Python版本符合要求 (>=3.8)")
        return True
    else:
        logger.error("✗ Python版本过低，需要Python 3.8+")
        return False

def test_package_imports() -> List[Tuple[str, bool, str]]:
    """测试关键包导入"""
    logger.info("=== 测试关键包导入 ===")
    
    packages_to_test = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"), 
        ("bitsandbytes", "BitsAndBytes"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("tqdm", "TQDM"),
        ("datasets", "Datasets"),
        ("tokenizers", "Tokenizers"),
    ]
    
    optional_packages = [
        ("wandb", "Wandb"),
        ("flash_attn", "Flash Attention"),
        ("xformers", "xFormers"),
    ]
    
    results = []
    
    # 测试必需包
    for package_name, display_name in packages_to_test:
        try:
            importlib.import_module(package_name)
            logger.info(f"✓ {display_name} 导入成功")
            results.append((display_name, True, "必需"))
        except ImportError as e:
            logger.error(f"✗ {display_name} 导入失败: {e}")
            results.append((display_name, False, "必需"))
    
    # 测试可选包
    for package_name, display_name in optional_packages:
        try:
            importlib.import_module(package_name)
            logger.info(f"✓ {display_name} 导入成功（可选）")
            results.append((display_name, True, "可选"))
        except ImportError:
            logger.warning(f"⚠ {display_name} 未安装（可选）")
            results.append((display_name, False, "可选"))
    
    return results

def test_torch_cuda() -> bool:
    """测试CUDA可用性"""
    logger.info("=== 测试CUDA环境 ===")
    
    try:
        import torch
        
        logger.info(f"PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"✓ CUDA可用，发现 {device_count} 个GPU")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory_mb = torch.cuda.get_device_properties(i).total_memory // 1024**2
                logger.info(f"  GPU {i}: {gpu_name} ({memory_mb} MB)")
            
            return True
        else:
            logger.warning("⚠ CUDA不可用，将使用CPU训练（速度较慢）")
            return False
            
    except Exception as e:
        logger.error(f"✗ 测试CUDA时出错: {e}")
        return False

def test_transformers_models() -> bool:
    """测试Transformers模型加载"""
    logger.info("=== 测试Transformers模型访问 ===")
    
    try:
        from transformers import AutoTokenizer
        
        # 测试能否访问HuggingFace Hub
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            logger.info("✓ 能够访问HuggingFace Hub")
            return True
        except Exception as e:
            logger.warning(f"⚠ 访问HuggingFace Hub时出现问题: {e}")
            logger.info("  这可能是网络问题，建议检查网络连接或使用镜像")
            return False
            
    except Exception as e:
        logger.error(f"✗ 测试Transformers时出错: {e}")
        return False

def test_data_loading() -> bool:
    """测试数据加载"""
    logger.info("=== 测试数据加载 ===")
    
    import os
    import json
    
    data_path = "dataset/train-00000-of-00001-cae87f8e074b4b5d.json"
    
    if not os.path.exists(data_path):
        logger.error(f"✗ 数据文件不存在: {data_path}")
        return False
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✓ 数据文件加载成功，包含 {len(data)} 个样本")
        
        # 检查数据格式
        if len(data) > 0:
            sample = data[0]
            required_keys = ['input', 'output', 'instruction']
            missing_keys = [key for key in required_keys if key not in sample]
            
            if missing_keys:
                logger.warning(f"⚠ 数据格式可能有问题，缺少字段: {missing_keys}")
            else:
                logger.info("✓ 数据格式正确")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 数据加载失败: {e}")
        return False

def test_project_modules() -> bool:
    """测试项目模块"""
    logger.info("=== 测试项目模块 ===")
    
    modules_to_test = [
        "data_loader",
        "model_manager", 
        "distillation_trainer",
        "policy_distillation_trainer",
    ]
    
    all_success = True
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            logger.info(f"✓ {module_name} 模块导入成功")
        except ImportError as e:
            logger.error(f"✗ {module_name} 模块导入失败: {e}")
            all_success = False
    
    return all_success

def generate_summary_report(package_results: List[Tuple[str, bool, str]], 
                          cuda_available: bool,
                          data_loading: bool,
                          modules_working: bool) -> None:
    """生成总结报告"""
    logger.info("=== 环境测试总结 ===")
    
    # 统计包导入结果
    required_packages = [r for r in package_results if r[2] == "必需"]
    failed_required = [r for r in required_packages if not r[1]]
    
    optional_packages = [r for r in package_results if r[2] == "可选"]
    success_optional = [r for r in optional_packages if r[1]]
    
    logger.info(f"必需包: {len(required_packages) - len(failed_required)}/{len(required_packages)} 成功")
    logger.info(f"可选包: {len(success_optional)}/{len(optional_packages)} 成功")
    logger.info(f"CUDA支持: {'✓' if cuda_available else '✗'}")
    logger.info(f"数据加载: {'✓' if data_loading else '✗'}")
    logger.info(f"项目模块: {'✓' if modules_working else '✗'}")
    
    # 评估整体状态
    if len(failed_required) == 0 and data_loading and modules_working:
        logger.info("🎉 环境配置完美！可以开始训练了。")
        status = "完美"
    elif len(failed_required) == 0:
        logger.info("✅ 环境基本配置正确，可以开始训练。")
        status = "良好"
    else:
        logger.error("❌ 环境配置有问题，请解决以下问题后再开始训练：")
        for package_name, success, package_type in failed_required:
            logger.error(f"  - 安装 {package_name}")
        status = "有问题"
    
    # 给出建议
    logger.info("=== 建议 ===")
    if not cuda_available:
        logger.info("- 安装CUDA支持的PyTorch版本以获得更好的性能")
    
    if len(success_optional) < len(optional_packages):
        logger.info("- 考虑安装可选包以获得更好的性能：")
        for package_name, success, package_type in optional_packages:
            if not success:
                if package_name == "Flash Attention":
                    logger.info("  pip install flash-attn --no-build-isolation")
                elif package_name == "xFormers": 
                    logger.info("  pip install xformers")
                elif package_name == "Wandb":
                    logger.info("  pip install wandb")
    
    return status

def main():
    """主测试函数"""
    logger.info("开始环境测试...")
    
    # 运行所有测试
    python_ok = test_python_version()
    package_results = test_package_imports()
    cuda_available = test_torch_cuda()
    transformers_ok = test_transformers_models()
    data_loading = test_data_loading()
    modules_working = test_project_modules()
    
    # 生成总结报告
    status = generate_summary_report(package_results, cuda_available, data_loading, modules_working)
    
    logger.info("环境测试完成！")
    
    return status == "完美" or status == "良好"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 