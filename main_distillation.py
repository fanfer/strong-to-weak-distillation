#!/usr/bin/env python3
"""
Qwen3 Strong-to-Weak Distillation 主训练脚本

该脚本实现了两阶段的蒸馏策略：
1. 非策略蒸馏阶段：结合教师模型在思考和非思考模式下的输出
2. 策略训练蒸馏阶段：通过KL散度对齐学生和教师模型的对数概率

用法:
    python main_distillation.py --config config/distillation_config.yaml
    python main_distillation.py --stage non_policy --config config/distillation_config.yaml
    python main_distillation.py --stage policy --config config/distillation_config.yaml
"""

import argparse
import logging
import os
import yaml
import json
from typing import Dict, Any
import torch

from distillation_trainer import DistillationConfig, create_distillation_trainer
from policy_distillation_trainer import PolicyDistillationConfig, create_policy_distillation_trainer
from model_manager import ModelManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('distillation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported config format. Use .yaml or .json")


def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    return {
        "models": {
            "teacher_model": "Qwen/Qwen3-235B-A22B",
            "student_model": "Qwen/Qwen3-8B",
            "load_in_4bit": True,
            "torch_dtype": "bfloat16"
        },
        "data": {
            "data_path": "dataset/train-00000-of-00001-cae87f8e074b4b5d.json",
            "max_length": 2048,
            "val_ratio": 0.1
        },
        "non_policy_distillation": {
            "num_epochs": 3,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "temperature": 3.0,
            "alpha": 0.7,
            "beta": 0.3,
            "mode_switch_loss_weight": 0.1,
            "save_steps": 500,
            "eval_steps": 100,
            "logging_steps": 50,
            "output_dir": "./outputs/non_policy_distillation"
        },
        "policy_distillation": {
            "num_epochs": 5,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "temperature": 2.0,
            "kl_weight": 1.0,
            "thinking_prob": 0.5,
            "sequence_length": 512,
            "top_k": 50,
            "top_p": 0.9,
            "save_steps": 200,
            "eval_steps": 50,
            "logging_steps": 20,
            "output_dir": "./outputs/policy_distillation"
        },
        "training": {
            "use_wandb": False,
            "wandb_project": "qwen3-distillation",
            "random_seed": 42
        }
    }


def save_config(config: Dict[str, Any], save_path: str):
    """保存配置文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if save_path.endswith('.yaml') or save_path.endswith('.yml'):
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    elif save_path.endswith('.json'):
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def setup_environment(config: Dict[str, Any]):
    """设置训练环境"""
    # 设置随机种子
    import random
    import numpy as np
    
    seed = config.get("training", {}).get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 设置CUDA优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
    
    # 显示配置信息
    logger.info("=== Training Configuration ===")
    logger.info(f"Teacher Model: {config['models']['teacher_model']}")
    logger.info(f"Student Model: {config['models']['student_model']}")
    logger.info(f"Data Path: {config['data']['data_path']}")
    logger.info(f"Use Wandb: {config['training'].get('use_wandb', False)}")


def run_non_policy_distillation(config: Dict[str, Any]) -> str:
    """运行非策略蒸馏阶段"""
    logger.info("=== Starting Non-Policy Distillation ===")
    
    # 创建配置对象
    non_policy_config = config["non_policy_distillation"]
    distillation_config = DistillationConfig(
        num_epochs=non_policy_config["num_epochs"],
        batch_size=non_policy_config["batch_size"],
        gradient_accumulation_steps=non_policy_config["gradient_accumulation_steps"],
        learning_rate=non_policy_config["learning_rate"],
        weight_decay=non_policy_config["weight_decay"],
        warmup_ratio=non_policy_config["warmup_ratio"],
        temperature=non_policy_config["temperature"],
        alpha=non_policy_config["alpha"],
        beta=non_policy_config["beta"],
        mode_switch_loss_weight=non_policy_config["mode_switch_loss_weight"],
        save_steps=non_policy_config["save_steps"],
        eval_steps=non_policy_config["eval_steps"],
        logging_steps=non_policy_config["logging_steps"],
        output_dir=non_policy_config["output_dir"],
        max_length=config["data"]["max_length"],
        data_path=config["data"]["data_path"],
        val_ratio=config["data"]["val_ratio"]
    )
    
    # 创建训练器
    trainer = create_distillation_trainer(
        config=distillation_config,
        teacher_model_name=config["models"]["teacher_model"],
        student_model_name=config["models"]["student_model"],
        use_wandb=config["training"].get("use_wandb", False)
    )
    
    # 开始训练
    trainer.train()
    
    # 返回最佳模型路径
    best_model_path = os.path.join(distillation_config.output_dir, "best_model")
    logger.info(f"Non-policy distillation completed. Best model saved to: {best_model_path}")
    
    return best_model_path


def run_policy_distillation(config: Dict[str, Any], pretrained_student_path: str):
    """运行策略训练蒸馏阶段"""
    logger.info("=== Starting Policy Distillation ===")
    
    # 创建配置对象
    policy_config = config["policy_distillation"]
    policy_distillation_config = PolicyDistillationConfig(
        num_epochs=policy_config["num_epochs"],
        batch_size=policy_config["batch_size"],
        gradient_accumulation_steps=policy_config["gradient_accumulation_steps"],
        learning_rate=policy_config["learning_rate"],
        weight_decay=policy_config["weight_decay"],
        warmup_ratio=policy_config["warmup_ratio"],
        temperature=policy_config["temperature"],
        kl_weight=policy_config["kl_weight"],
        thinking_prob=policy_config["thinking_prob"],
        sequence_length=policy_config["sequence_length"],
        top_k=policy_config["top_k"],
        top_p=policy_config["top_p"],
        save_steps=policy_config["save_steps"],
        eval_steps=policy_config["eval_steps"],
        logging_steps=policy_config["logging_steps"],
        output_dir=policy_config["output_dir"],
        max_length=config["data"]["max_length"],
        data_path=config["data"]["data_path"],
        val_ratio=config["data"]["val_ratio"]
    )
    
    # 创建训练器
    trainer = create_policy_distillation_trainer(
        config=policy_distillation_config,
        teacher_model_name=config["models"]["teacher_model"],
        student_model_name=config["models"]["student_model"],
        pretrained_student_path=pretrained_student_path,
        use_wandb=config["training"].get("use_wandb", False)
    )
    
    # 开始训练
    trainer.train()
    
    # 返回最终模型路径
    final_model_path = os.path.join(policy_distillation_config.output_dir, "final_policy_model")
    logger.info(f"Policy distillation completed. Final model saved to: {final_model_path}")
    
    return final_model_path


def test_model_loading(config: Dict[str, Any]):
    """测试模型加载"""
    logger.info("=== Testing Model Loading ===")
    
    try:
        # 创建模型管理器
        model_manager = ModelManager(
            teacher_model_name=config["models"]["teacher_model"],
            student_model_name=config["models"]["student_model"],
            load_in_4bit=config["models"]["load_in_4bit"],
            torch_dtype=getattr(torch, config["models"]["torch_dtype"])
        )
        
        # 加载分词器
        tokenizer = model_manager.load_tokenizer()
        logger.info(f"✓ Tokenizer loaded: {tokenizer.name_or_path}")
        
        # 尝试加载学生模型
        student_model = model_manager.load_student_model()
        logger.info(f"✓ Student model loaded: {config['models']['student_model']}")
        
        # 尝试加载教师模型（可能会失败）
        try:
            teacher_model = model_manager.load_teacher_model()
            if teacher_model is not None:
                logger.info(f"✓ Teacher model loaded: {config['models']['teacher_model']}")
            else:
                logger.warning("⚠ Teacher model loaded as pipeline")
        except Exception as e:
            logger.warning(f"⚠ Teacher model loading failed: {e}")
        
        # 显示模型信息
        info = model_manager.get_model_info()
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Model loading test completed successfully!")
        
    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Qwen3 Strong-to-Weak Distillation")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/distillation_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--stage", 
        type=str, 
        choices=["all", "non_policy", "policy", "test"],
        default="all",
        help="训练阶段：all（全部）、non_policy（非策略蒸馏）、policy（策略蒸馏）、test（测试模型加载）"
    )
    parser.add_argument(
        "--pretrained_student",
        type=str,
        help="预训练学生模型路径（用于策略蒸馏阶段）"
    )
    parser.add_argument(
        "--create_config",
        action="store_true",
        help="创建默认配置文件"
    )
    
    args = parser.parse_args()
    
    # 创建默认配置文件
    if args.create_config:
        config = create_default_config()
        os.makedirs("config", exist_ok=True)
        save_config(config, "config/distillation_config.yaml")
        logger.info("Default config created at config/distillation_config.yaml")
        return
    
    # 加载配置
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Config loaded from {args.config}")
    else:
        logger.info("Config file not found, using default config")
        config = create_default_config()
        # 保存默认配置
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        save_config(config, args.config)
    
    # 设置环境
    setup_environment(config)
    
    try:
        if args.stage == "test":
            # 仅测试模型加载
            test_model_loading(config)
            
        elif args.stage == "non_policy":
            # 仅运行非策略蒸馏
            best_model_path = run_non_policy_distillation(config)
            logger.info(f"Non-policy distillation completed. Model saved to: {best_model_path}")
            
        elif args.stage == "policy":
            # 仅运行策略蒸馏
            if not args.pretrained_student:
                # 尝试使用默认路径
                default_path = "./outputs/non_policy_distillation/best_model"
                if os.path.exists(default_path):
                    args.pretrained_student = default_path
                    logger.info(f"Using default pretrained student path: {default_path}")
                else:
                    raise ValueError("Pretrained student model path required for policy distillation")
            
            final_model_path = run_policy_distillation(config, args.pretrained_student)
            logger.info(f"Policy distillation completed. Model saved to: {final_model_path}")
            
        elif args.stage == "all":
            # 运行完整的两阶段训练
            logger.info("Starting complete two-stage distillation training...")
            
            # 阶段1：非策略蒸馏
            best_model_path = run_non_policy_distillation(config)
            
            # 阶段2：策略蒸馏
            final_model_path = run_policy_distillation(config, best_model_path)
            
            logger.info("=== Complete Distillation Training Finished ===")
            logger.info(f"Non-policy model: {best_model_path}")
            logger.info(f"Final policy model: {final_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main() 