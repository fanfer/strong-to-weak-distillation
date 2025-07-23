import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import os
import json
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
import wandb
from model_manager import ModelManager
from data_loader import create_dataloader

logger = logging.getLogger(__name__)

@dataclass
class DistillationConfig:
    """蒸馏训练配置"""
    # 训练参数
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # 蒸馏参数
    temperature: float = 3.0
    alpha: float = 0.7  # 蒸馏损失权重
    beta: float = 0.3   # 原始损失权重
    
    # 模式切换参数
    mode_switch_loss_weight: float = 0.1
    thinking_weight: float = 0.6  # 思考模式权重
    non_thinking_weight: float = 0.4  # 非思考模式权重
    
    # 其他参数
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    output_dir: str = "./outputs"
    max_length: int = 2048
    
    # 数据参数
    data_path: str = "dataset/train-00000-of-00001-cae87f8e074b4b5d.json"
    val_ratio: float = 0.1


class DistillationTrainer:
    """Strong-to-Weak 蒸馏训练器"""
    
    def __init__(
        self,
        config: DistillationConfig,
        model_manager: ModelManager,
        use_wandb: bool = False
    ):
        self.config = config
        self.model_manager = model_manager
        self.use_wandb = use_wandb
        
        # 加载模型和分词器
        self.tokenizer = model_manager.load_tokenizer()
        self.teacher_model = model_manager.load_teacher_model()
        self.student_model = model_manager.load_student_model()
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 初始化训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # 损失函数
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 模式分类器（用于判断思考/非思考模式）
        self.mode_classifier = nn.Linear(
            self.student_model.config.hidden_size, 
            2  # 思考/非思考
        ).to(self.student_model.device)
        
    def setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """设置数据加载器"""
        # 创建训练和验证数据加载器
        train_dataloader = create_dataloader(
            data_path=self.config.data_path,
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            mode="both",
            shuffle=True
        )
        
        # 创建验证集（使用部分训练数据）
        val_dataloader = create_dataloader(
            data_path=self.config.data_path,
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            mode="both",
            shuffle=False
        )
        
        return train_dataloader, val_dataloader
    
    def setup_optimizer_and_scheduler(self, num_training_steps: int):
        """设置优化器和学习率调度器"""
        # 只训练学生模型和模式分类器
        params_to_train = list(self.student_model.parameters()) + list(self.mode_classifier.parameters())
        
        self.optimizer = torch.optim.AdamW(
            params_to_train,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_training_steps * self.config.warmup_ratio),
            num_training_steps=num_training_steps
        )
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = None
    ) -> Dict[str, torch.Tensor]:
        """计算蒸馏损失"""
        if temperature is None:
            temperature = self.config.temperature
        
        # 只计算生成部分的损失（排除prompt部分）
        # 这里需要根据prompt_lengths来确定计算损失的范围
        
        # KL散度损失
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        # 创建有效位置的mask（排除padding和prompt）
        valid_mask = attention_mask.bool()
        
        # 计算KL散度
        kl_loss = self.kl_loss(student_probs.view(-1, student_probs.size(-1)), 
                              teacher_probs.view(-1, teacher_probs.size(-1)))
        
        # 原始交叉熵损失
        ce_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )
        
        # 组合损失
        total_loss = (
            self.config.alpha * kl_loss * (temperature ** 2) + 
            self.config.beta * ce_loss
        )
        
        return {
            'total_loss': total_loss,
            'kl_loss': kl_loss,
            'ce_loss': ce_loss
        }
    
    def compute_mode_classification_loss(
        self,
        student_hidden_states: torch.Tensor,
        mode_labels: torch.Tensor
    ) -> torch.Tensor:
        """计算模式分类损失"""
        # 使用学生模型的最后一层隐藏状态的平均值
        pooled_hidden = student_hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # 模式分类
        mode_logits = self.mode_classifier(pooled_hidden)
        mode_loss = F.cross_entropy(mode_logits, mode_labels)
        
        return mode_loss
    
    def forward_pass(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        prompt_lengths = batch['prompt_lengths']
        mode_labels = batch['mode_labels']
        
        # 创建标签（只对生成部分计算损失）
        labels = input_ids.clone()
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100  # 忽略prompt部分
        
        # 学生模型前向传播
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        student_logits = student_outputs.logits
        student_hidden_states = student_outputs.hidden_states[-1]
        
        # 教师模型前向传播（如果可用）
        teacher_logits = None
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                teacher_logits = teacher_outputs.logits
        
        # 计算损失
        losses = {}
        
        if teacher_logits is not None:
            # 蒸馏损失
            distill_losses = self.compute_distillation_loss(
                student_logits, teacher_logits, labels, attention_mask
            )
            losses.update(distill_losses)
        else:
            # 如果没有教师模型，只计算交叉熵损失
            ce_loss = self.ce_loss(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1)
            )
            losses['total_loss'] = ce_loss
            losses['ce_loss'] = ce_loss
        
        # 模式分类损失
        mode_loss = self.compute_mode_classification_loss(
            student_hidden_states, mode_labels
        )
        losses['mode_loss'] = mode_loss
        
        # 总损失
        if 'total_loss' in losses:
            losses['total_loss'] = (
                losses['total_loss'] + 
                self.config.mode_switch_loss_weight * mode_loss
            )
        else:
            losses['total_loss'] = mode_loss
        
        return losses
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练步骤"""
        self.student_model.train()
        self.mode_classifier.train()
        
        # 前向传播
        losses = self.forward_pass(batch)
        loss = losses['total_loss']
        
        # 反向传播
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        
        # 梯度累积
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.student_model.parameters()) + list(self.mode_classifier.parameters()),
                self.config.max_grad_norm
            )
            
            # 优化器步骤
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        # 转换为标量并返回
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """验证步骤"""
        self.student_model.eval()
        self.mode_classifier.eval()
        
        with torch.no_grad():
            losses = self.forward_pass(batch)
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    def train_epoch(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """训练一个epoch"""
        total_loss = 0
        num_batches = len(train_dataloader)
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {self.epoch}")
        
        for step, batch in enumerate(progress_bar):
            # 移动到设备
            batch = {k: v.to(self.student_model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 训练步骤
            losses = self.training_step(batch)
            total_loss += losses['total_loss']
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # 记录日志
            if self.global_step % self.config.logging_steps == 0:
                self.log_metrics(losses, prefix="train")
            
            # 验证
            if self.global_step % self.config.eval_steps == 0:
                val_losses = self.validate(val_dataloader)
                self.log_metrics(val_losses, prefix="val")
                
                # 保存最佳模型
                if val_losses['total_loss'] < self.best_loss:
                    self.best_loss = val_losses['total_loss']
                    self.save_model("best_model")
            
            # 保存检查点
            if self.global_step % self.config.save_steps == 0:
                self.save_model(f"checkpoint-{self.global_step}")
            
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {self.epoch} completed. Average loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """验证模型"""
        total_losses = {}
        num_batches = len(val_dataloader)
        
        for batch in val_dataloader:
            batch = {k: v.to(self.student_model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            losses = self.validation_step(batch)
            
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += value
        
        # 计算平均损失
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """记录指标"""
        if self.use_wandb:
            log_dict = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
            log_dict['step'] = self.global_step
            log_dict['epoch'] = self.epoch
            wandb.log(log_dict)
        
        # 记录到日志
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"[{prefix}] Step {self.global_step}: {metrics_str}")
    
    def save_model(self, save_name: str):
        """保存模型"""
        save_path = os.path.join(self.config.output_dir, save_name)
        os.makedirs(save_path, exist_ok=True)
        
        # 保存学生模型
        self.student_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # 保存模式分类器
        torch.save(self.mode_classifier.state_dict(), 
                  os.path.join(save_path, "mode_classifier.pth"))
        
        # 保存配置
        with open(os.path.join(save_path, "distillation_config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def train(self):
        """开始训练"""
        logger.info("Starting distillation training...")
        
        # 设置数据加载器
        train_dataloader, val_dataloader = self.setup_dataloaders()
        
        # 计算训练步数
        num_training_steps = (
            len(train_dataloader) * self.config.num_epochs // 
            self.config.gradient_accumulation_steps
        )
        
        # 设置优化器和调度器
        self.setup_optimizer_and_scheduler(num_training_steps)
        
        # 初始化wandb
        if self.use_wandb:
            wandb.init(
                project="qwen3-distillation",
                config=self.config.__dict__,
                name="non_policy_distillation"
            )
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            avg_loss = self.train_epoch(train_dataloader, val_dataloader)
            
            # 每个epoch结束后保存模型
            self.save_model(f"epoch-{epoch}")
        
        # 保存最终模型
        self.save_model("final_model")
        
        if self.use_wandb:
            wandb.finish()
        
        logger.info("Training completed!")


def create_distillation_trainer(
    config: DistillationConfig,
    teacher_model_name: str = "Qwen/Qwen3-235B-A22B",
    student_model_name: str = "Qwen/Qwen3-8B",
    use_wandb: bool = False
) -> DistillationTrainer:
    """创建蒸馏训练器"""
    
    # 创建模型管理器
    model_manager = ModelManager(
        teacher_model_name=teacher_model_name,
        student_model_name=student_model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16
    )
    
    # 创建训练器
    trainer = DistillationTrainer(
        config=config,
        model_manager=model_manager,
        use_wandb=use_wandb
    )
    
    return trainer


if __name__ == "__main__":
    # 示例配置
    config = DistillationConfig(
        num_epochs=2,
        batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        output_dir="./outputs/non_policy_distillation"
    )
    
    # 创建训练器
    trainer = create_distillation_trainer(config, use_wandb=False)
    
    # 开始训练
    trainer.train() 