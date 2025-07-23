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
import random
from model_manager import ModelManager
from data_loader import create_dataloader

logger = logging.getLogger(__name__)

@dataclass
class PolicyDistillationConfig:
    """策略训练蒸馏配置"""
    # 训练参数
    num_epochs: int = 5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5  # 策略训练使用更小的学习率
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    
    # 蒸馏参数
    temperature: float = 2.0  # 策略训练使用较低的温度
    kl_weight: float = 1.0  # KL散度权重
    
    # 策略训练特定参数
    thinking_prob: float = 0.5  # 思考模式的采样概率
    sequence_length: int = 512  # 生成序列的最大长度
    top_k: int = 50
    top_p: float = 0.9
    
    # RL相关参数（如果使用PPO等）
    use_ppo: bool = False
    ppo_epochs: int = 4
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_bonus: float = 0.01
    
    # 其他参数
    save_steps: int = 200
    eval_steps: int = 50
    logging_steps: int = 20
    output_dir: str = "./outputs/policy_distillation"
    max_length: int = 2048
    
    # 数据参数
    data_path: str = "dataset/train-00000-of-00001-cae87f8e074b4b5d.json"
    val_ratio: float = 0.1


class PolicyDistillationTrainer:
    """策略训练蒸馏器"""
    
    def __init__(
        self,
        config: PolicyDistillationConfig,
        model_manager: ModelManager,
        use_wandb: bool = False,
        pretrained_student_path: Optional[str] = None
    ):
        self.config = config
        self.model_manager = model_manager
        self.use_wandb = use_wandb
        
        # 加载模型和分词器
        self.tokenizer = model_manager.load_tokenizer()
        self.teacher_model = model_manager.load_teacher_model()
        self.student_model = model_manager.load_student_model()
        
        # 如果提供了预训练的学生模型路径，加载它
        if pretrained_student_path:
            self.load_pretrained_student(pretrained_student_path)
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 初始化训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_kl_divergence = float('inf')
        
        # 损失函数
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def load_pretrained_student(self, pretrained_path: str):
        """加载预训练的学生模型（来自非策略蒸馏阶段）"""
        logger.info(f"Loading pretrained student model from {pretrained_path}")
        
        # 加载模型状态
        if os.path.exists(os.path.join(pretrained_path, "pytorch_model.bin")):
            state_dict = torch.load(
                os.path.join(pretrained_path, "pytorch_model.bin"), 
                map_location=self.student_model.device
            )
            self.student_model.load_state_dict(state_dict)
        
        logger.info("Pretrained student model loaded successfully")
    
    def setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """设置数据加载器"""
        # 策略训练阶段主要使用prompt，让模型自己生成
        train_dataloader = create_dataloader(
            data_path=self.config.data_path,
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            mode="both",
            shuffle=True
        )
        
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
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_training_steps * self.config.warmup_ratio),
            num_training_steps=num_training_steps
        )
    
    def extract_prompt_from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """从批次中提取prompt部分"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        prompt_lengths = batch['prompt_lengths']
        
        # 提取prompt部分
        prompts = []
        prompt_masks = []
        
        for i, prompt_len in enumerate(prompt_lengths):
            prompt = input_ids[i, :prompt_len]
            prompt_mask = attention_mask[i, :prompt_len]
            
            prompts.append(prompt)
            prompt_masks.append(prompt_mask)
        
        # 填充到相同长度
        max_prompt_len = max([p.shape[0] for p in prompts])
        
        padded_prompts = []
        padded_masks = []
        
        for prompt, mask in zip(prompts, prompt_masks):
            pad_len = max_prompt_len - prompt.shape[0]
            if pad_len > 0:
                padded_prompt = torch.cat([
                    torch.full((pad_len,), self.tokenizer.pad_token_id, 
                              dtype=prompt.dtype, device=prompt.device),
                    prompt
                ])
                padded_mask = torch.cat([
                    torch.zeros(pad_len, dtype=mask.dtype, device=mask.device),
                    mask
                ])
            else:
                padded_prompt = prompt
                padded_mask = mask
            
            padded_prompts.append(padded_prompt)
            padded_masks.append(padded_mask)
        
        return {
            'input_ids': torch.stack(padded_prompts),
            'attention_mask': torch.stack(padded_masks)
        }
    
    def generate_sequences(
        self, 
        prompts: Dict[str, torch.Tensor], 
        thinking_mode: bool = False
    ) -> Dict[str, torch.Tensor]:
        """生成学生模型的序列"""
        
        # 为思考模式添加特殊token
        if thinking_mode:
            # 添加思考模式的提示
            thinking_prompt = "\n请先在<思考>标签内进行推理分析，然后给出最终答案。\n<思考>\n"
            thinking_tokens = self.tokenizer.encode(thinking_prompt, add_special_tokens=False)
            
            # 添加到prompts
            batch_size = prompts['input_ids'].shape[0]
            thinking_tensor = torch.tensor(thinking_tokens, device=prompts['input_ids'].device)
            thinking_tensor = thinking_tensor.unsqueeze(0).repeat(batch_size, 1)
            
            # 合并
            prompts['input_ids'] = torch.cat([prompts['input_ids'], thinking_tensor], dim=1)
            thinking_mask = torch.ones_like(thinking_tensor)
            prompts['attention_mask'] = torch.cat([prompts['attention_mask'], thinking_mask], dim=1)
        
        # 使用学生模型生成
        with torch.no_grad():
            generated = self.student_model.generate(
                **prompts,
                max_new_tokens=self.config.sequence_length,
                do_sample=True,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                temperature=1.0,  # 生成时使用标准温度
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        return {
            'sequences': generated.sequences,
            'scores': generated.scores
        }
    
    def compute_teacher_student_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = None
    ) -> torch.Tensor:
        """计算学生和教师模型之间的KL散度"""
        if temperature is None:
            temperature = self.config.temperature
        
        # 应用温度缩放
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        # 计算KL散度，只在有效位置计算
        kl_div = F.kl_div(
            student_log_probs.view(-1, student_log_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction='none'
        ).sum(dim=-1)
        
        # 应用attention mask
        valid_mask = attention_mask.view(-1).bool()
        kl_div = kl_div * valid_mask.float()
        
        # 计算平均KL散度
        return kl_div.sum() / valid_mask.sum()
    
    def policy_training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """策略训练步骤"""
        self.student_model.train()
        
        # 1. 从批次中提取prompts
        prompts = self.extract_prompt_from_batch(batch)
        
        # 2. 决定使用思考模式还是非思考模式
        thinking_mode = random.random() < self.config.thinking_prob
        
        # 3. 学生模型生成序列
        student_generation = self.generate_sequences(prompts, thinking_mode)
        student_sequences = student_generation['sequences']
        
        # 4. 获取学生模型的logits
        student_outputs = self.student_model(
            input_ids=student_sequences,
            attention_mask=torch.ones_like(student_sequences),
            return_dict=True
        )
        student_logits = student_outputs.logits
        
        # 5. 获取教师模型的logits
        teacher_logits = None
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=student_sequences,
                    attention_mask=torch.ones_like(student_sequences),
                    return_dict=True
                )
                teacher_logits = teacher_outputs.logits
        
        # 6. 计算KL散度损失
        if teacher_logits is not None:
            attention_mask = torch.ones_like(student_sequences)
            kl_loss = self.compute_teacher_student_kl(
                student_logits[:, :-1],  # 排除最后一个token
                teacher_logits[:, :-1],
                attention_mask[:, :-1]
            )
            
            total_loss = self.config.kl_weight * kl_loss
        else:
            # 如果没有教师模型，使用自回归损失
            labels = student_sequences[:, 1:].clone()  # 移位标签
            total_loss = F.cross_entropy(
                student_logits[:, :-1].reshape(-1, student_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
            kl_loss = total_loss
        
        # 7. 反向传播
        loss = total_loss / self.config.gradient_accumulation_steps
        loss.backward()
        
        # 8. 梯度累积和优化
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.student_model.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return {
            'total_loss': total_loss.item(),
            'kl_loss': kl_loss.item(),
            'thinking_mode': int(thinking_mode)
        }
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """验证步骤"""
        self.student_model.eval()
        
        with torch.no_grad():
            # 提取prompts
            prompts = self.extract_prompt_from_batch(batch)
            
            # 测试两种模式
            losses = {}
            
            for mode_name, thinking_mode in [("thinking", True), ("non_thinking", False)]:
                # 生成序列
                generation = self.generate_sequences(prompts, thinking_mode)
                sequences = generation['sequences']
                
                # 获取logits
                student_outputs = self.student_model(
                    input_ids=sequences,
                    attention_mask=torch.ones_like(sequences),
                    return_dict=True
                )
                student_logits = student_outputs.logits
                
                if self.teacher_model is not None:
                    teacher_outputs = self.teacher_model(
                        input_ids=sequences,
                        attention_mask=torch.ones_like(sequences),
                        return_dict=True
                    )
                    teacher_logits = teacher_outputs.logits
                    
                    # 计算KL散度
                    attention_mask = torch.ones_like(sequences)
                    kl_loss = self.compute_teacher_student_kl(
                        student_logits[:, :-1],
                        teacher_logits[:, :-1],
                        attention_mask[:, :-1]
                    )
                    
                    losses[f'{mode_name}_kl_loss'] = kl_loss.item()
        
        # 计算平均验证损失
        avg_kl_loss = np.mean([v for k, v in losses.items() if 'kl_loss' in k])
        losses['avg_kl_loss'] = avg_kl_loss
        
        return losses
    
    def train_epoch(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """训练一个epoch"""
        total_loss = 0
        num_batches = len(train_dataloader)
        
        progress_bar = tqdm(train_dataloader, desc=f"Policy Epoch {self.epoch}")
        
        for step, batch in enumerate(progress_bar):
            # 移动到设备
            batch = {k: v.to(self.student_model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 策略训练步骤
            losses = self.policy_training_step(batch)
            total_loss += losses['total_loss']
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'kl': f"{losses['kl_loss']:.4f}",
                'mode': "T" if losses['thinking_mode'] else "N",
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
                if val_losses['avg_kl_loss'] < self.best_kl_divergence:
                    self.best_kl_divergence = val_losses['avg_kl_loss']
                    self.save_model("best_policy_model")
            
            # 保存检查点
            if self.global_step % self.config.save_steps == 0:
                self.save_model(f"policy_checkpoint-{self.global_step}")
            
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Policy Epoch {self.epoch} completed. Average loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """验证模型"""
        total_losses = {}
        num_batches = min(len(val_dataloader), 10)  # 限制验证批次数量以节省时间
        
        for i, batch in enumerate(val_dataloader):
            if i >= num_batches:
                break
                
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
        
        # 保存配置
        with open(os.path.join(save_path, "policy_distillation_config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Policy model saved to {save_path}")
    
    def train(self):
        """开始策略训练"""
        logger.info("Starting policy distillation training...")
        
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
                name="policy_distillation"
            )
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            avg_loss = self.train_epoch(train_dataloader, val_dataloader)
            
            # 每个epoch结束后保存模型
            self.save_model(f"policy_epoch-{epoch}")
        
        # 保存最终模型
        self.save_model("final_policy_model")
        
        if self.use_wandb:
            wandb.finish()
        
        logger.info("Policy training completed!")


def create_policy_distillation_trainer(
    config: PolicyDistillationConfig,
    teacher_model_name: str = "Qwen/Qwen3-235B-A22B",
    student_model_name: str = "Qwen/Qwen3-8B",
    pretrained_student_path: Optional[str] = None,
    use_wandb: bool = False
) -> PolicyDistillationTrainer:
    """创建策略蒸馏训练器"""
    
    # 创建模型管理器
    model_manager = ModelManager(
        teacher_model_name=teacher_model_name,
        student_model_name=student_model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16
    )
    
    # 创建训练器
    trainer = PolicyDistillationTrainer(
        config=config,
        model_manager=model_manager,
        pretrained_student_path=pretrained_student_path,
        use_wandb=use_wandb
    )
    
    return trainer


if __name__ == "__main__":
    # 示例配置
    config = PolicyDistillationConfig(
        num_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        output_dir="./outputs/policy_distillation"
    )
    
    # 创建训练器（假设已有非策略蒸馏的结果）
    trainer = create_policy_distillation_trainer(
        config, 
        pretrained_student_path="./outputs/non_policy_distillation/best_model",
        use_wandb=False
    )
    
    # 开始训练
    trainer.train() 