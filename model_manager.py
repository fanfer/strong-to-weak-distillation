import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
from typing import Dict, List, Optional, Tuple, Union
import logging
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import gc

logger = logging.getLogger(__name__)

class ModelManager:
    """模型管理器，负责加载和管理教师模型和学生模型"""
    
    def __init__(
        self,
        teacher_model_name: str = "Qwen/Qwen3-235B-A22B",
        student_model_name: str = "Qwen/Qwen3-8B",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_memory: Optional[Dict] = None,
        offload_folder: Optional[str] = "./offload"
    ):
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_memory = max_memory
        self.offload_folder = offload_folder
        
        self.teacher_model = None
        self.student_model = None
        self.tokenizer = None
        
        # 配置量化
        self.bnb_config = None
        if load_in_4bit or load_in_8bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
    
    def load_tokenizer(self) -> AutoTokenizer:
        """加载分词器"""
        if self.tokenizer is None:
            logger.info(f"Loading tokenizer from {self.student_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.student_model_name,
                trust_remote_code=True,
                pad_token="<|endoftext|>",
                eos_token="<|im_end|>",
                padding_side="left"
            )
            
            # 确保有pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        return self.tokenizer
    
    def load_teacher_model(self) -> AutoModelForCausalLM:
        """加载教师模型"""
        if self.teacher_model is None:
            logger.info(f"Loading teacher model: {self.teacher_model_name}")
            
            try:
                self.teacher_model = AutoModelForCausalLM.from_pretrained(
                    self.teacher_model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                    trust_remote_code=True,
                    quantization_config=self.bnb_config,
                    max_memory=self.max_memory,
                    offload_folder=self.offload_folder,
                    low_cpu_mem_usage=True,
                )
                
                # 设置为评估模式（教师模型不需要训练）
                self.teacher_model.eval()
                
                # 禁用梯度计算
                for param in self.teacher_model.parameters():
                    param.requires_grad = False
                    
                logger.info("Teacher model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load teacher model: {e}")
                # 如果加载失败，尝试使用pipeline方式
                logger.info("Trying to load teacher model with pipeline...")
                self.teacher_pipeline = pipeline(
                    "text-generation",
                    model=self.teacher_model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                    trust_remote_code=True,
                    model_kwargs={
                        "quantization_config": self.bnb_config,
                        "low_cpu_mem_usage": True,
                    }
                )
                return None
                
        return self.teacher_model
    
    def load_student_model(self) -> AutoModelForCausalLM:
        """加载学生模型"""
        if self.student_model is None:
            logger.info(f"Loading student model: {self.student_model_name}")
            
            self.student_model = AutoModelForCausalLM.from_pretrained(
                self.student_model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                trust_remote_code=True,
                quantization_config=self.bnb_config if not self.load_in_4bit else None,  # 学生模型避免4bit量化以保持训练精度
                low_cpu_mem_usage=True,
            )
            
            # 学生模型需要训练
            self.student_model.train()
            
            logger.info("Student model loaded successfully")
                
        return self.student_model
    
    def get_teacher_logits(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """获取教师模型的logits"""
        if self.teacher_model is None:
            raise ValueError("Teacher model not loaded")
            
        with torch.no_grad():
            outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # 应用温度缩放
            logits = outputs.logits / temperature
            
        return logits
    
    def get_student_logits(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """获取学生模型的logits"""
        if self.student_model is None:
            raise ValueError("Student model not loaded")
            
        outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 应用温度缩放
        logits = outputs.logits / temperature
        
        return logits
    
    def generate_teacher_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        do_sample: bool = True,
        thinking_mode: bool = False
    ) -> str:
        """使用教师模型生成响应"""
        if thinking_mode:
            # 为思考模式添加特殊提示
            prompt += "\n请先在<思考>标签内进行推理分析，然后给出最终答案。"
        
        if hasattr(self, 'teacher_pipeline') and self.teacher_pipeline is not None:
            # 使用pipeline生成
            outputs = self.teacher_pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return outputs[0]['generated_text']
        
        elif self.teacher_model is not None:
            # 直接使用模型生成
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.teacher_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.teacher_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解码生成的文本
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            return generated_text
        
        else:
            raise ValueError("No teacher model available for generation")
    
    def clear_cache(self):
        """清理缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_model_info(self) -> Dict[str, str]:
        """获取模型信息"""
        info = {
            "teacher_model": self.teacher_model_name,
            "student_model": self.student_model_name,
            "torch_dtype": str(self.torch_dtype),
            "device_map": str(self.device_map),
            "quantization": f"8bit: {self.load_in_8bit}, 4bit: {self.load_in_4bit}"
        }
        
        if self.teacher_model is not None:
            info["teacher_loaded"] = "Direct model"
        elif hasattr(self, 'teacher_pipeline'):
            info["teacher_loaded"] = "Pipeline"
        else:
            info["teacher_loaded"] = "Not loaded"
            
        if self.student_model is not None:
            info["student_loaded"] = "Yes"
        else:
            info["student_loaded"] = "No"
            
        return info


def create_model_manager(
    teacher_model_name: str = "Qwen/Qwen3-235B-A22B",
    student_model_name: str = "Qwen/Qwen3-8B", 
    load_in_4bit: bool = True,
    max_memory: Optional[Dict] = None
) -> ModelManager:
    """创建模型管理器的便捷函数"""
    
    # 如果没有指定最大内存，自动设置
    if max_memory is None:
        max_memory = {0: "40GB", "cpu": "50GB"}  # 根据实际硬件情况调整
    
    manager = ModelManager(
        teacher_model_name=teacher_model_name,
        student_model_name=student_model_name,
        load_in_4bit=load_in_4bit,
        max_memory=max_memory,
        torch_dtype=torch.bfloat16
    )
    
    return manager


if __name__ == "__main__":
    # 测试模型管理器
    manager = create_model_manager()
    
    # 加载分词器
    tokenizer = manager.load_tokenizer()
    print("Tokenizer loaded:", tokenizer.name_or_path)
    
    # 显示模型信息
    info = manager.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}") 