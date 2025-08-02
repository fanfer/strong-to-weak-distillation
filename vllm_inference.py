"""
VLLM推理模块
支持Qwen3-4B模型加载和批量推理
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import time

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Warning: vllm not installed. Please install it with: pip install vllm")
    LLM = None
    SamplingParams = None

@dataclass
class InferenceConfig:
    """推理配置类"""
    model_name: str = "Qwen/Qwen2.5-4B-Instruct"  # Qwen3-4B模型路径
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 512
    batch_size: int = 32
    reserved_tokens: int = 512  # 为输出预留的token数

class VLLMInferenceEngine:
    """VLLM推理引擎"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.llm = None
        self.sampling_params = None
        self._initialize_model()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("VLLMInference")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_model(self):
        """初始化VLLM模型"""
        if LLM is None:
            raise ImportError("vllm is not installed. Please install it first.")
        
        try:
            self.logger.info(f"Loading model: {self.config.model_name}")
            self.llm = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                trust_remote_code=True,
            )
            
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )
            
            # 初始化tokenizer用于截断
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name, 
                    trust_remote_code=True
                )
                self.logger.info("Tokenizer loaded for text truncation")
            except Exception as e:
                self.logger.warning(f"Failed to load tokenizer: {e}, will use character-based truncation")
                self.tokenizer = None
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def format_prompt(self, instruction: str, input_text: str, think: str = "") -> str:
        """格式化提示词"""
        # 构建基础prompt模板
        base_prompt = f"""请根据以下指令和输入，提供评分和理由。

指令: {instruction}
输入: {{input_text}}
"""
        if think:
            base_prompt += f"思考过程: {think}\n"
        
        base_prompt += """
请以JSON格式回答，包含score (0-9的整数) 和reason (详细理由):
{"score": 分数, "reason": "理由"}
"""
        
        # 计算可用于input的最大token数
        max_input_tokens = self.config.max_model_len - self.config.reserved_tokens
        
        # 截断input_text以适应长度限制
        truncated_input = self._truncate_input_text(
            base_prompt, input_text, max_input_tokens
        )
        
        # 生成最终prompt
        final_prompt = base_prompt.format(input_text=truncated_input)
        
        return final_prompt
    
    def _truncate_input_text(self, base_prompt: str, input_text: str, max_tokens: int) -> str:
        """截断输入文本以适应长度限制"""
        if self.tokenizer is None:
            # 如果没有tokenizer，使用字符截断作为fallback
            return self._truncate_by_chars(base_prompt, input_text, max_tokens)
        
        try:
            # 计算基础prompt(不包含input_text)的token数
            base_prompt_tokens = len(self.tokenizer.encode(base_prompt.format(input_text="")))
            
            # 计算input_text可用的最大token数
            available_tokens = max_tokens - base_prompt_tokens
            
            if available_tokens <= 0:
                self.logger.warning("Base prompt too long, using minimal input")
                return input_text[:100] + "..."  # 最小输入
            
            # 编码input_text
            input_tokens = self.tokenizer.encode(input_text)
            
            # 如果不需要截断，直接返回
            if len(input_tokens) <= available_tokens:
                return input_text
            
            # 截断并解码
            truncated_tokens = input_tokens[:available_tokens - 10]  # 预留一些空间
            truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
            # 添加截断标识
            truncated_text += "\n[注: 输入文本已截断]"
            
            self.logger.debug(f"Input truncated from {len(input_tokens)} to {len(truncated_tokens)} tokens")
            
            return truncated_text
            
        except Exception as e:
            self.logger.warning(f"Token-based truncation failed: {e}, using character-based fallback")
            return self._truncate_by_chars(base_prompt, input_text, max_tokens)
    
    def _truncate_by_chars(self, base_prompt: str, input_text: str, max_tokens: int) -> str:
        """基于字符数的截断fallback方法"""
        # 粗略估算：1个token ≈ 1.5个中文字符或0.75个英文单词
        estimated_base_chars = len(base_prompt.format(input_text=""))
        estimated_max_chars = max_tokens * 1.5
        
        available_chars = int(estimated_max_chars - estimated_base_chars)
        
        if available_chars <= 0:
            return input_text[:100] + "..."
        
        if len(input_text) <= available_chars:
            return input_text
        
        # 截断并添加标识
        truncated_text = input_text[:available_chars - 20] + "\n[注: 输入文本已截断]"
        
        self.logger.debug(f"Input truncated by characters from {len(input_text)} to {len(truncated_text)} chars")
        
        return truncated_text
    
    def batch_inference(self, prompts: List[str]) -> List[str]:
        """批量推理"""
        try:
            self.logger.info(f"Starting batch inference for {len(prompts)} prompts")
            start_time = time.time()
            
            outputs = self.llm.generate(prompts, self.sampling_params)
            results = [output.outputs[0].text.strip() for output in outputs]
            
            end_time = time.time()
            self.logger.info(f"Batch inference completed in {end_time - start_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch inference failed: {e}")
            raise
    
    def single_inference(self, prompt: str) -> str:
        """单条推理"""
        return self.batch_inference([prompt])[0]
    
    def process_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """处理单个JSONL文件"""
        self.logger.info(f"Processing file: {file_path}")
        
        # 读取JSONL文件
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        
        if not data:
            self.logger.warning(f"No valid data found in {file_path}")
            return []
        
        # 准备批量推理
        prompts = []
        for item in data:
            prompt = self.format_prompt(
                item.get('instruction', ''),
                item.get('input', ''),
                item.get('think', '')
            )
            prompts.append(prompt)
        
        # 批量推理
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = self.batch_inference(batch_prompts)
            
            for j, result in enumerate(batch_results):
                original_idx = i + j
                processed_result = {
                    'index': original_idx,
                    'original_data': data[original_idx],
                    'generated_text': result,
                    'parsed_output': self._parse_output(result)
                }
                results.append(processed_result)
        
        self.logger.info(f"Completed processing {file_path}: {len(results)} items")
        return results
    
    def _parse_output(self, generated_text: str) -> Optional[Dict[str, Any]]:
        """解析生成的输出"""
        try:
            # 尝试提取JSON部分
            start = generated_text.find('{')
            end = generated_text.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = generated_text[start:end]
                parsed = json.loads(json_str)
                
                # 验证必需字段
                if 'score' in parsed and 'reason' in parsed:
                    # 确保score是0-9的整数
                    try:
                        score = int(parsed['score'])
                        if 0 <= score <= 9:
                            parsed['score'] = score
                            return parsed
                    except (ValueError, TypeError):
                        pass
            
            return None
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse output: {e}")
            return None
    
    def save_intermediate_results(self, results: List[Dict[str, Any]], 
                                output_path: str):
        """保存中间结果"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Saved intermediate results to {output_path}")

def main():
    """测试函数"""
    config = InferenceConfig()
    engine = VLLMInferenceEngine(config)
    
    # 测试处理文件
    test_file = "dataset/test/test_file1.jsonl"
    if os.path.exists(test_file):
        results = engine.process_jsonl_file(test_file)
        print(f"Processed {len(results)} items")
        for result in results[:2]:  # 显示前两个结果
            print(f"Generated: {result['generated_text']}")
            print(f"Parsed: {result['parsed_output']}")
            print("-" * 50)

if __name__ == "__main__":
    main() 