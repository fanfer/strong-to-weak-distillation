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
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def format_prompt(self, instruction: str, input_text: str, think: str = "") -> str:
        """格式化提示词"""
        prompt = f"""请根据以下指令和输入，提供评分和理由。

指令: {instruction}
输入: {input_text}
"""
        if think:
            prompt += f"思考过程: {think}\n"
        
        prompt += """
请以JSON格式回答，包含score (0-9的整数) 和reason (详细理由):
{"score": 分数, "reason": "理由"}
"""
        
        return prompt
    
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