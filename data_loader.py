import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DistillationDataset(Dataset):
    """Strong-to-Weak 蒸馏数据集类"""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        mode: str = "both"  # "thinking", "non_thinking", "both"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
        # 加载数据
        self.data = self._load_data(data_path)
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载JSON格式的数据集"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _create_prompt(self, item: Dict[str, Any], thinking_mode: bool = False) -> str:
        """根据模式创建提示词"""
        instruction = item['instruction']
        input_text = item['input']
        
        if thinking_mode:
            # 思考模式：引导模型进行推理
            prompt = f"""<|im_start|>system
你是一个优秀的AI助手。在回答问题时，请先进行深入思考，然后给出最终答案。

<|im_start|>user
{instruction}

输入文本：{input_text}

请先在<思考>标签内进行推理分析，然后给出最终答案。<|im_end|>
<|im_start|>assistant
<思考>
让我仔细分析这个问题和输入文本：
"""
        else:
            # 非思考模式：直接回答
            prompt = f"""<|im_start|>system
你是一个优秀的AI助手。请根据给定的指令和输入文本，直接给出准确的答案。

<|im_start|>user
{instruction}

输入文本：{input_text}<|im_end|>
<|im_start|>assistant
"""
        
        return prompt
    
    def _create_response(self, item: Dict[str, Any], thinking_mode: bool = False) -> str:
        """创建期望的响应"""
        output = item['output']
        
        if thinking_mode:
            # 为思考模式创建带推理过程的响应
            response = f"""<思考>
根据输入文本，我需要分析给定的关系选项，并确定最符合上下文的关系。
通过分析文本内容，可以确定正确的关系。
</思考>

{output}"""
        else:
            # 直接响应
            response = output
            
        return response
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # 根据模式创建不同的样本
        if self.mode == "thinking":
            prompt = self._create_prompt(item, thinking_mode=True)
            response = self._create_response(item, thinking_mode=True)
            mode_label = 1
        elif self.mode == "non_thinking":
            prompt = self._create_prompt(item, thinking_mode=False)
            response = self._create_response(item, thinking_mode=False)
            mode_label = 0
        else:  # both
            # 随机选择思考或非思考模式
            thinking_mode = torch.rand(1).item() > 0.5
            prompt = self._create_prompt(item, thinking_mode=thinking_mode)
            response = self._create_response(item, thinking_mode=thinking_mode)
            mode_label = 1 if thinking_mode else 0
        
        # 组合完整文本
        full_text = prompt + response
        
        # 分别编码prompt和完整文本
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        # 计算响应部分的位置
        prompt_length = prompt_encoding['input_ids'].shape[1]
        full_length = full_encoding['input_ids'].shape[1]
        
        return {
            'input_ids': full_encoding['input_ids'].squeeze(0),
            'attention_mask': full_encoding['attention_mask'].squeeze(0),
            'prompt_length': prompt_length,
            'full_length': full_length,
            'mode_label': mode_label,
            'original_text': full_text
        }


def create_dataloader(
    data_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    mode: str = "both",
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """创建数据加载器"""
    
    dataset = DistillationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        mode=mode
    )
    
    def collate_fn(batch):
        """批处理函数"""
        # 找到批次中的最大长度
        max_len = max([item['input_ids'].shape[0] for item in batch])
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_prompt_lengths = []
        batch_mode_labels = []
        batch_texts = []
        
        for item in batch:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            
            # 填充到最大长度
            pad_length = max_len - input_ids.shape[0]
            if pad_length > 0:
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_length,), tokenizer.pad_token_id, dtype=input_ids.dtype)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_length, dtype=attention_mask.dtype)
                ])
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_prompt_lengths.append(item['prompt_length'])
            batch_mode_labels.append(item['mode_label'])
            batch_texts.append(item['original_text'])
        
        return {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
            'prompt_lengths': torch.tensor(batch_prompt_lengths),
            'mode_labels': torch.tensor(batch_mode_labels),
            'texts': batch_texts
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def test_dataloader():
    """测试数据加载器"""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataloader = create_dataloader(
        data_path="dataset/train-00000-of-00001-cae87f8e074b4b5d.json",
        tokenizer=tokenizer,
        batch_size=2,
        max_length=1024,
        mode="both"
    )
    
    for batch in dataloader:
        print("Batch shape:", batch['input_ids'].shape)
        print("Prompt lengths:", batch['prompt_lengths'])
        print("Mode labels:", batch['mode_labels'])
        print("Sample text:", batch['texts'][0][:200] + "...")
        break


if __name__ == "__main__":
    test_dataloader() 