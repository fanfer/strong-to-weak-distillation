"""
评估模块
计算score分类准确性、F1分数、reason的逐token F1分数
"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
import logging
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from transformers import AutoTokenizer

@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    score_accuracy: float
    score_f1_macro: float
    score_f1_weighted: float
    reason_token_f1: float
    total_samples: int
    valid_predictions: int
    score_distribution: Dict[int, int]
    detailed_report: str

class TextTokenizer:
    """文本分词器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-4B-Instruct"):
        # 初始化transformers tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            # 如果模型不可用，回退到通用中文tokenizer
            print(f"Warning: Failed to load tokenizer {model_name}, falling back to bert-base-chinese: {e}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            except Exception as e2:
                print(f"Warning: Failed to load bert-base-chinese, using basic tokenization: {e2}")
                self.tokenizer = None
    
    def tokenize(self, text: str) -> List[str]:
        """对文本进行分词"""
        if not text or not isinstance(text, str):
            return []
        
        # 清理文本
        text = text.strip()
        if not text:
            return []
        
        if self.tokenizer is not None:
            try:
                # 使用transformers tokenizer
                # 先编码再解码，获取子词tokens
                tokens = self.tokenizer.tokenize(text)
                
                # 过滤特殊token和空token
                tokens = [token for token in tokens if token and not token.startswith('[') and not token.startswith('<')]
                
                return tokens
            except Exception as e:
                print(f"Warning: Tokenization failed, using basic fallback: {e}")
        
        # 基础分词回退方案（按字符和标点分割）
        import re
        # 使用正则表达式进行基础分词
        tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+|[^\s\u4e00-\u9fff\w]', text)
        tokens = [token.strip() for token in tokens if token.strip()]
        
        return tokens

class EvaluationEngine:
    """评估引擎"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-4B-Instruct"):
        self.logger = self._setup_logger()
        self.tokenizer = TextTokenizer(model_name)
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("Evaluation")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def extract_true_output(self, original_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提取真实标签"""
        output = original_data.get('output')
        if not output:
            return None
        
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                return None
        
        if isinstance(output, dict) and 'score' in output and 'reason' in output:
            try:
                score = int(output['score'])
                if 0 <= score <= 9:
                    return {
                        'score': score,
                        'reason': str(output['reason'])
                    }
            except (ValueError, TypeError):
                pass
        
        return None
    
    def calculate_score_metrics(self, true_scores: List[int], 
                              pred_scores: List[int]) -> Tuple[float, float, float, str]:
        """计算score相关指标"""
        if not true_scores or not pred_scores or len(true_scores) != len(pred_scores):
            return 0.0, 0.0, 0.0, ""
        
        # 准确率
        accuracy = accuracy_score(true_scores, pred_scores)
        
        # F1分数
        f1_macro = f1_score(true_scores, pred_scores, average='macro', zero_division=0)
        f1_weighted = f1_score(true_scores, pred_scores, average='weighted', zero_division=0)
        
        # 详细报告
        try:
            report = classification_report(
                true_scores, pred_scores, 
                labels=list(range(10)),  # 0-9分类
                zero_division=0,
                output_dict=False
            )
        except Exception as e:
            self.logger.warning(f"Failed to generate classification report: {e}")
            report = "Classification report generation failed"
        
        return accuracy, f1_macro, f1_weighted, report
    
    def calculate_token_f1(self, true_text: str, pred_text: str) -> float:
        """计算两个文本之间的token级F1分数"""
        true_tokens = set(self.tokenizer.tokenize(true_text))
        pred_tokens = set(self.tokenizer.tokenize(pred_text))
        
        if not true_tokens and not pred_tokens:
            return 1.0  # 都为空，认为完全匹配
        
        if not true_tokens or not pred_tokens:
            return 0.0  # 一个为空一个不为空
        
        # 计算交集
        intersection = true_tokens & pred_tokens
        
        # 计算精确率和召回率
        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
        recall = len(intersection) / len(true_tokens) if true_tokens else 0
        
        # 计算F1分数
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def calculate_reason_metrics(self, true_reasons: List[str], 
                               pred_reasons: List[str]) -> float:
        """计算reason的平均token F1分数"""
        if not true_reasons or not pred_reasons or len(true_reasons) != len(pred_reasons):
            return 0.0
        
        f1_scores = []
        for true_reason, pred_reason in zip(true_reasons, pred_reasons):
            f1 = self.calculate_token_f1(true_reason, pred_reason)
            f1_scores.append(f1)
        
        return np.mean(f1_scores) if f1_scores else 0.0
    
    def find_badcases(self, evaluation_data: List[Dict[str, Any]], 
                     threshold: float = 0.5) -> List[Dict[str, Any]]:
        """找出badcase（分数错误的案例）"""
        badcases = []
        
        for item in evaluation_data:
            original_data = item.get('original_data', {})
            predicted_output = item.get('parsed_output')
            
            if not predicted_output:
                # 解析失败的案例
                badcases.append({
                    'type': 'parse_failed',
                    'original_data': original_data,
                    'generated_text': item.get('generated_text', ''),
                    'issue': 'Failed to parse generated output'
                })
                continue
            
            true_output = self.extract_true_output(original_data)
            if not true_output:
                continue
            
            # 检查score是否错误
            true_score = true_output['score']
            pred_score = predicted_output.get('score')
            
            score_correct = (pred_score == true_score)
            
            # 检查reason的质量
            true_reason = true_output['reason']
            pred_reason = predicted_output.get('reason', '')
            reason_f1 = self.calculate_token_f1(true_reason, pred_reason)
            
            # 如果分数错误或reason质量太低，认为是badcase
            if not score_correct or reason_f1 < threshold:
                badcase = {
                    'type': 'prediction_error',
                    'original_data': original_data,
                    'generated_text': item.get('generated_text', ''),
                    'true_score': true_score,
                    'pred_score': pred_score,
                    'true_reason': true_reason,
                    'pred_reason': pred_reason,
                    'score_correct': score_correct,
                    'reason_f1': reason_f1,
                    'issues': []
                }
                
                if not score_correct:
                    badcase['issues'].append(f'Score mismatch: expected {true_score}, got {pred_score}')
                
                if reason_f1 < threshold:
                    badcase['issues'].append(f'Low reason quality: F1={reason_f1:.3f}')
                
                badcases.append(badcase)
        
        return badcases
    
    def evaluate_file_results(self, results: List[Dict[str, Any]]) -> EvaluationMetrics:
        """评估单个文件的结果"""
        self.logger.info(f"Evaluating {len(results)} samples")
        
        true_scores = []
        pred_scores = []
        true_reasons = []
        pred_reasons = []
        valid_predictions = 0
        score_distribution = Counter()
        
        for item in results:
            original_data = item.get('original_data', {})
            predicted_output = item.get('parsed_output')
            
            # 提取真实标签
            true_output = self.extract_true_output(original_data)
            if not true_output:
                continue
            
            true_score = true_output['score']
            true_reason = true_output['reason']
            
            # 提取预测结果
            if predicted_output and isinstance(predicted_output, dict):
                pred_score = predicted_output.get('score')
                pred_reason = predicted_output.get('reason', '')
                
                if pred_score is not None:
                    true_scores.append(true_score)
                    pred_scores.append(pred_score)
                    true_reasons.append(true_reason)
                    pred_reasons.append(pred_reason)
                    valid_predictions += 1
                    score_distribution[pred_score] += 1
        
        # 计算各项指标
        if true_scores:
            accuracy, f1_macro, f1_weighted, detailed_report = self.calculate_score_metrics(
                true_scores, pred_scores
            )
            reason_token_f1 = self.calculate_reason_metrics(true_reasons, pred_reasons)
        else:
            accuracy = f1_macro = f1_weighted = reason_token_f1 = 0.0
            detailed_report = "No valid predictions found"
        
        metrics = EvaluationMetrics(
            score_accuracy=accuracy,
            score_f1_macro=f1_macro,
            score_f1_weighted=f1_weighted,
            reason_token_f1=reason_token_f1,
            total_samples=len(results),
            valid_predictions=valid_predictions,
            score_distribution=dict(score_distribution),
            detailed_report=detailed_report
        )
        
        self.logger.info(f"Evaluation completed: Accuracy={accuracy:.3f}, "
                        f"F1_macro={f1_macro:.3f}, Reason_F1={reason_token_f1:.3f}")
        
        return metrics
    
    def save_badcases(self, badcases: List[Dict[str, Any]], output_path: str):
        """保存badcase"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for badcase in badcases:
                f.write(json.dumps(badcase, ensure_ascii=False, indent=2) + '\n')
        
        self.logger.info(f"Saved {len(badcases)} badcases to {output_path}")
    
    def save_evaluation_results(self, metrics: EvaluationMetrics, 
                              file_path: str, output_path: str):
        """保存评估结果"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results = {
            'file_path': file_path,
            'metrics': {
                'score_accuracy': metrics.score_accuracy,
                'score_f1_macro': metrics.score_f1_macro,
                'score_f1_weighted': metrics.score_f1_weighted,
                'reason_token_f1': metrics.reason_token_f1,
                'total_samples': metrics.total_samples,
                'valid_predictions': metrics.valid_predictions,
                'score_distribution': metrics.score_distribution
            },
            'detailed_report': metrics.detailed_report
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved evaluation results to {output_path}")

def main():
    """测试函数"""
    evaluator = EvaluationEngine()
    
    # 测试token F1计算
    true_text = "代码功能正确，但缺少文档注释和错误处理"
    pred_text = "代码功能基本正确，缺少注释和异常处理"
    f1 = evaluator.calculate_token_f1(true_text, pred_text)
    print(f"Token F1: {f1:.3f}")
    
    # 测试分词
    tokens = evaluator.tokenizer.tokenize(true_text)
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")

if __name__ == "__main__":
    main() 