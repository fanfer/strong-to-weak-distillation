"""
主运行脚本
整合VLLM推理、评估、中间结果保存和最终报告生成
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import argparse
import logging
from collections import defaultdict

from vllm_inference import VLLMInferenceEngine, InferenceConfig
from evaluation import EvaluationEngine, EvaluationMetrics

class PipelineManager:
    """推理评估流水线管理器"""
    
    def __init__(self, config: InferenceConfig, output_dir: str = "results", badcase_threshold: float = 0.5):
        self.inference_engine = VLLMInferenceEngine(config)
        self.evaluation_engine = EvaluationEngine(config.model_name)
        self.output_dir = output_dir
        self.badcase_threshold = badcase_threshold
        self.logger = self._setup_logger()
        
        # 创建输出目录结构
        self.create_output_structure()
        
        # 存储所有结果用于最终报告
        self.all_metrics = {}
        self.all_badcases = {}
        
    def _setup_logger(self) -> logging.Logger:
        """设置主日志器"""
        logger = logging.getLogger("Pipeline")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 控制台输出
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # 文件输出
            os.makedirs(self.output_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(self.output_dir, 'pipeline.log'),
                encoding='utf-8'
            )
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def create_output_structure(self):
        """创建输出目录结构"""
        subdirs = [
            'inference_results',  # 推理结果
            'evaluation_results',  # 评估结果
            'badcases',  # badcase文件
            'reports'  # 最终报告
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        
        self.logger.info(f"Created output structure in {self.output_dir}")
    
    def find_jsonl_files(self, dataset_dir: str) -> Dict[str, List[str]]:
        """查找所有jsonl文件"""
        dataset_path = Path(dataset_dir)
        files = {
            'test': [],
            'val': []
        }
        
        for split in ['test', 'val']:
            split_dir = dataset_path / split
            if split_dir.exists():
                jsonl_files = list(split_dir.glob('*.jsonl'))
                files[split] = [str(f) for f in jsonl_files]
                self.logger.info(f"Found {len(jsonl_files)} jsonl files in {split}")
        
        return files
    
    def process_single_file(self, file_path: str, split: str) -> Dict[str, Any]:
        """处理单个文件"""
        self.logger.info(f"Processing {file_path}")
        start_time = time.time()
        
        # 1. 推理
        inference_results = self.inference_engine.process_jsonl_file(file_path)
        
        # 2. 保存推理结果
        file_name = Path(file_path).stem
        inference_output_path = os.path.join(
            self.output_dir, 'inference_results', f"{split}_{file_name}_inference.jsonl"
        )
        self.inference_engine.save_intermediate_results(inference_results, inference_output_path)
        
        # 3. 评估
        metrics = self.evaluation_engine.evaluate_file_results(inference_results)
        
        # 4. 保存评估结果
        evaluation_output_path = os.path.join(
            self.output_dir, 'evaluation_results', f"{split}_{file_name}_evaluation.json"
        )
        self.evaluation_engine.save_evaluation_results(
            metrics, file_path, evaluation_output_path
        )
        
        # 5. 对val数据保存badcase
        badcases = []
        if split == 'val':
            badcases = self.evaluation_engine.find_badcases(inference_results, self.badcase_threshold)
            if badcases:
                badcase_output_path = os.path.join(
                    self.output_dir, 'badcases', f"{file_name}_badcases.jsonl"
                )
                self.evaluation_engine.save_badcases(badcases, badcase_output_path)
        
        processing_time = time.time() - start_time
        
        result = {
            'file_path': file_path,
            'split': split,
            'metrics': metrics,
            'badcase_count': len(badcases),
            'processing_time': processing_time,
            'inference_output_path': inference_output_path,
            'evaluation_output_path': evaluation_output_path
        }
        
        self.logger.info(f"Completed {file_path} in {processing_time:.2f}s - "
                        f"Accuracy: {metrics.score_accuracy:.3f}, "
                        f"Reason F1: {metrics.reason_token_f1:.3f}")
        
        return result
    
    def process_dataset(self, dataset_dir: str):
        """处理整个数据集"""
        self.logger.info("Starting dataset processing")
        
        # 查找所有文件
        files = self.find_jsonl_files(dataset_dir)
        total_files = sum(len(file_list) for file_list in files.values())
        
        if total_files == 0:
            self.logger.warning("No jsonl files found in dataset directory")
            return
        
        self.logger.info(f"Total files to process: {total_files}")
        
        # 处理每个文件
        processed_count = 0
        start_time = time.time()
        
        for split, file_list in files.items():
            split_results = []
            split_badcases = []
            
            for file_path in file_list:
                try:
                    result = self.process_single_file(file_path, split)
                    split_results.append(result)
                    
                    # 收集结果用于最终报告
                    if split not in self.all_metrics:
                        self.all_metrics[split] = []
                    self.all_metrics[split].append(result)
                    
                    processed_count += 1
                    self.logger.info(f"Progress: {processed_count}/{total_files} files completed")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {e}")
                    continue
        
        total_time = time.time() - start_time
        self.logger.info(f"Dataset processing completed in {total_time:.2f}s")
        
        # 生成最终报告
        self.generate_final_report()
    
    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合指标"""
        if not results:
            return {}
        
        # 收集所有指标
        score_accuracies = []
        score_f1_macros = []
        score_f1_weighteds = []
        reason_token_f1s = []
        total_samples = 0
        valid_predictions = 0
        badcase_counts = []
        processing_times = []
        
        score_distributions = defaultdict(int)
        
        for result in results:
            metrics = result['metrics']
            score_accuracies.append(metrics.score_accuracy)
            score_f1_macros.append(metrics.score_f1_macro)
            score_f1_weighteds.append(metrics.score_f1_weighted)
            reason_token_f1s.append(metrics.reason_token_f1)
            total_samples += metrics.total_samples
            valid_predictions += metrics.valid_predictions
            badcase_counts.append(result.get('badcase_count', 0))
            processing_times.append(result['processing_time'])
            
            for score, count in metrics.score_distribution.items():
                score_distributions[score] += count
        
        # 计算平均值和其他统计信息
        aggregated = {
            'file_count': len(results),
            'total_samples': total_samples,
            'valid_predictions': valid_predictions,
            'avg_score_accuracy': sum(score_accuracies) / len(score_accuracies),
            'avg_score_f1_macro': sum(score_f1_macros) / len(score_f1_macros),
            'avg_score_f1_weighted': sum(score_f1_weighteds) / len(score_f1_weighteds),
            'avg_reason_token_f1': sum(reason_token_f1s) / len(reason_token_f1s),
            'total_badcases': sum(badcase_counts),
            'total_processing_time': sum(processing_times),
            'avg_processing_time_per_file': sum(processing_times) / len(processing_times),
            'score_distribution': dict(score_distributions),
            'per_file_metrics': {
                'score_accuracies': score_accuracies,
                'score_f1_macros': score_f1_macros,
                'score_f1_weighteds': score_f1_weighteds,
                'reason_token_f1s': reason_token_f1s
            }
        }
        
        return aggregated
    
    def generate_final_report(self):
        """生成最终评估报告"""
        self.logger.info("Generating final evaluation report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_results': {}
        }
        
        overall_summary = {
            'total_files': 0,
            'total_samples': 0,
            'total_valid_predictions': 0,
            'total_badcases': 0,
            'total_processing_time': 0
        }
        
        # 处理每个split的结果
        for split, results in self.all_metrics.items():
            if not results:
                continue
            
            aggregated = self.aggregate_metrics(results)
            report['detailed_results'][split] = aggregated
            
            # 更新总体摘要
            overall_summary['total_files'] += aggregated['file_count']
            overall_summary['total_samples'] += aggregated['total_samples']
            overall_summary['total_valid_predictions'] += aggregated['valid_predictions']
            overall_summary['total_badcases'] += aggregated['total_badcases']
            overall_summary['total_processing_time'] += aggregated['total_processing_time']
        
        report['summary'] = overall_summary
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'reports', 'final_evaluation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成可读性更好的报告
        self.generate_readable_report(report)
        
        self.logger.info(f"Final report saved to {report_path}")
    
    def generate_readable_report(self, report: Dict[str, Any]):
        """生成可读性更好的报告"""
        readable_report_path = os.path.join(self.output_dir, 'reports', 'evaluation_summary.txt')
        
        with open(readable_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("VLLM 推理评估报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"生成时间: {report['timestamp']}\n\n")
            
            # 总体摘要
            f.write("总体摘要:\n")
            f.write("-" * 40 + "\n")
            summary = report['summary']
            f.write(f"总文件数: {summary['total_files']}\n")
            f.write(f"总样本数: {summary['total_samples']}\n")
            f.write(f"有效预测数: {summary['total_valid_predictions']}\n")
            f.write(f"总badcase数: {summary['total_badcases']}\n")
            f.write(f"总处理时间: {summary['total_processing_time']:.2f}秒\n\n")
            
            # 详细结果
            for split, results in report['detailed_results'].items():
                f.write(f"{split.upper()} 数据集结果:\n")
                f.write("-" * 40 + "\n")
                f.write(f"文件数: {results['file_count']}\n")
                f.write(f"样本数: {results['total_samples']}\n")
                f.write(f"有效预测数: {results['valid_predictions']}\n")
                f.write(f"平均Score准确率: {results['avg_score_accuracy']:.4f}\n")
                f.write(f"平均Score F1 (macro): {results['avg_score_f1_macro']:.4f}\n")
                f.write(f"平均Score F1 (weighted): {results['avg_score_f1_weighted']:.4f}\n")
                f.write(f"平均Reason Token F1: {results['avg_reason_token_f1']:.4f}\n")
                f.write(f"Badcase数: {results['total_badcases']}\n")
                f.write(f"处理时间: {results['total_processing_time']:.2f}秒\n")
                f.write(f"平均每文件处理时间: {results['avg_processing_time_per_file']:.2f}秒\n")
                
                # Score分布
                f.write("\nScore分布:\n")
                for score in sorted(results['score_distribution'].keys()):
                    count = results['score_distribution'][score]
                    f.write(f"  Score {score}: {count}\n")
                
                f.write("\n")
        
        self.logger.info(f"Readable report saved to {readable_report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VLLM推理评估流水线")
    
    # 配置文件选项
    parser.add_argument("--config", type=str, default=None,
                       help="从配置文件加载参数（如：--config config.py）")
    
    # 基本设置
    parser.add_argument("--dataset_dir", type=str, default="dataset", 
                       help="数据集目录路径")
    parser.add_argument("--output_dir", type=str, default="results", 
                       help="输出目录路径")
    
    # 模型设置
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-4B-Instruct",
                       help="模型名称或路径")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="张量并行大小（GPU数量）")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU内存利用率 (0.0-1.0)")
    parser.add_argument("--max_model_len", type=int, default=4096,
                       help="模型最大序列长度")
    
    # 推理设置
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="批处理大小")
    parser.add_argument("--temperature", type=float, default=0.1, 
                       help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="nucleus采样参数")
    parser.add_argument("--max_tokens", type=int, default=512, 
                       help="最大生成token数")
    
    # 评估设置
    parser.add_argument("--badcase_threshold", type=float, default=0.5,
                       help="badcase检测的reason F1阈值")
    
    args = parser.parse_args()
    
    # 如果指定了配置文件，则从配置文件加载参数
    if args.config:
        try:
            import importlib.util
            import sys
            
            # 动态导入配置文件
            spec = importlib.util.spec_from_file_location("config", args.config)
            config_module = importlib.util.module_from_spec(spec)
            sys.modules["config"] = config_module
            spec.loader.exec_module(config_module)
            
            # 从配置文件覆盖参数（只覆盖用户未在命令行指定的参数）
            if hasattr(config_module, 'MODEL_NAME') and not any('--model_name' in arg for arg in sys.argv):
                args.model_name = config_module.MODEL_NAME
            if hasattr(config_module, 'DATASET_DIR') and not any('--dataset_dir' in arg for arg in sys.argv):
                args.dataset_dir = config_module.DATASET_DIR
            if hasattr(config_module, 'OUTPUT_DIR') and not any('--output_dir' in arg for arg in sys.argv):
                args.output_dir = config_module.OUTPUT_DIR
            if hasattr(config_module, 'TENSOR_PARALLEL_SIZE') and not any('--tensor_parallel_size' in arg for arg in sys.argv):
                args.tensor_parallel_size = config_module.TENSOR_PARALLEL_SIZE
            if hasattr(config_module, 'GPU_MEMORY_UTILIZATION') and not any('--gpu_memory_utilization' in arg for arg in sys.argv):
                args.gpu_memory_utilization = config_module.GPU_MEMORY_UTILIZATION
            if hasattr(config_module, 'MAX_MODEL_LEN') and not any('--max_model_len' in arg for arg in sys.argv):
                args.max_model_len = config_module.MAX_MODEL_LEN
            if hasattr(config_module, 'BATCH_SIZE') and not any('--batch_size' in arg for arg in sys.argv):
                args.batch_size = config_module.BATCH_SIZE
            if hasattr(config_module, 'TEMPERATURE') and not any('--temperature' in arg for arg in sys.argv):
                args.temperature = config_module.TEMPERATURE
            if hasattr(config_module, 'TOP_P') and not any('--top_p' in arg for arg in sys.argv):
                args.top_p = config_module.TOP_P
            if hasattr(config_module, 'MAX_TOKENS') and not any('--max_tokens' in arg for arg in sys.argv):
                args.max_tokens = config_module.MAX_TOKENS
            if hasattr(config_module, 'BADCASE_THRESHOLD') and not any('--badcase_threshold' in arg for arg in sys.argv):
                args.badcase_threshold = config_module.BADCASE_THRESHOLD
                
            print(f"✅ 已从配置文件 {args.config} 加载参数")
            
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            print("使用命令行参数或默认值")
    
    # 创建配置
    config = InferenceConfig(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size
    )
    
    # 创建并运行流水线
    try:
        pipeline = PipelineManager(config, args.output_dir, args.badcase_threshold)
        pipeline.process_dataset(args.dataset_dir)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 