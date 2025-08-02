"""
配置文件
直接修改下面的参数来配置VLLM推理评估系统
"""

# =============================================================================
# 基本设置
# =============================================================================
DATASET_DIR = "dataset"          # 数据集目录路径
OUTPUT_DIR = "results"           # 输出目录路径

# =============================================================================
# 模型设置
# =============================================================================
MODEL_NAME = "Qwen/Qwen2.5-4B-Instruct"  # 模型名称或路径
TENSOR_PARALLEL_SIZE = 1                   # 张量并行大小（GPU数量）
GPU_MEMORY_UTILIZATION = 0.9              # GPU内存利用率 (0.0-1.0)
MAX_MODEL_LEN = 4096                       # 模型最大序列长度

# =============================================================================
# 推理设置
# =============================================================================
BATCH_SIZE = 32                   # 批处理大小
TEMPERATURE = 0.1                 # 采样温度
TOP_P = 0.9                       # nucleus采样参数
MAX_TOKENS = 512                  # 最大生成token数

# =============================================================================
# 评估设置
# =============================================================================
BADCASE_THRESHOLD = 0.5           # badcase检测的reason F1阈值

# =============================================================================
# 预设配置（可选择使用）
# 取消注释下面的配置之一来快速设置参数
# =============================================================================

# # 高性能配置 - 适合单GPU快速推理
# BATCH_SIZE = 64
# TEMPERATURE = 0.0
# MAX_TOKENS = 256
# GPU_MEMORY_UTILIZATION = 0.95

# # 多GPU配置 - 适合大模型并行推理
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# TENSOR_PARALLEL_SIZE = 2
# BATCH_SIZE = 32
# GPU_MEMORY_UTILIZATION = 0.9

# # 大模型配置 - 适合14B+模型
# MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
# TENSOR_PARALLEL_SIZE = 2
# MAX_MODEL_LEN = 8192
# BATCH_SIZE = 8
# MAX_TOKENS = 1024
# GPU_MEMORY_UTILIZATION = 0.95

# # 质量优先配置 - 生成高质量输出
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# TEMPERATURE = 0.3
# TOP_P = 0.95
# MAX_TOKENS = 1024
# BATCH_SIZE = 16
# BADCASE_THRESHOLD = 0.7

# # 内存受限配置 - 适合显存较小的GPU
# BATCH_SIZE = 8
# MAX_MODEL_LEN = 2048
# MAX_TOKENS = 256
# GPU_MEMORY_UTILIZATION = 0.7 