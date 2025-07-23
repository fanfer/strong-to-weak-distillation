# Qwen3 Strong-to-Weak Distillation

基于 Transformers 实现的 Qwen3 Strong-to-Weak 蒸馏训练框架，支持两阶段蒸馏策略：非策略蒸馏和策略训练蒸馏。

## 项目概述

该项目实现了 Qwen3 的 Strong-to-Weak Distillation 策略，将大型教师模型（Qwen3-235B-A22B）的知识蒸馏到轻量级学生模型（Qwen3-8B）中。整个蒸馏过程分为两个主要阶段：

1. **非策略蒸馏阶段**：结合教师模型在"思考"和"不思考"模式下生成的输出，进行响应蒸馏，帮助学生模型发展基本推理能力和模式切换能力。

2. **策略训练蒸馏阶段**：通过最小化 KL 散度来对齐学生模型和教师模型的对数概率分布，进一步优化学生模型的性能。

## 主要特性

- ✅ **两阶段蒸馏策略**：完整实现非策略蒸馏和策略训练蒸馏
- ✅ **模式切换能力**：支持思考模式和非思考模式的动态切换
- ✅ **内存优化**：支持 4bit/8bit 量化，适应有限GPU资源
- ✅ **灵活配置**：YAML/JSON 配置文件，支持各种训练参数调节
- ✅ **训练监控**：集成 Wandb 和 TensorBoard，实时监控训练过程
- ✅ **模块化设计**：清晰的代码结构，易于扩展和修改

## 项目结构

```
strong2weak/
├── dataset/                          # 数据集目录
│   └── train-00000-of-00001-cae87f8e074b4b5d.json
├── data_loader.py                    # 数据加载和预处理
├── model_manager.py                  # 模型管理（教师/学生模型）
├── distillation_trainer.py          # 非策略蒸馏训练器
├── policy_distillation_trainer.py   # 策略训练蒸馏器
├── main_distillation.py             # 主训练脚本
├── requirements.txt                  # 项目依赖
├── README.md                        # 项目文档
└── config/                          # 配置文件目录
    └── distillation_config.yaml     # 默认配置文件
```

## 安装说明

### 环境要求

- Python 3.8+
- CUDA 11.7+ (推荐)
- GPU 内存: 40GB+ (教师模型) + 16GB+ (学生模型)

### 安装步骤

1. 克隆项目：
```bash
git clone <repository-url>
cd strong2weak
```

2. 创建虚拟环境：
```bash
conda create -n qwen3-distillation python=3.9
conda activate qwen3-distillation
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 安装 Flash Attention (可选，提升性能)：
```bash
pip install flash-attn --no-build-isolation
```

## 快速开始

### 1. 创建配置文件

```bash
python main_distillation.py --create_config
```

这将在 `config/distillation_config.yaml` 创建默认配置文件。

### 2. 测试模型加载

```bash
python main_distillation.py --stage test
```

### 3. 开始完整训练

```bash
# 运行完整的两阶段蒸馏
python main_distillation.py --stage all

# 或者分阶段运行
python main_distillation.py --stage non_policy  # 非策略蒸馏
python main_distillation.py --stage policy      # 策略蒸馏
```

## 配置说明

### 主要配置参数

```yaml
models:
  teacher_model: "Qwen/Qwen3-235B-A22B"    # 教师模型
  student_model: "Qwen/Qwen3-8B"           # 学生模型
  load_in_4bit: true                       # 4bit量化
  torch_dtype: "bfloat16"                  # 数据类型

data:
  data_path: "dataset/train-00000-of-00001-cae87f8e074b4b5d.json"
  max_length: 2048                         # 最大序列长度
  val_ratio: 0.1                          # 验证集比例

non_policy_distillation:
  num_epochs: 3                            # 训练轮数
  batch_size: 4                            # 批大小
  learning_rate: 5e-5                      # 学习率
  temperature: 3.0                         # 蒸馏温度
  alpha: 0.7                              # 蒸馏损失权重
  beta: 0.3                               # 原始损失权重

policy_distillation:
  num_epochs: 5                            # 训练轮数
  learning_rate: 1e-5                      # 更小的学习率
  temperature: 2.0                         # 策略训练温度
  thinking_prob: 0.5                       # 思考模式概率
```

### 自定义配置

你可以根据需要修改配置文件：

1. **硬件限制**：调整 `batch_size`、`gradient_accumulation_steps`
2. **训练策略**：修改学习率、训练轮数、温度参数
3. **模型选择**：更换教师模型和学生模型

## 使用示例

### 基本训练

```bash
# 使用默认配置进行完整训练
python main_distillation.py

# 使用自定义配置
python main_distillation.py --config my_config.yaml

# 启用 Wandb 监控
# 修改配置文件中的 use_wandb: true
```

### 仅运行特定阶段

```bash
# 只运行非策略蒸馏
python main_distillation.py --stage non_policy

# 使用预训练的学生模型运行策略蒸馏
python main_distillation.py --stage policy --pretrained_student ./outputs/non_policy_distillation/best_model
```

### 继续训练

```bash
# 从检查点继续训练
python main_distillation.py --stage policy --pretrained_student ./outputs/non_policy_distillation/checkpoint-1000
```

## 输出说明

训练完成后，你将得到以下输出：

```
outputs/
├── non_policy_distillation/           # 非策略蒸馏结果
│   ├── best_model/                    # 最佳模型
│   ├── final_model/                   # 最终模型
│   └── checkpoint-*/                  # 训练检查点
└── policy_distillation/               # 策略蒸馏结果
    ├── best_policy_model/             # 最佳策略模型
    ├── final_policy_model/            # 最终策略模型
    └── policy_checkpoint-*/           # 策略训练检查点
```

每个模型目录包含：
- `pytorch_model.bin`: 模型权重
- `config.json`: 模型配置
- `tokenizer.json`: 分词器
- `*_config.json`: 训练配置

## 性能优化

### 内存优化

1. **量化加载**：
   - 4bit量化：`load_in_4bit: true`
   - 8bit量化：`load_in_8bit: true`

2. **梯度累积**：
   ```yaml
   gradient_accumulation_steps: 8  # 增加此值以减少内存使用
   ```

3. **序列长度**：
   ```yaml
   max_length: 1024  # 减少序列长度
   ```

### 训练加速

1. **Flash Attention**：
   ```bash
   pip install flash-attn
   ```

2. **并行训练**：
   ```bash
   torchrun --nproc_per_node=2 main_distillation.py
   ```

## 监控和调试

### Wandb 集成

1. 设置配置：
   ```yaml
   training:
     use_wandb: true
     wandb_project: "qwen3-distillation"
   ```

2. 登录 Wandb：
   ```bash
   wandb login
   ```

### 日志查看

训练日志保存在 `distillation.log`，包含详细的训练信息。

### 常见问题排查

1. **内存不足**：
   - 减少 `batch_size`
   - 增加 `gradient_accumulation_steps`
   - 启用更激进的量化

2. **训练不稳定**：
   - 降低学习率
   - 调整温度参数
   - 检查数据质量

3. **模型加载失败**：
   - 检查网络连接
   - 确认模型名称正确
   - 尝试使用 HuggingFace Hub 镜像

## 进阶使用

### 自定义数据集

1. 数据格式：
   ```json
   [
     {
       "input": "输入文本",
       "output": "期望输出", 
       "instruction": "任务指令"
     }
   ]
   ```

2. 修改 `data_loader.py` 中的数据处理逻辑

### 模型适配

1. 添加新的教师/学生模型支持
2. 修改 `model_manager.py` 中的模型加载逻辑
3. 调整提示词模板

### 蒸馏策略定制

1. 修改损失函数权重
2. 添加新的正则化项
3. 实现自定义采样策略

## 致谢

本项目基于以下开源项目：

- [Transformers](https://github.com/huggingface/transformers)
- [Accelerate](https://github.com/huggingface/accelerate) 
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。 