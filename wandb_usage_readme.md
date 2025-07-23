# Weights & Biases (wandb) 使用指南

## 正确配置 Tsinghua 账户

我们已经成功配置了环境，使其能够直接上传到您的 Tsinghua 大学账户。这是通过设置关键的环境变量和明确指定实体来实现的。

## 如何上传到您的 Tsinghua 账户

要将数据记录到您的 Tsinghua 大学账户，请使用以下模式:

```python
import wandb
import os

# 设置关键的 wandb 环境变量 (可选但推荐)
os.environ["WANDB_API_KEY"] = "9c973791b10e62adc1089ca11baa273755d50d7f"
os.environ["WANDB_ENTITY"] = "electrixoul-tsinghua-university"

# 初始化 wandb 运行，明确指定实体
run = wandb.init(
    entity="electrixoul-tsinghua-university",  # 关键 - 明确指定您的 Tsinghua 账户
    project="your-project-name",              # 您的项目名称
    name="your-run-name",                     # 描述性运行名称
    config={
        # ... 配置参数
    }
)

# 记录指标
wandb.log({
    "your_metric": value,
})

# 完成运行
wandb.finish()
```

## 已验证的测试脚本

本项目包含几个验证 wandb 功能的测试脚本:

1. `test_wandb.py` - 项目中找到的原始测试脚本（上传到 neuroevolution 组织）
2. `wandb_test_minimal.py` - 最小测试脚本（上传到 neuroevolution 组织）
3. `wandb_org_test.py` - 创建个人项目空间的脚本（仍在 neuroevolution 组织内）
4. `wandb_tsinghua_test.py` - **推荐的脚本**，正确上传到您的 Tsinghua 账户

## 查看您的运行

您的运行将在以下位置可用:
https://wandb.ai/electrixoul-tsinghua-university/[your-project-name]

例如，我们的测试运行位于:
https://wandb.ai/electrixoul-tsinghua-university/tsinghua-test/runs/xyafdtyv

## 关键因素

成功的关键是:

1. 设置正确的环境变量 (WANDB_API_KEY, WANDB_ENTITY)
2. 在 wandb.init() 调用中明确指定 entity 参数
