# TIR 数据工程

独立运行精确答案数学任务的数据工程流程：

```bash
python3 -m areal.v2.data_engineering \
  --config examples/data_engineering/tir/data_generation.yaml \
  --output /tmp/tir-data-job \
  --export-hf /tmp/tir-training-data
```

该 CPU 示例需要 AReaL 的常规 Python 依赖，输出目录必须尚不存在。`data_recipe.py` 实现生成、清洗、数值评分、筛选和
`messages`/`answer` 训练视图。`data_calibration.py` 将已有 TIR 评估报告转换为校准结果；自定义校准通过 Python
`Calibrator` 协议接入。完整约定见[数据工程协议与用法](../../../docs/zh/customization/data_engineering.md)。
