# BILSTM_CRF Torch Version

这是 `BILSTM_CRF` 的 PyTorch 版本。

## 依赖安装

```bash
pip install -r BILSTM_CRF/requirements.txt
```

## 生成伪造数据

```bash
python -m BILSTM_CRF.generate_fake_data
```

默认会生成到：

- `BILSTM_CRF/demo_data/train.json`
- `BILSTM_CRF/demo_data/dev.json`
- `BILSTM_CRF/demo_data/vocab.json`

## 兼容真实数据格式

支持两种输入：

1. `jsonl`

```json
{"text": "肯德基在哪里", "label": "B-BRD I-BRD I-BRD O O O"}
```

2. `conll/bio` 文本

```text
肯 B-BRD
德 I-BRD
基 I-BRD
在 O
哪 O
里 O
```

空行分隔样本。

如需把真实数据转成当前训练格式：

```bash
python -m BILSTM_CRF.dataset_adapters \
  --input your_dataset.txt \
  --output-dir BILSTM_CRF/real_data \
  --format conll
```

## 训练

```bash
python -m BILSTM_CRF.train \
  --vocab-file BILSTM_CRF/demo_data/vocab.json \
  --train-file BILSTM_CRF/demo_data/train.json \
  --train-format jsonl \
  --dev-file BILSTM_CRF/demo_data/dev.json \
  --dev-format jsonl \
  --model-dir BILSTM_CRF/demo_data/model \
  --epochs 5
```

## 评估

```bash
python -m BILSTM_CRF.evaluate \
  --vocab-file BILSTM_CRF/demo_data/vocab.json \
  --model-dir BILSTM_CRF/demo_data/model \
  --eval-file BILSTM_CRF/demo_data/dev.json \
  --eval-format jsonl
```

## 预测

```bash
python -m BILSTM_CRF.predict \
  --vocab-file BILSTM_CRF/demo_data/vocab.json \
  --model-path BILSTM_CRF/demo_data/model/ner.pt \
  --text "肯德基优惠券在哪里"
```

## 当前边界

- 目前已迁移 `BILSTM_CRF`
- 指标统计是 token 级别的 accuracy / precision / recall / f1
- 伪造数据只用于验证训练链路，不代表真实业务效果
