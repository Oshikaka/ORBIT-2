# ORBIT-2 Checkpoints 速查

**总路径**：`/lustre/orion/csc662/proj-shared/xinru/models/orbit2/`

来源：HuggingFace `jychoi-hpc/ORBIT-2`，commit `718625a`（2026-02-20），全仓库 26 个文件 / 8.1 GB，
已用 sha256 校验与线上一致。重下：`hf download jychoi-hpc/ORBIT-2 --local-dir <目标>`

---

## 微调模型（可直接推理）

| 文件 | 大小 | 参数 | epoch | embed/blocks | 用途 |
|---|---|---|---|---|---|
| `us-finetune/us_9.5m_precipitation.ckpt` | 35 MB | 7.16M | 29 | 256 / 6 | 美国区域降水，轻量 |
| `us-finetune/us_9.5m_temperature.ckpt` | 35 MB | 7.16M | 29 | 256 / 6 | 美国区域温度，轻量 |
| `us-finetune/us_126m_precipitation.ckpt` | 557 MB | 116.75M | 77 | 1024 / 8 | 美国区域降水，主力 |
| `us-finetune/us_126m_temperature.ckpt` | 557 MB | 116.75M | 40 | 1024 / 8 | 美国区域温度，主力 |
| `global-finetune/global_9.5m_precipitation.ckpt` | 42 MB | 9.65M | 6 | 256 / 6 | 全球降水，轻量（epoch 少，慎用） |
| `global-finetune/global_126m_precipitation.ckpt` | 605 MB | 126.71M | 69 | 1024 / 8 | 全球降水，主力 |

每个 ckpt 都有同名 `.yaml`，是该模型训练时的原始配置，跑推理直接用它最稳妥。

**US 版和 global 版的唯一区别是 `pos_embed`**：US 是 7200 tokens（区域网格），global 是 16928
（全球网格），其余 168 个张量结构完全相同。参数量差值也正好对得上
（126.71M − 116.75M = 9.96M = (16928−7200) × 1024）。**所以两者不能互换**，网格对不上会直接加载失败。

## 预训练模型（微调的起点）

| 文件 | 大小 | 参数 | epoch | embed/blocks | 备注 |
|---|---|---|---|---|---|
| `pretrain/intermediate_8m.ckpt` | 37 MB | 9.50M | 10 | 256 / 6 | pos_embed 16200 tokens（1.0° 网格） |
| `pretrain/intermediate_117m.ckpt` | 497 MB | 126.10M | 3 | 1024 / 8 | 同上网格；名字叫 117m，实际 126.10M |
| `pretrain/intermediate_1b.ckpt_rank_0..3` | 1.39 GB ×4 | 278.46M ×4 | 29 | — | **4 路 FSDP 分片**，合计约 1.11B |

`intermediate_1b` 必须用 `fsdp: 4` 加载，单卡读单个分片没有意义。

## 配套数据

- `static_variables/` — 海陆掩膜、地形、landcover、纬度，均 (720, 1440) 的 0.25° 全球网格，共 20 MB
- `mean_std/era5/0.25_deg/` — 归一化 mean/std，各含 79 个变量

---

## 已接好的配置

- `configs/inference.yaml` → `global_126m_precipitation.ckpt`
  （配置是 1024/8 + INFER 28km + 降水，三项都指向全球 126M）
- `configs/interm_fine_tune_template.yaml` → `pretrain/intermediate_8m.ckpt`
  （同时把 `decoder_depth` 由 2 改为 4——全部 HF 权重的 head 都是 5 层，2 会加载失败）

**坑**：所有 HF 权重的 `decoder_depth` 都是 4。自己写配置时若填 2，加载会报形状不匹配。

**坑**：微调模板换数据集后若网格变了，`pos_embed` 形状会对不上（预训练是 16200 tokens）。
真跑之前先在 debug 队列试一个 epoch。
