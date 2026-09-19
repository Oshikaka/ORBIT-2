# ORBIT-2 降尺度与预报流程调试记录

**日期**：2026-09-19
**平台**：ORNL Frontier，分支 `frontier-setup`
**目标**：用已下载的 HuggingFace checkpoint 把降尺度推理跑通，把 Sparse-Reslim 预报示例跑通，不从头训练。

---

## 0. 结论

两条流程现在都能跑。仓库根目录新增 `run_orbit2.sh` 作为统一入口：

```bash
./run_orbit2.sh              # 列出所有可跑的东西
./run_orbit2.sh us-precip    # 降尺度：美国降水，单卡 2-4 分钟出图
./run_orbit2.sh fc-smoke     # 预报：冒烟测试，20 秒，不需要数据
./run_orbit2.sh fc-train     # 预报：真实 ERA5 训练，3 分钟
./run_orbit2.sh env          # 环境自检
```

脚本在执行前会把真实命令打印出来，方便照抄后自己改参数。

---

## 1. 环境

```bash
module load miniforge3/23.11.0-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /lustre/orion/csc662/proj-shared/xinru/envs/orbit
```

python 3.11.16 / torch 2.8.0+rocm6.4 / pytorch-lightning 2.6.6 / numpy 2.4.6，
`climate_learn` 以 editable 模式装在本仓库。注意环境里**没有** `lightning` 包，只有 `pytorch_lightning`。

**Frontier 登录节点自带一张可用 GPU**（login08 实测 AMD Instinct MI210，`torch.cuda.device_count()==1`）。
所以单卡推理和小规模训练不用排队，直接在登录节点跑即可。这张卡是所有登录用户共享的，
同时几个进程会明显变慢（实测单次运行从 45 秒拉长到 4 分钟）。只有全球 126M 模型
（720×1440 → 2880×5760，16 块 tiling）需要走 `sbatch -A csc662 -q debug`。

---

## 2. 降尺度：四个已验证的组合

| 命令 | checkpoint | 数据（低分辨率 → 高分辨率） | PSNR / SSIM / RMSE |
|---|---|---|---|
| `us-precip` | `us_9.5m_precipitation` | `era5-usa/15.0_arcmin`（precip 通道覆盖自 `daymet/15.0_arcmin`）→ `daymet/3.75_arcmin` | 27.30 / 0.849 / 0.200 |
| `us-precip-126m` | `us_126m_precipitation` | 同上 | 30.05 / 0.937 / 0.146 |
| `us-temp` | `us_9.5m_temperature` | `daymet/15.0_arcmin` → `daymet/3.75_arcmin` | 25.71 / 0.922 / 4.34 K |
| `global-precip` | `global_126m_precipitation` | `ERA5-IMERG-FUSED/0.25_deg` → `IMERG/0.0625_deg` | 34.56 / 0.919 / 0.111 |

指标是 `visualize.py` 在 index 0 上打印的。降水的 RMSE 在 `log1p(mm/day)` 空间，温度的在 K。

产物在 `outputs/<名字>/`：

- `N_comparison.png` — 三联对比图（低分辨率输入 / ORBIT-2 降尺度结果 / 真值），先看这张
- `N_preds.npy` (480×960)、`N_truth.npy`、`N_input.npy` (120×240) — 原始数组
- `--index N` 换样本，测试集是 2021 全年，N 取 0–364

### 独立复跑验证

`us-precip --index 50` 与 `--index 200` 两次全新运行均成功，日志确认
`num_patches 7200` 与 checkpoint 的 `pos_embed` 精确匹配、`<All keys matched successfully>`，
PSNR 分别为 28.68 和 25.62。三联图经目视检查物理合理：美国域（lat 24–53°N、lon 235–295°E）
降水集中在太平洋西北岸，预测与真值吻合且纹理比输入更细。

---

## 3. 预报：Sparse-Reslim 示例

```bash
./run_orbit2.sh fc-smoke       # 合成张量，验证前向+反向
./run_orbit2.sh fc-train       # ERA5 5.625°，GPU，2 epoch 短跑
./run_orbit2.sh fc-train-cpu   # 同上但 CPU，35 秒，用来确认数据通路
```

冒烟测试输出：`Smoke test passed: forecast=(2, 1, 8, 16), sparse_tokens=8/32, loss=1.6330`

真实数据短跑输出：`train/mse_epoch` 0.161 → 0.0233，`val/mse` 0.020，`test/mse 0.020249169319868088`。
**两次独立运行的 `test/mse` 到小数点后 16 位完全一致**，确认 seed 修复后运行可复现。

推荐数据集 `/lustre/orion/lrn036/world-shared/data/superres/era5/5.625_deg`：32×64 网格
（patch_size=2 → 512 token），单卡 14 ms/batch，布局与示例要求逐项吻合、零改动可用。

### 单卡显存经验规律（MI210, 64 GB）

峰值显存 ≈ **32 KiB × (batch_size × token 数)**，几乎完美线性；
**OOM 门槛约在 `batch_size × num_patches ≳ 2.0×10⁶`**。显存与分辨率本身无关，
只与「batch × token」乘积有关（SDPA 走 memory-efficient backend，没有物化 L×L 注意力矩阵）。

| 网格 | patch | token | batch | 每 batch | 峰值显存 |
|---|---|---|---|---|---|
| 32×64 (5.625°) | 2 | 512 | 16 | 14.1 ms | 0.34 GiB |
| 32×64 | 2 | 512 | 256 | 94.0 ms | 4.11 GiB |
| 180×360 (1.0°) | 4 | 4050 | 16 | 191 ms | 2.29 GiB |
| 180×360 | 2 | 16200 | 96 | 6.64 s | 47.88 GiB |
| 720×1440 (0.25°) | 4 | 64800 | 1 | 1.57 s | 2.28 GiB |

5.625° 推荐生产配置：`--batch-size 64~256`、`--patch-size 2`。

**I/O 才是瓶颈，不是 GPU**：2 epoch 的 GPU 跑 wall time 2m48s，其中 GPU 计算只有约 24 秒，
其余是 dataloader worker 每个 epoch 重启后从 Lustre 重读整个变量数组。长跑建议 `--num-workers 4`，
且不要用太小的 `--limit-train-batches`。

---

## 4. 原来为什么跑不起来

### 降尺度 —— 三个都是静默失败类型

1. **`launch_visualize.sh` 默认指向 `configs/interm_8m_ft.yaml`，这个文件整个仓库里不存在**，
   脚本永远跑不起来。已改为 `CONFIG/OUTPUT_DIR/INDEX/VARIABLE/NTASKS` 环境变量驱动。
   另外 `conda activate orbit` 按名字激活会失败（该名字没注册到 conda envs），已改成完整前缀。

2. **数据路径全是空目录。** `configs/inference.yaml` 原先指向
   `kurihana/super-res-torchlight/superres/era5/0.25_deg_test` —— 整棵树是空的，且该目录树下
   没有任何替代数据。HuggingFace 上 4 个 US checkpoint 自带 yaml 指向的
   `kurihana/regridding/.../ERA5-Daymet-1dy-superres/` 同样已被清空（`train/` `val/` 都是 0 个文件，
   连 `normalize_mean.npz` 和 `lat.npy` 都没有）。**照 HF 的 yaml 原样跑必然失败。**

3. **`overlap: 2` 是错的，必须是 4。** `global_126m` 的 `pos_embed` 是 `(1, 16928, 1024)`；
   `div=4` 时 tile 尺寸 = `(720/4 + top+bottom) × (1440/4 + left+right)`，
   `overlap=4` → `184×368` → `(184/2)×(368/2) = 16928` 严丝合缝，`overlap=2` → `16562`。
   要命的是 `src/climate_learn/models/hub/res_slimvit.py:270` 的
   `interpolate_pos_embed_on_the_fly()` **token 数对不上不会报错，只会静默 bicubic 插值**，
   模型精度无声下降。**tiling 参数必须自己核对。**

### 降尺度 —— 数据溯源问题

US 模型的低分辨率**降水通道必须从 `daymet/15.0_arcmin` 读，而不是 ERA5**。
在 2021 测试集 59 天上做的 A/B/C 对照：

```
low-res precip 源      对齐日       PSNR     SSIM   RMSE(land, log1p)
A  ERA5-QM           daymet[t]   13.9727  0.6153         0.8218
B  ERA5-raw          daymet[t]   13.4266  0.4450         0.8193
C  daymet15（采用）   daymet[t]   27.8499  0.8387         0.1915
```

C 比其它方案高 **12 dB**。修正前后单样本指标：PSNR 11.85 → 27.30，SSIM 0.60 → 0.849。

根因是**累积窗口约定差异**：ERA5 和 Daymet 把同一场雨记在相差一天的位置上
（低分辨率 ERA5 precip[t] 与 daymet[t-1] 相关系数 0.83，与 daymet[t] 只有 0.33；温度没有这个偏移）。
这不是代码 bug，但会让人严重误判模型精度。

为此给 `IterDataModule` / `NpyReader` 加了可选的**逐变量数据源覆盖**
（配置项 `data.low_res_var_dirs`），因为 US 微调模型的输入是「ERA5 高空场 + Daymet 地面场」的融合数据，
而这两半在 Lustre 上是两个独立目录。默认 `None`，对现有训练路径零影响。

### 预报 —— 两个真 bug

1. **eval 模式下 token 路由是随机的。** `_route()` 不区分 train/eval，
   所以同一输入连续两次前向输出最大差 **1.39**（归一化单位）。对一个自称
   "deterministic forecasting" 的模型是硬伤：`val/mse`、`test/mse` 以及 `ModelCheckpoint`
   的 best 选择全都带噪，"最好的 epoch"很大程度上是运气。
   修法是 eval 时改用固定种子的 CPU generator 抽一份子集并 expand 到整个 batch，
   `num_keep` 与训练时一致（不退化成 dense）。修后实测两次前向最大差 **0.0**。

2. **多 worker 下文件分片错乱。** 原来是先 `random.shuffle(files)` 再
   `files[worker.id::num_workers]`。PyTorch 给每个 worker 不同的随机种子，
   各 worker 洗出的排列不同，再按 stride 切就会重叠+漏掉。实测 40 个 train 文件、
   `num_workers=4` 时：`assigned=40 unique=30` —— **10 个文件被重复训练，10 个从未被训练到**。
   修法是先用确定性的 `sorted()` 列表分片，再在各自分片内 shuffle。
   修后 `num_workers=0/2/4/8` 全部 `unique=40`。

另外还修了 6 处：`pl.seed_everything()` 调用位置在建模型之后（`--seed` 不影响权重初始化）、
`_load_normalization` 无脑取 `[0]`（多元素统计量会被静默截断）、
`--limit-train-batches` 不足以做短跑（val split 仍跑满，新增 `--limit-val-batches` / `--limit-test-batches`）、
`--limit-val-batches 0` 会崩、`trainer.test(ckpt_path="best")` 在没写出 checkpoint 时硬抛、
`--devices > 1` 会让每个 rank 静默训练同一批样本（现改为直接报错拦截）。

---

## 5. Checkpoint ↔ 数据对应表

所有 9 个 checkpoint 都用仓库里真正的 `Res_Slim_ViT` 按推断的 `img_size` 重建模型后
`load_state_dict(strict=True)` 验证通过，所以下面的网格不是推测。

| checkpoint | 参数量 | epoch | embed/depth | pos_embed token | 输入网格 | 出通道 |
|---|---|---|---|---|---|---|
| `us_9.5m_precipitation` | 7.16 M | 29 | 256 / 6 | 7200 | 120×240（无 tiling） | 1 |
| `us_9.5m_temperature` | 7.16 M | 29 | 256 / 6 | 7200 | 120×240 | 1 |
| `us_126m_precipitation` | 116.75 M | 77 | 1024 / 8 | 7200 | 120×240 | 1 |
| `us_126m_temperature` | 116.75 M | 40 | 1024 / 8 | 7200 | 120×240 | 1 |
| `global_9.5m_precipitation` | 9.65 M | 6 | 256 / 6 | 16928 | 184×368（720×1440 的 tile） | 1 |
| `global_126m_precipitation` | 126.71 M | 69 | 1024 / 8 | 16928 | 184×368 | 1 |
| `intermediate_8m` | 9.50 M | 10 | 256 / 6 | 16200 | 180×360（无 tiling） | 3 |
| `intermediate_117m` | 126.10 M | 3 | 1024 / 8 | 16200 | 180×360 | 3 |
| `intermediate_1b_rank_{0..3}` | 合并 986.37 M | 29 | 3072 / 8 | 1152/片 | 48×96（180×360 的 tile） | 3 |

token 数验算：

| 模型 | 计算 | 结果 |
|---|---|---|
| US | div=1，120×240 → 60×120 | 7200 |
| Global | 720×1440 div=4 → 180×360；overlap=4 → 184×368 → 92×184 | 16928 |
| pretrain 8m/117m | div=1，180×360 → 90×180 | 16200 |
| pretrain 1b | 180×360 div=4 → 45×90；overlap=3（奇）→ 48×96 → 24×48 | 1152 |

overlap 公式见 `src/climate_learn/data/iterdataset.py::calculate_tile_overlap`（经度方向 ×2）。

数据侧：`/lustre/orion/lrn036/world-shared/data/superres/` 下 88 个
`<dataset>/<resolution>` 目录中只有 `ERA5-1hr/1.0_deg` 是空的，其余 87 个结构完整。

---

## 6. 已知坑

1. **`interpolate_pos_embed_on_the_fly` 静默插值** —— 见上，最容易踩且最难发现。
2. **HF checkpoint 自带 yaml 的 `checkpoint` / `pretrain` 字段全部失效**，指向别人的
   proj-shared 或相对路径。推理必须用 `trainer.pretrain` 指向本地 ckpt。
3. **所有 HF 权重的 `decoder_depth` 都是 4**，配置里写 2 会形状不匹配、加载失败。
4. **`in_channels` 在权重里不可验证**：`token_embeds` / `var_embed` 永远按 `default_vars`（23 个）建，
   与 `dict_in_variables` 无关。所以入参列表写错了**权重照样 strict 加载成功**，
   只会在 forward 时静默取错变量。
5. **`latitude` 与 `lattitude` 同时存在**（后者是拼写变体，配置里用的是 `lattitude`）。
   `normalize_mean.npz` 里 `latitude` 是个标量，`lattitude` 才是真正的纬度场统计，搞混会让归一化跑偏。
6. **不要用 `mean_std/era5/0.25_deg/` 去归一化 US 数据** —— 那是 79 变量的全球 ERA5 统计，
   与 US 子集数值差很多（`land_sea_mask` 均值：全球 0.335 vs `era5-usa` 0.724）。
   每个数据目录已自带正确的 `normalize_*.npz`。
7. **降水全程在 `log1p(mm/day)` 空间**：`Denormalize` 对降水变量把 mean/std 写死成 0/1
   （`src/climate_learn/transforms/denormalize.py:23-24`），等于恒等变换。
   `N_preds.npy` 存的是 log 空间的值，要 mm/day 请自己 `np.expm1()`。
   打印的 PSNR/SSIM/RMSE 也仍在 log 空间，与论文数值不一定可比。
8. **Daymet 只有陆地**，海上是填充值（温度约 213 K，降水 0）。模型会照样复现填充值，
   温度 RMSE 4.34 K 被海面拉高了不少，真正的陆地误差要自己用 `land_sea_mask` 掩膜后算。
9. **`should_flip_image()` 按数据集名字判断是否上下翻转**（`FLIP_REQUIRED_SOURCES = {"ERA5","PRISM","DAYMET"}`），
   全球配置的 key 叫 `INFER` 不在名单里，导致上游写出的裸图南北颠倒。
   新增的三联图已按 `origin` 修正，但 `process_single_tile` / `adjust_coords_for_flip`
   那套翻转+坐标重映射逻辑与 tiling 拼接耦合，未改动。更干净的修法是按 `lat` 数组升降序判断。
10. **`examples/intermediate_downscaling.py:481` 的 `create_data_module()` 是死代码**：
    它调用 `IterDataModule.train_dataloader()` 时传了一堆参数，而该方法不接受任何参数。
    `main()` 没有调用它，所以不影响运行，但照它改就会炸。
11. **`intermediate_1b` 是 Megatron TP=4 张量并行分片，不是 FSDP**，且**不能简单合并**：
    118 个「完整形状」key 里有 34 个跨 rank 并不同步（`head.6.weight` 最低余弦相似度 0.473），
    row-parallel 的 bias 按 Megatron 语义通常要跨 rank 求和。正确用法是 `tensor_par: 4` 起 4 个 rank
    各读各的分片。另注意 4 个文件参数量相加 1.11B ≠ 真实模型规模 0.99B（差额是重复存储的 key）。
12. **`intermediate_117m` 只训到 epoch=3，`global_9.5m_precipitation` 只到 epoch=6**
    （对比 `us_126m_precipitation` 的 epoch=77），这两个权重很可能欠训练。
13. **`models/orbit2/README.md` 内容有错**：把 `intermediate_8m`（实测 9.50 M）说成 "8 billion"，
    列出的文件名 `126m_us_precipitation.ckpt` 等并不存在，且完全没提 `global-finetune/`、
    `static_variables/`、`mean_std/`。

---

## 7. 没覆盖到的

- `us_126m_temperature`、`global_9.5m_precipitation` 没有实跑。张量形状已核对一致，
  照现有 config 换 ckpt 路径和 `embed_dim/depth/num_heads` 即可。
- `pretrain/intermediate_8m` / `intermediate_117m` 没写 config。它们需要
  `era5-daymet/{10.0,2.5}_arcmin` 这一对，且输出是 3 通道（tp/tmin/tmax）。
- `pretrain/intermediate_1b.ckpt_rank_{0..3}` 完全没跑，需要 debug 队列开 4 卡。
- 预报示例的 `--devices > 1` 只是加了拦截，**没有实现分布式**。正确修法是把分片改成
  `(rank × num_workers + worker_id) :: (world_size × num_workers)`，但各文件时间步数不等，
  rank 间 batch 数不齐会让 DDP 在 epoch 末尾 hang 死，需要同时做 rank 分片 + batch 数对齐。
- 预报的损失是朴素 MSE，**没有纬度加权**，极地格点被严重过度加权，
  报出的 `val/mse` / `test/mse` 不能与 WeatherBench 基线对比。数据目录里有 `lat.npy`，要加权很容易。

---

## 8. 本次改动的文件

**新增**：`run_orbit2.sh`、`examples/run_visualize_1gpu.sh`、
`configs/infer_us_9.5m_precip.yaml`、`configs/infer_us_126m_precip.yaml`、`configs/infer_us_9.5m_temp.yaml`

**修改**：`configs/inference.yaml`、`examples/launch_visualize.sh`、`examples/visualize.py`、
`src/climate_learn/utils/visualize.py`、`src/climate_learn/data/iterdataset.py`、
`src/climate_learn/data/itermodule.py`、`examples/sparse_reslim_forecasting/{model.py,train.py,README.md}`、
`.gitignore`
