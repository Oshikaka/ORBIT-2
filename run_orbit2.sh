#!/bin/bash
# ---------------------------------------------------------------------------
# ORBIT-2 on Frontier — 每个部分怎么跑。
#
#   ./run_orbit2.sh              列出所有可跑的东西
#   ./run_orbit2.sh <命令>       跑其中一个（跑之前会把真实命令打印出来）
#
# 所有命令都从仓库根目录执行，环境由各自的启动脚本负责激活，不用先 conda activate。
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

ENV_PREFIX=/lustre/orion/csc662/proj-shared/xinru/envs/orbit
ERA5_5625=/lustre/orion/lrn036/world-shared/data/superres/era5/5.625_deg

activate_env() {
    module load miniforge3/23.11.0-0 >/dev/null 2>&1 || true
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PREFIX"
    export PYTHONNOUSERSITE=1
}

run() { echo "+ $*" >&2; "$@"; }

usage() {
cat <<'EOF'
ORBIT-2 运行手册

=== A. 降尺度 Downscaling（用已下载的 checkpoint 推理 + 出图，不训练）=====

  us-precip        美国降水，9.5M 模型，单卡 ~2-4 分钟          【从这个开始】
  us-precip-126m   美国降水，126M 模型，单卡，精度最好
  us-temp          美国温度，9.5M 模型，单卡
  global-precip    全球降水 720x1440 -> 2880x5760，16 块 tiling
                   太重，提交到 debug 队列（约 1.5 分钟出结果）

  加 --index N 换样本（测试集是 2021 全年，N 取 0-364）：
      ./run_orbit2.sh us-precip --index 100

  产物在 outputs/<名字>/：
      N_comparison.png   三联对比图（输入 / 降尺度结果 / 真值），先看这张
      N_preds.npy        预测数组 480x960
      N_truth.npy        真值数组
      N_input.npy        低分辨率输入 120x240
  终端会打印 PSNR / SSIM / RMSE。

=== B. 预报 Forecasting（Sparse-Reslim 示例，从头训，单卡）==============

  fc-smoke         合成数据冒烟测试，不需要任何数据，~20 秒   【从这个开始】
  fc-train         真实 ERA5 5.625° 数据，2 epoch 短跑，~3 分钟
  fc-train-cpu     同上但用 CPU，~35 秒，用来确认数据通路

  想自己调参就直接调 train.py，选项见
  examples/sparse_reslim_forecasting/README.md

=== C. 其它 =============================================================

  env              打印环境自检（python / torch / GPU / 关键包）
  status           列出已有的输出文件

EOF
}

# --------------------------------------------------------------------------
# A. 降尺度
# --------------------------------------------------------------------------
downscale() {
    local config="$1" outdir="$2" variable="$3"; shift 3
    run ./examples/run_visualize_1gpu.sh "$config" \
        --variable "$variable" --output-dir "outputs/$outdir" "$@"
    echo
    echo "看图：outputs/$outdir/*_comparison.png"
}

# --------------------------------------------------------------------------
# B. 预报
# --------------------------------------------------------------------------
forecast() {
    activate_env
    export OMP_NUM_THREADS=8
    run python examples/sparse_reslim_forecasting/train.py "$@"
}

# --------------------------------------------------------------------------
case "${1:-help}" in
  us-precip)
      shift; downscale configs/infer_us_9.5m_precip.yaml us_9.5m_precip \
                       total_precipitation_24hr "$@" ;;

  us-precip-126m)
      shift; downscale configs/infer_us_126m_precip.yaml us_126m_precip \
                       total_precipitation_24hr "$@" ;;

  us-temp)
      shift; downscale configs/infer_us_9.5m_temp.yaml us_9.5m_temp \
                       2m_temperature_min "$@" ;;

  global-precip)
      shift
      mkdir -p outputs/global_126m_precip
      cd examples
      run env CONFIG=../configs/inference.yaml \
              OUTPUT_DIR=../outputs/global_126m_precip \
              INDEX="${INDEX:-0}" NTASKS=1 \
          sbatch -A csc662 -q debug -t 00:20:00 -N 1 launch_visualize.sh
      echo
      echo "查看进度：squeue -u \$USER"
      echo "看日志：  tail -f examples/flash-<JOBID>.out" ;;

  fc-smoke)
      shift; forecast --smoke-test "$@" ;;

  fc-train)
      shift
      forecast "$ERA5_5625" \
          --accelerator gpu --devices 1 \
          --max-epochs 2 --batch-size 16 --num-workers 2 \
          --limit-train-batches 50 --limit-val-batches 10 --limit-test-batches 10 \
          --output-dir outputs/forecast_5625 "$@" ;;

  fc-train-cpu)
      shift
      forecast "$ERA5_5625" \
          --accelerator cpu --devices 1 --patience 0 \
          --max-epochs 1 --batch-size 8 --num-workers 2 \
          --limit-train-batches 20 --limit-val-batches 5 --limit-test-batches 5 \
          --output-dir outputs/forecast_5625_cpu "$@" ;;

  env)
      activate_env
      python - <<'PY'
import torch, sys, pytorch_lightning as pl
print("python           ", sys.version.split()[0])
print("torch            ", torch.__version__)
print("pytorch-lightning", pl.__version__)
print("GPU count        ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name         ", torch.cuda.get_device_name(0))
import climate_learn
print("climate_learn    ", climate_learn.__file__)
PY
      ;;

  status)
      if [ -d outputs ]; then find outputs -name '*_comparison.png' -o -name '*.ckpt' | sort
      else echo "还没有任何输出，先跑 ./run_orbit2.sh us-precip"; fi ;;

  help|-h|--help) usage ;;
  *) echo "未知命令：$1" >&2; echo >&2; usage >&2; exit 2 ;;
esac
