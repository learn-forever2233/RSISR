#!/bin/bash -l
#SBATCH -p qgpu_4090
#SBATCH --job-name=test_bicubic_perturb_15000
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=96:00:00
#SBATCH --mem=64GB

module load cuda/11.8
module load gcc/9.5.0
module load MPI/mpich/4.1.2-gcc-9.5.0

export PATH="/hpcfs/fpublic/app/miniforge3/conda/condabin:$PATH"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gpsmamba

echo "开始测试：使用 finetune_GPS_x4_bicubic_perturb 的 net_g_60000.pth 模型"
echo "模型路径：/hpcfs/fhome/competitor01/lhx/GPSMamba_2/GPSMamba/experiments/finetune_GPS_x4_bicubic_perturb/models/net_g_60000.pth"
echo "输入数据：/hpcfs/fhome/competitor01/lhx/testLR_X4/X4"
export PYTHONPATH=/hpcfs/fhome/competitor01/lhx/GPSMamba_2/GPSMamba:$PYTHONPATH
cd /hpcfs/fhome/competitor01/lhx/GPSMamba_2/GPSMamba
python basicsr/test.py -opt options/test/GPSMamba/test_GPSMamba_x4_bicubic_perturb_60000.yml

echo "测试完成！"

