#!/bin/bash
#============ Slurm Options ===========
#SBATCH -p gr20100g             # ジョブキュー(パーティション)の指定。投入したいキュー名に適宜変更が必要。
#SBATCH -t 24:00:00              # 経過時間の指定(1時間を指定する例)。
#SBATCH --rsc g=1  # 要求リソースの指定(4プロセスx8スレッドのハイブリッド並列の例)。
#SBATCH -o logs/epoch40_シード10.%j.out            # ジョブの標準出力ファイルの指定。%xはジョブ名、%j はジョブIDに置換されます。
#============ Shell Script ============

# (任意) set -x を指定するとジョブスクリプトの実行経過を把握しやすくなります。
#set -x

# module purge
module load pytorch/2.2.0.py311_cuda-12.1

/usr/bin/time -f "Total elapsed time: %E" python3 SLTHinQUBO.py --quad-scale 2.0 --epochs 40 --seed 10