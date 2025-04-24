#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-task=2
#SBATCH --time=02:30:00
#SBATCH --mem-per-cpu=1443

module purge
module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5 Python/3.11.3 PyTorch-bundle/2.1.2-CUDA-12.1.1 PyTorch/2.1.2-CUDA-12.1.1 SciPy-bundle/2023.07 Pillow/10.0.0 OpenCV/4.8.1 matplotlib/3.7.2

source this_venv/bin/activate

srun --exclusive --gres=gpu:2 --ntasks=1 --cpus-per-task=4 --gpus-per-task=2 --mem-per-cpu=1443 --output log.txt ./KeypointDetector.py < --batch_size 1 --learning_rate 0.001 --global_image_size 600>

echo "Waiting for all job steps to complete..."
wait
echo "All jobs completed!"