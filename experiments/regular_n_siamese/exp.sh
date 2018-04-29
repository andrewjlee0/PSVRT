#!/bin/bash
n_machines=40
script_name='exp.py'
username='jk9'
n_gpus=2

# submit job
PARTITION='bibs-gpu' # batch # bibs-smp # bibs-gpu # gpu # small-batch
QOS='bibs-tserre-condo' # pri-jk9

for i_machine in $(seq 1 $n_machines); do
sbatch -J "$script_name[$i_machine]" <<EOF
#!/bin/bash
#SBATCH -p $PARTITION
#SBATCH -n 3
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:$n_gpus
#SBATCH --mem=16G
#SBATCH --begin=now
#SBATCH --qos=$QOS
#SBATCH --output=/gpfs/scratch/$username/slurm/slurm-%j.out
#SBATCH --error=/gpfs/scratch/$username/slurm/slurm-%j.out

echo "Starting job $i_machine on $HOSTNAME"
LC_ALL=en_US.utf8 \
module load tensorflow boost hdf5 ffmpeg/1.2 cuda/7.5.18

nvidia-smi
python $script_name $n_machines $i_machine $n_gpus
EOF
done
