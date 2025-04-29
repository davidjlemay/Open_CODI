#!/bin/bash
#SBATCH --account={CC_ACCOUNT}
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48 
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --mail-user={EMAIL}
#SBATCH --mail-type=END
#SBATCH --output=%N.%j.out

# https://docs.alliancecan.ca/wiki/Python#Creating_virtual_environments_inside_of_your_jobs
# https://docs.alliancecan.ca/wiki/Huggingface#Training_Large_Language_Models_(LLMs)

# Define cleanup function for error handling
cleanup() {
    echo "Error detected at line $1. Saving any output files before exiting..."
    
    # Create tarball of any results that were generated
    if [ -d "${SLURM_TMPDIR}/results" ]; then
        echo "Archiving results from ${SLURM_TMPDIR}/results"
        tar -czvf ${SLURM_TMPDIR}/results.tar.gz ${SLURM_TMPDIR}/results/*
        rsync -a ${SLURM_TMPDIR}/results.tar.gz ${PWD}/results/
        echo "Outputs saved to ${PWD}/results/results.tar.gz"
    else
        echo "No results directory found at ${SLURM_TMPDIR}/results"
    fi
    
    exit 1
}

# Set trap to catch errors
trap 'cleanup $LINENO' ERR


export HF_HOME=$SCRATCH/hf_models
export HF_HUB_OFFLINE=1
#huggingface-cli download MODEL

echo "loading models from $HF_HOME ..."

module load gcc arrow/15.0.1 python/3.11


virtualenv --no-download $SLURM_TMPDIR/venv

source $SLURM_TMPDIR/venv/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt


# Set PyTorch memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
# Don't restrict to GPU 0 - use both GPUs
# export CUDA_VISIBLE_DEVICES=0,1  # Let the code handle this instead

cp -R $SLURM_TMPDIR/data $SLURM_TMPDIR/data

accelerate launch \
—-config_file="fsdp.yaml" \ 
--mixed_precision="fp16" \ 
--num_machines=$SLURM_NNODES \
--machine_rank=$SLURM_NODEID \
--num_processes=4 \
train.py 


# If we get here, everything succeeded
echo "Python script completed successfully. Archiving results..."
tar -czvf ${SLURM_TMPDIR}/results.tar.gz ${SLURM_TMPDIR}/results/*
rsync -a ${SLURM_TMPDIR}/results.tar.gz ${PWD}/results/
echo "Job completed successfully!"
