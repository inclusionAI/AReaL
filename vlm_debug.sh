workdir=/storage/openpsi/users/lichangye.lcy/AReaL 
srun --mpi=pmi2 --ntasks=1 --gres=gpu:8 --chdir=$workdir --cpus-per-task=64 --mem=1500G --pty singularity shell --nv --no-home --writable-tmpfs --bind /storage:/storage  /storage/openpsi/images/areal-v0.3.0.post1.sif
pip install -e . 