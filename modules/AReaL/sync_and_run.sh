set -ex
script_path=$1

    workdir=/storage/openpsi/users/xushusheng.xss/projects/areal_megatron@0922

    rsync -avzPr /home/admin/xushusheng.xss/inclusionAI/AReaL/ $workdir/ --exclude **__pycache__** --delete

cd $workdir

git status
bash $script_path ${@:2} 
