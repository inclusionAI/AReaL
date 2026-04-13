# NOTE: expect to run in the root directory of verl-tool!
export HF_ENDPOINT=https://hf-mirror.com

# download hfd.sh
set -x
sql_database_path=${1:-"data/nl2sql"}
if [[ ! -f "hfd.sh" ]]; then
    echo "downloading hfd.sh"
    
    wget https://hf-mirror.com/hfd/hfd.sh
    chmod a+x hfd.sh
else
    echo "hfd.sh already exists."
fi

# download processed verl-tool-compatible dataset
bash hfd.sh VerlTool/SkyRL-SQL-Reproduction --dataset --tool wget
mv SkyRL-SQL-Reproduction/data/ ./data/skyrl_processed/

# download OmniSQL-datasets, takes around 50GB
bash hfd.sh seeklhy/OmniSQL-datasets --dataset --tool wget
unzip OmniSQL-datasets/data.zip -d ./data/synsql/
rm -r OmniSQL-datasets

rm hfd.sh