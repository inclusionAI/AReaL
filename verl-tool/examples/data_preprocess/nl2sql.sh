set -x
sql_database_path=${1:-"data/nl2sql"}
if [[ ! -f "hfd.sh" ]]; then
    echo "downloading hfd.sh"
    
    wget https://hf-mirror.com/hfd/hfd.sh
    chmod a+x hfd.sh
else
    echo "hfd.sh already exists."
fi
# parquets
bash hfd.sh JasperHaozhe/NL2SQL-Queries --dataset --tool wget
# database
bash hfd.sh JasperHaozhe/NL2SQL-Database --dataset --tool wget
rm hfd.sh

cache_path=${CACHE_PATH:-"data/nl2sql/cache"}
mkdir -p $cache_path
mkdir -p $sql_database_path
# parquets
mv NL2SQL-Queries $sql_database_path/
# database
mv NL2SQL-Database $sql_database_path/
unzip $sql_database_path/NL2SQL-Database/sql_database.zip -d $cache_path/
mv $cache_path/home/ma-user/work/haozhe/workspace/verl-tool/sql_database/* $cache_path/ && \
rm -rf "$cache_path/home" # internal path of the zip, this line removes it.

echo "utils/sql_executor.py finds databases in data/nl2sql/cache by default"
echo "If you want to copy to another folder, set `export CACHE_PATH=` and change the database path in utils/sql_executor.py accordingly"