import pandas as pd

max_seq_length = 8192 # threshold

processed_train_path = "/data/zhenxiong/data/mmau_pro/train_w_length.parquet"
processed_test_path = "/data/zhenxiong/data/mmau_pro/test_w_length.parquet"

train_df = pd.read_parquet(processed_train_path)
test_df = pd.read_parquet(processed_test_path)

train_df = train_df[train_df['seq_length'] < max_seq_length]
test_df = test_df[test_df['seq_length'] < max_seq_length]

train_df.to_parquet("/data/zhenxiong/data/mmau_pro/train_w_length_filtered.parquet", index=False)
test_df.to_parquet("/data/zhenxiong/data/mmau_pro/test_w_length_filtered.parquet", index=False)
