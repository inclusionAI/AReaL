# Code for ablating data quality

## Train
Refer to directory train/ train_moe

Environment
```
pip install -r requirememt.txt
```
or 
```
pip install -r requirement_moe.txt
```

then complete the path and run
```
./train/sft_multiverse.sh
```
or 
```
./train_moe/sft_multiverse.sh
```

## Test 
Go to masking and set the path correctly.

The run 
```
./test8time_real.sh
```
it will run 8 times with 8 GPUs simutaneously