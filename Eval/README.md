# Code for test
## Run
```bash
./test_32_times_auto.sh -m /your/path/to/model
```

It will dump all the evaluation result to a directory in the model's directory `$OUTPUT_BASE_DIR`, which is composed in the following way
```bash
OUTPUT_BASE_DIR="${MODEL}/Test_${TIMESTAMP}"
```
where the `$MODEL` is the model path you input, and the `$TIMESTAMP` is the timestamp when the processed is started.

The input file path should be changed before starting (line 22)

## Report
```bash
python3 aggregate_accuracy.py --test-dir OUTPUT_BASE_DIR
```

It will output a summary report of the eval called `summary_report.txt` 

Where it will have accuracy from 1-30 31-60 61-90 problem. Using the `AIME2425.json` given in HF, it indicate the result of AIME24 AIME25 HMMT respectively.



