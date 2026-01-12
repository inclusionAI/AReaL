---
trigger: always_on
---

The training env are inited from `.llamafactory` and you should always 

```
source .llamafactory/bin/activate
```
then launch training jobs via `eai-run`. For example, the training jobs 

`llamafactory-cli train train_parallel.yaml` 
now should be 

```
eai-run -i -J train_parallel llamafactory-cli train train_parallel.yaml
```
and the trainings logs are under the `run/dev/train_parallel`. (-J xxx means the job name and also for log output dir)


For multiple nodes, the launch would be look like 

```
eai-run -J train_parallel-N 2 llamafactory-cli train  train_parallel.yaml
```


Note this launch may take up to 30mins. So be patient and wait.