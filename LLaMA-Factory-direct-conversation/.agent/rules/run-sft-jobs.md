---
trigger: always_on
---

The training env are inited from `.llamafactory` and you should always 

```
source .llamafactory/bin/activate
```

then launch training jobs via `eai-run`. For example, the training jobs `llamafactory-cli train train_parallel.yaml` now should be `eai-run -i --pty llamafactory-cli train train_parallel.yaml