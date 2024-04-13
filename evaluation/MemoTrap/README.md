# Memo Trap

Written for "A Novel Adaptation of DoLA in T5 and FLANT5"

## How to run

You need to run memotrap_eval to generate a jsonl file with your memo-trap data. 

Then, call `memo_trap_results_eval`. For example:

```bash
python3 memo_trap_results_eval.py \
  --data-path flan-t5-base-memo.jsonl 
```
