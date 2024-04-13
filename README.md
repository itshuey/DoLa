Enhancing Instruction-Following Capabilities in Seq2Seq Models: A Novel Adaptation of DoLA in T5 and FLAN-T5
===

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)](https://github.com/huggingface/transformers)


Code for the paper "Enhancing Instruction-Following Capabilities in Seq2Seq Models: A Novel Adaptation of DoLA in T5 and FLAN-T5"

Authors: **Huey Sun** $^\dagger$, **Lorenzo Gilly** $^\dagger$, **Anabel Yong** $^\dagger$

$^\dagger$ University College London

## Overview

![DoLa](figure.png)

We adapted the DoLa to T5 and instruction-tuned FLAN-T5 models, and investigated how DoLa can improve keyword inclusion by analysing logit evolution through the model layers.
## Setup

```
pip install -e transformers-4.28.1
pip install datasets
pip install accelerate
```

## Experiments

### Arguments

| Argument          | Example           | Description   |
| ----------------- | ----------------- | ------------- |
| `--model-name`    | `google/flan-t5-large` | Specifies the model you want to use, currently we support LLaMA-v1 and the T5 family. |
| `--data-path`     | `/path/to/dataset` | Path to the dataset file or folder. |
| `--output-path`   | `output-path.json` | Where to store the output results. |
| `--num-gpus`      | `1` | Number of GPUs to use |
| `--max_gpu_memory`| `27` | Maximum GPU memory size (in GiB) to allocate. Default: 27 (for 32G V100).  |
| `--print-logits`| | Adding this argument prints the top 5 logits in the premature layers for each token generated |

### Understanding `--early-exit-layers`

The `--early-exit-layers` argument takes a string containing a sequence of layer numbers separated by commas, with no spaces in between. By specifying different number of layers, we make the model decode at different modes.


| Number of Layers Specified  | Example (str)     | Description of Decoding Mode                                                                                     |
| ---------------------------| ------------- | ----------------------------------------------------------------------------------------------- |
| 1                          | `-1`      | **Naive decoding** from the final layer output.       |
| 2                          | `16,32`   | **DoLa-static decoding** with the second specified layer (i.e. `32`) as the `mature_layer` and first specified layer (i.e. `16`) as `premature_layer`. |
| >2                         | `0,2,4,6,8,10,12,14,32`    | **DoLa decoding** with the last specified layer (i.e. `32`) as the `mature_layer` and all the preceding layers (i.e. `0,2,4,6,8,10,12,14`) as `candidate_premature_layers`. |

### IfEval
The input prompts can be found in data/ifeval-input-data.jsonl. Further instructions for analyzing model output can be found in evaluation/IfEval

#### Baseline
```bash
python ifeval_eval.py --model-name google/flan-t5-small --data-path ./data/ --output-path output-path.json --num-gpus 1
python ifeval_eval.py --model-name google/flan-t5-base --data-path ./data/ --output-path output-path.json --num-gpus 1
python ifeval_eval.py --model-name google/flan-t5-large --data-path ./data/ --output-path output-path.json --num-gpus 1
python ifeval_eval.py --model-name google/flan-t5-xl --data-path ./data/ --output-path output-path.json --num-gpus 1
```

#### DoLa
```bash
python ifeval_eval.py --model-name google/flan-t5-small --early-exit-layers 0,2,4,6,8 --data-path ./data/ --output-path output-path.json --num-gpus 1
python ifeval_eval.py --model-name google/flan-t5-base --early-exit-layers 0,2,4,6,8,10,12 --data-path ./data/ --output-path output-path.json --num-gpus 1
python ifeval_eval.py --model-name google/flan-t5-large --early-exit-layers 0,2,4,6,8,10,12,14,16,18,20,22,24 --data-path ./data/--output-path output-path.json --num-gpus 1
python ifeval_eval.py --model-name google/flan-t5-xl --early-exit-layers 0,2,4,6,8,10,12,14,16,18,20,22,24 --data-path ./data/ --output-path output-path.json --num-gpus 1
```

### Memo Trap
The input prompts can be found in data/memotrap-input-data.jsonl. Further instructions for analyzing model output can be found in evaluation/MemoTrap

#### Baseline
```bash
python memo_trap_eval.py --model-name google/flan-t5-small --data-path ./data/ --output-path output-path.json --num-gpus 1
python memo_trap_eval.py --model-name google/flan-t5-base --data-path ./data/ --output-path output-path.json --num-gpus 1
python memo_trap_eval.py --model-name google/flan-t5-large --data-path ./data/ --output-path output-path.json --num-gpus 1
python memo_trap_eval.py--model-name google/flan-t5-xl --data-path ./data/ --output-path output-path.json --num-gpus 1
```

#### DoLa
```bash
python memo_trap_eval.py --model-name google/flan-t5-small --early-exit-layers 0,2,4,6,8 --data-path ./data/ --output-path output-path.json --num-gpus 1
python memo_trap_eval.py--model-name google/flan-t5-base --early-exit-layers 0,2,4,6,8,10,12 --data-path ./data/ --output-path output-path.json --num-gpus 1
python memo_trap_eval.py --model-name google/flan-t5-large --early-exit-layers 0,2,4,6,8,10,12,14,16,18,20,22,24 --data-path ./data/ --output-path output-path.json --num-gpus 1
python memo_trap_eval.py --model-name google/flan-t5-xl --early-exit-layers 0,2,4,6,8,10,12,14,16,18,20,22,24 --data-path ./data/ --output-path output-path.json --num-gpus 1
```

## Reference Repositories
- DoLa: https://github.com/voidism/DoLa
- IfEval: https://huggingface.co/datasets/HuggingFaceH4/ifeval
- Memo Trap: https://github.com/liujch1998/memo-trap
- FLAN: https://huggingface.co/docs/transformers/en/model_doc/flan-t5