# Dwarf: DeepSeek Coder SQL QLoRA (Axolotl)

QLoRA fine-tune of `deepseek-ai/deepseek-coder-1.3b-instruct` on synthetic Text-to-SQL, with a tidier layout.

## Structure
```
├── README.md
├── scripts/
│   ├── prepare_data.py      # download + format gretelai/synthetic_text_to_sql
│   └── train.sh             # prepare -> train -> merge LoRA
├── configs/
│   └── deepseek_sql_qlora.yaml
└── data/
    └── train_sample.jsonl   # small sample (chat_template)
```
Generated at runtime (gitignored): `outputs/`, `last_run_prepared/`, and any regenerated JSONL.

## Prereqs
- `axolotl` CLI available.
- HF auth if required (`HF_TOKEN`, `HF_HOME`).
- Optional: flash-attn if you keep `flash_attention: true` in the config.
- Ensure scripts are executable (run once if needed):
  ```
  /bin/bash -lc "chmod +x ./scripts/train.sh ./scripts/prepare_data.py"
  ```

## Quick start
```bash
cd "$(dirname "$0")"
./scripts/train.sh
```
Defaults: split=`train`, limit=`200`, config=`configs/deepseek_sql_qlora.yaml`, data output=`data/train_sample.jsonl`.

## Customization
- Limit rows: `LIMIT=500 ./scripts/train.sh`
- Full split: `LIMIT=0 ./scripts/train.sh`
- Choose split: `SPLIT=validation ./scripts/train.sh`
- Custom dataset path: `DATA=/tmp/sql.jsonl ./scripts/train.sh`
- Custom config: `CONFIG=/path/to/config.yaml ./scripts/train.sh`
- Skip merge step: `SKIP_MERGE=1 ./scripts/train.sh`
- Extra Axolotl args: `./scripts/train.sh --max_steps 200 --save_steps 100`

## Data format (chat_template)
Each row:
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert SQL assistant...Table schema:\n..."},
    {"role": "user", "content": "<natural language request>"},
    {"role": "assistant", "content": "<SQL query>"}
  ]
}
```

## Outputs
- Checkpoints/merged model: `outputs/`
- Prepared dataset cache: `last_run_prepared/`

## Example run output
```
root@27f8d2a78cc8:/workspace/finetune/dwarf# ./scripts/train.sh
[dwarf] Preparing data -> /workspace/finetune/dwarf/data/train.jsonl (split=train, limit=0)
Wrote 100000 rows to /workspace/finetune/dwarf/data/train.jsonl
[dwarf] Training...

     #@@ #@@      @@# @@#
    @@  @@          @@  @@           =@@#                               @@                 #@    =@@#.
    @@    #@@@@@@@@@    @@           #@#@=                              @@                 #@     .=@@
      #@@@@@@@@@@@@@@@@@            =@# @#     ##=     ##    =####=+    @@      =#####+  =#@@###.   @@
    @@@@@@@@@@/  +@@/  +@@          #@  =@=     #@=   @@   =@#+  +#@#   @@    =@#+  +#@#   #@.      @@
    @@@@@@@@@@  ##@@  ##@@         =@#   @#      =@# @#    @@      @@   @@    @@      #@   #@       @@
     @@@@@@@@@@@@@@@@@@@@          #@=+++#@=      =@@#     @@      @@   @@    @@      #@   #@       @@
                                  =@#=====@@     =@# @#    @@      @@   @@    @@      #@   #@       @@
    @@@@@@@@@@@@@@@@  @@@@        #@      #@=   #@=  +@@   #@#    =@#   @@.   =@#    =@#   #@.      @@
                                 =@#       @#  #@=     #@   =#@@@@#=    +#@@=  +#@@@@#=    .##@@+   @@
    @@@@  @@@@@@@@@@@@@@@@
```
