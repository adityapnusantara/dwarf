# Dwarf: Text2SQL Experiments

QLoRA fine-tune on synthetic Text-to-SQL, with a tidier layout.

## Structure
```
├── README.md
├── requirements.txt
├── configs/
│   ├── deepseek_sql_qlora.yaml
│   ├── gemma-3-1b-qlora.yml
│   └── gemma-3-1b-qlora-e-2.yml
├── scripts/
│   ├── prepare_data.py        # download + format gretelai/synthetic_text_to_sql
│   ├── train.sh               # prepare -> train -> merge LoRA
│   └── schema_converter.py    # trim LangChain SQL schema text into promptable DDL+inserts
├── data/
│   ├── train.jsonl            # fine-tuning data (chat_template)
│   └── test.jsonl             # eval slice
├── database/                  # synthetic SQLite schemas (one .sqlite per scenario)
│   └── <domain_name>/*.sqlite
├── notebooks/
│   └── chain_text2sql.ipynb   # pipeline to process text into SQL from user question
└── outputs/
    ├── deepseek-sql-qlora/
    ├── gemma-3-1b-sql-qlora/
    └── gemma-3-1b-sql-qlora-e-2/

```

## Install & setup
1) Start Axolotl container (GPU required):
   ```bash
   docker run --gpus '"all"' -it axolotlai/axolotl:main-latest
   ```
2) Inside the container, clone this repo:
   ```bash
   git clone https://github.com/adityapnusantara/dwarf
   cd dwarf
   ```
3) Make scripts executable, then launch training:
   ```bash
   /bin/bash -lc "chmod +x ./scripts/train.sh ./scripts/prepare_data.py"
   HF_TOKEN=<HF_TOKEN> ./scripts/train.sh
   ```

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
