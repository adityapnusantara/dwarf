# Dwarf: DeepSeek Coder 1.3B Finetune with Axolotl

Self-contained recipe to QLoRA-finetune `deepseek-ai/deepseek-coder-1.3b-instruct` on the synthetic Text-to-SQL dataset.

## Files
- `deepseek-coder-1.3b-instruct.yml` — Axolotl training config (relative paths, chat_template dataset).
- `run_finetune.sh` — one-shot pipeline: generate dataset → train → merge LoRA.
- `load_synthetic_sql.py` — fetch and format `gretelai/synthetic_text_to_sql` into chat messages.
- `formatted_synthetic_sql.jsonl` — generated dataset (gitignored).

## Prereqs
- Axolotl installed and on PATH (`axolotl` CLI).
- HF auth if the model/dataset requires it (`HF_TOKEN`/`HF_HOME` as needed).
- Optional: flash-attn if you keep `flash_attention: true` in the config.

## Quick start
```bash
cd "$(dirname "$0")"
./run_finetune.sh
```
This pulls the full `train` split, trains, then merges LoRA.

## Customizing runs
- Limit rows: `LIMIT=500 ./run_finetune.sh`
- Choose split: `SPLIT=validation ./run_finetune.sh`
- Change dataset path: `DATA=/tmp/sql.jsonl ./run_finetune.sh`
- Change config: `CONFIG=/path/to/your.yml ./run_finetune.sh`
- Pass extra Axolotl args: `./run_finetune.sh --max_steps 200`

## Dataset format (chat_template)
Each row in `formatted_synthetic_sql.jsonl`:
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
