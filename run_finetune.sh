#!/bin/bash

python load_synthetic_sql.py --limit 0

axolotl train deepseek-coder-1.3b-instruct.yml

axolotl merge-lora deepseek-coder-1.3b-instruct.yml