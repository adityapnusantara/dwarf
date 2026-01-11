#!/usr/bin/env python3
"""Convert chat messages to a plain text prompt using the model's chat template."""

from transformers import AutoTokenizer


def main():
    model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # messages = [{"role": "system", "content": "You are an expert SQL assistant. Use ONLY the provided table schema to write a correct SQL query. Return valid SQL without explanations.\nTable schema:\nCREATE TABLE salesperson (salesperson_id INT, name TEXT, region TEXT); INSERT INTO salesperson (salesperson_id, name, region) VALUES (1, 'John Doe', 'North'), (2, 'Jane Smith', 'South'); CREATE TABLE timber_sales (sales_id INT, salesperson_id INT, volume REAL, sale_date DATE); INSERT INTO timber_sales (sales_id, salesperson_id, volume, sale_date) VALUES (1, 1, 120, '2021-01-01'), (2, 1, 150, '2021-02-01'), (3, 2, 180, '2021-01-01');"}, {"role": "user", "content": "What is the total volume of timber sold by each salesperson, sorted by salesperson?"}]
    messages = [{"role": "user", "content": "You are an expert SQL assistant. Use ONLY the provided table schema to write a correct SQL query. Return valid SQL without explanations.\nTable schema:\nCREATE TABLE salesperson (salesperson_id INT, name TEXT, region TEXT); INSERT INTO salesperson (salesperson_id, name, region) VALUES (1, 'John Doe', 'North'), (2, 'Jane Smith', 'South'); CREATE TABLE timber_sales (sales_id INT, salesperson_id INT, volume REAL, sale_date DATE); INSERT INTO timber_sales (sales_id, salesperson_id, volume, sale_date) VALUES (1, 1, 120, '2021-01-01'), (2, 1, 150, '2021-02-01'), (3, 2, 180, '2021-01-01');\nQuestion: What is the total volume of timber sold by each salesperson, sorted by salesperson?"}, {"role": "assistant", "content": "SELECT salesperson_id, name, SUM(volume) as total_volume FROM timber_sales JOIN salesperson ON timber_sales.salesperson_id = salesperson.salesperson_id GROUP BY salesperson_id, name ORDER BY total_volume DESC;"}]

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )

    print(prompt)


if __name__ == "__main__":
    main()
