import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


def load_chunks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data.get("chunks", [])
    return chunks


def to_messages_examples(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for ch in chunks:
        page = ch.get("page")
        context = ch.get("chunk_text", "").strip()
        for qa in ch.get("question_answer_pairs", []) or []:
            q = (qa.get("question") or "").strip()
            a = (qa.get("answer") or "").strip()
            if not q or not a or not context:
                continue
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant. Answer concisely based only on the given context."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"},
                        {"role": "assistant", "content": a},
                    ],
                    "meta": {"page": page},
                }
            )
    return examples


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    # Try tokenizer chat template, fallback if not configured
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    # Fallback: simple concatenation
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<{role}>: {content}")
    return "\n".join(parts)


def build_dataset(tokenizer, examples: List[Dict[str, Any]]) -> Dataset:
    texts = [apply_chat_template(tokenizer, ex["messages"]) for ex in examples]
    return Dataset.from_dict({"text": texts})


def main():
    data_path = os.path.join(os.path.dirname(__file__), "out.json")
    model_name = "google/gemma-3-270m"

    # Load the requested model only
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Cap sequence length for training when TRL config lacks max_seq_length
    tokenizer.model_max_length = 1024

    chunks = load_chunks(data_path)
    examples = to_messages_examples(chunks)
    dataset = build_dataset(tokenizer, examples)

    # Simple collator for text-only SFT
    def collate_fn(examples_batch):
        # If SFTTrainer has already tokenized the dataset, examples will have input_ids
        if "input_ids" in examples_batch[0]:
            import torch
            input_ids = [torch.tensor(ex["input_ids"]) for ex in examples_batch]
            # attention mask if provided
            if "attention_mask" in examples_batch[0]:
                attn = [torch.tensor(ex["attention_mask"]) for ex in examples_batch]
            else:
                attn = [torch.ones_like(t) for t in input_ids]
            # labels if provided, else copy input_ids
            if "labels" in examples_batch[0]:
                labels = [torch.tensor(ex["labels"]) for ex in examples_batch]
            else:
                labels = [t.clone() for t in input_ids]

            pad_id = tokenizer.pad_token_id or 0
            max_len = max(t.size(0) for t in input_ids)
            def pad(seq_list, pad_value):
                return torch.stack([torch.nn.functional.pad(s, (0, max_len - s.size(0)), value=pad_value) for s in seq_list])

            input_ids = pad(input_ids, pad_id)
            attn = pad(attn, 0)
            labels = pad(labels, -100)
            return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
        # Otherwise tokenize here from raw text
        texts = [ex["text"] for ex in examples_batch]
        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

    peft_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    training_args = SFTConfig(
        output_dir=os.path.join(os.path.dirname(__file__), "gemma_270m_lora_out"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=200,
        eval_strategy="no",
        optim="adamw_torch",
        warmup_ratio=0.03,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_cfg,
        data_collator=collate_fn,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()


