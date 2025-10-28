import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=str, default=None, help="Context text to ground the answer")
    parser.add_argument("--question", type=str, default=None, help="User question")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--no_adapter", action="store_true", help="Do not load LoRA adapter")
    args = parser.parse_args()
    base_model = "google/gemma-3-270m"
    adapter_dir = os.path.join(os.path.dirname(__file__), "gemma_270m_lora_out")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    # Load LoRA adapter (unless disabled)
    if not args.no_adapter:
        try:
            model.load_adapter(adapter_dir)
        except Exception:
            # Some environments may need PEFT wrapper to load adapters; try graceful fallback
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_dir)

    system = (
        "You are an on-device (edge) car assistant for a vehicle owner's manual. "
        "Answer concisely and rely only on the provided context. "
        "If the answer is not in the context, say: 'I’m not sure from the provided context.'"
    )

    # Single-shot mode if --context and --question are provided
    if args.context and args.question:
        prompt = (
            f"{system}\n\n"
            f"Context:\n{args.context}\n\n"
            f"User: {args.question}\n"
            f"Assistant:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print("\n==== Inference Output ====\n")
        print(f"Q: {args.question}")
        print(f"A: {answer}")
        return

    # Interactive mode
    print("\nInteractive mode. Press Enter on an empty line to quit.\n")
    while True:
        ctx = input("Context: ").strip()
        if not ctx:
            break
        q = input("Question: ").strip()
        if not q:
            break
        prompt = (
            f"{system}\n\n"
            f"Context:\n{ctx}\n\n"
            f"User: {q}\n"
            f"Assistant:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"Q: {q}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()


