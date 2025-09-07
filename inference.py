#!/usr/bin/env python3
"""
inference.py
Simple interactive chat with the trained GPT-2-style model.

Usage:
  python inference.py --model_dir checkpoints/400m --device auto
"""
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="checkpoints/400m")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--system", type=str, default="You are a helpful, concise assistant.")
    return p

def pick_device(device_flag):
    if device_flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_flag

def main():
    args = build_argparser().parse_args()
    device = pick_device(args.device)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"]})
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"

    print(f"Loading model from {args.model_dir} on {device} ...")
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    system_prefix = f"<|system|> {args.system}".strip()

    while True:
        try:
            user = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if user.lower() in {"exit", "quit"}:
            print("bye.")
            break

        prompt = f"{system_prefix} <|user|> {user} <|assistant|>"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = output[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        print(f"Assistant: {text.strip()}")

if __name__ == "__main__":
    main()
