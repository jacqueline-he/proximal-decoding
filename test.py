from proximaldecode import ProximalDecodingFactory, generate
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import torch

console = Console()

# @ TODO enter here
safe_model_path = ""
risky_model_path = ""

console.print("[bold blue]Loading models and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(safe_model_path, padding_side="left")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

common_kwargs = dict(
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

safe_model = AutoModelForCausalLM.from_pretrained(safe_model_path, **common_kwargs)
risky_model = AutoModelForCausalLM.from_pretrained(risky_model_path, **common_kwargs)

# 2. Initialize the factory using the loaded models
factory = ProximalDecodingFactory.from_pretrained(
    safe_model=safe_model,
    risky_model=risky_model,
    tokenizer=tokenizer,
    k_radius=1.5,
)

prompts = [
    "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last ",
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, ",
    "Generate a factual biography about Elena Ferrante.\n\nBiography:"
]

config = GenerationConfig(
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(safe_model.device)
output_safe = safe_model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    generation_config=config
)
output_risky = risky_model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    generation_config=config
)
output_proximal = factory.generate(text=prompts, generation_config=config, seed=42)


for i, prompt in enumerate(prompts):
    console.print(Panel(Text(prompt, style="italic"), title=f"Prompt {i+1}", border_style="blue"))
    
    # Helper to decode and strip prompt
    def get_clean_output(sequences, idx):
        decoded = tokenizer.decode(sequences[idx], skip_special_tokens=True)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]
        return decoded

    console.print(Panel(Text(get_clean_output(output_safe, i), style="bold yellow"), title=f"Example {i+1}: Safe Model Output", border_style="yellow"))
    console.print(Panel(Text(get_clean_output(output_risky, i), style="bold red"), title=f"Example {i+1}: Risky Model Output", border_style="red"))
    console.print(Panel(Text(get_clean_output(output_proximal.sequences, i), style="bold green"), title=f"Example {i+1}: Proximal Decoding", border_style="green"))
    console.print("\n")
