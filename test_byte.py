from proximaldecode import BytewiseProximalDecodingFactory
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import torch

console = Console()

safe_model_path = "common-pile/comma-v0.1-2t"
risky_model_path = "meta-llama/Llama-3.1-70B"

console.print("[bold blue]Loading models...")
common_kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)

safe_model = AutoModelForCausalLM.from_pretrained(safe_model_path, **common_kwargs)
risky_model = AutoModelForCausalLM.from_pretrained(risky_model_path, **common_kwargs)

# Initialize the bytewise factory
console.print("[bold blue]Initializing BytewiseProximalDecodingFactory...")
factory = BytewiseProximalDecodingFactory.from_pretrained(
    safe_model=safe_model,
    risky_model=risky_model,
    safe_model_path=safe_model_path,
    risky_model_path=risky_model_path,
    k_radius=0.5,
)

# Ensure tokenizers have pad tokens and use left-padding for decoder-only models
for tcs in [factory.tcs_safe, factory.tcs_risky]:
    if tcs.tokenizer.pad_token is None:
        tcs.tokenizer.pad_token = tcs.tokenizer.eos_token
    tcs.tokenizer.padding_side = "left"

prompts = [
    "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last ",
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, ",
    "Generate a factual biography about Elena Ferrante.\n\nBiography:"
]

config = GenerationConfig(
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
)

# Generate with each model
console.print("[bold blue]Generating outputs...")

safe_inputs = factory.tcs_safe.tokenizer(prompts, return_tensors="pt", padding=True)
output_safe = safe_model.generate(
    input_ids=safe_inputs.input_ids.to(safe_model.device),
    attention_mask=safe_inputs.attention_mask.to(safe_model.device),
    generation_config=config,
)

risky_inputs = factory.tcs_risky.tokenizer(prompts, return_tensors="pt", padding=True)
output_risky = risky_model.generate(
    input_ids=risky_inputs.input_ids.to(risky_model.device),
    attention_mask=risky_inputs.attention_mask.to(risky_model.device),
    generation_config=config,
)
output_proximal = factory.generate(text=prompts, generation_config=config, seed=42)

# Display results
for i, prompt in enumerate(prompts):
    console.print(Panel(Text(prompt, style="italic"), title=f"Prompt {i+1}", border_style="blue"))
    
    safe_text = factory.tcs_safe.tokenizer.decode(output_safe[i], skip_special_tokens=True)[len(prompt):]
    risky_text = factory.tcs_risky.tokenizer.decode(output_risky[i], skip_special_tokens=True)[len(prompt):]
    proximal_text = output_proximal.text[i][len(prompt):]
    
    console.print(Panel(Text(safe_text, style="bold yellow"), title="Safe Model", border_style="yellow"))
    console.print(Panel(Text(risky_text, style="bold red"), title="Risky Model", border_style="red"))
    console.print(Panel(Text(proximal_text, style="bold green"), title="Proximal Decoding", border_style="green"))
    console.print()
