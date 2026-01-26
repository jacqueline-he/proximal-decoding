from proximaldecode import ProximalDecodingFactory, generate
from transformers import GenerationConfig
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
import time

console = Console()

factory = ProximalDecodingFactory.from_pretrained(
    safe_model_path="jacquelinehe/comma-1.7b-v5",
    risky_model_path="meta-llama/Llama-3.1-8B",
    k_radius=0.15,
    device='cuda',
)

prompts = [
    "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. ",
    "While I was still in Amsterdam, I dreamed about my mother for the first time in years. I'd been shut up in my hotel for more than a week, afraid to telephone anybody or go out;"
]

config = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
)

with console.status("[bold green]Generating with Proximal Decoding...", spinner="dots"):
    output_text = factory.generate(text=prompts, generation_config=config, seed=42)

for i, prompt in enumerate(prompts):
    console.print(Panel(Text(prompt, style="italic"), title=f"Prompt {i+1}", border_style="blue"))
    decoded = factory.tokenizer.decode(output_text.sequences[i], skip_special_tokens=True)
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]
    console.print(Panel(Text(decoded, style="bold green"), title=f"Generated Response {i+1}", border_style="green"))
    console.print("\n")
