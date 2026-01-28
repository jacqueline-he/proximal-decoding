from typing import List, Optional, Union
from dataclasses import dataclass
import random
import numpy as np 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .byte_sampling.byte_conditioning import ByteConditioning 
from .byte_utils import safe_kl_pd_pc, solve_optimization_newton
from .byte_sampling.utils import sample_from_logits, sample_from_prob_tree

TOKEN_TO_BYTE = 4

@dataclass
class BytewiseGenerateOutput:
    """Output container for bytewise generation, mimics transformers GenerateDecoderOnlyOutput."""
    sequences: torch.Tensor
    text: List[str]


class BytewiseProximalDecodingFactory:
    def __init__(self, tcs_safe: ByteConditioning, tcs_risky: ByteConditioning, k_radius: float = 1.0, **kwargs):
        self.tcs_safe = tcs_safe
        self.tcs_risky = tcs_risky
        self.k_radius = k_radius
        self.kwargs = kwargs
        self._last_sampler = None

    @classmethod
    def from_pretrained(
        cls,
        safe_model_path: Optional[str] = None,
        risky_model_path: Optional[str] = None,
        safe_model: Optional[torch.nn.Module] = None,
        risky_model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        k_radius: float = 1.0,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Create a BytewiseProximalDecodingFactory from model paths or pre-loaded models.
        
        Args:
            safe_model_path: Path to safe model (used if safe_model not provided)
            risky_model_path: Path to risky model (used if risky_model not provided)
            safe_model: Pre-loaded safe model
            risky_model: Pre-loaded risky model
            tokenizer: Tokenizer (if None, loaded from safe_model_path)
            k_radius: KL budget radius
            device: Device to use
            torch_dtype: Torch dtype for models
            device_map: Device map for model loading
            trust_remote_code: Whether to trust remote code
            load_in_4bit: Load in 4-bit quantization
            load_in_8bit: Load in 8-bit quantization
            verbose: Print loading info
            **kwargs: Additional kwargs passed to ByteConditioning
        """
        load_kwargs = dict(
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
        
        # Create ByteConditioning for safe model
        if safe_model is not None:
            # Model object provided - need tokenizer
            safe_tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(safe_model_path or safe_model.config._name_or_path)
            tcs_safe = ByteConditioning(safe_model, tokenizer=safe_tokenizer)
            if verbose:
                print(f"[INFO] Created ByteConditioning for safe model (from object)")
        elif safe_model_path is not None:
            # Load from path
            tcs_safe = ByteConditioning(safe_model_path, load_kwargs=load_kwargs)
            if verbose:
                print(f"[INFO] Created ByteConditioning for safe model from: {safe_model_path}")
        else:
            raise ValueError("Either safe_model or safe_model_path must be provided")
        
        # Create ByteConditioning for risky model
        if risky_model is not None:
            # Model object provided - need its own tokenizer
            risky_tokenizer = AutoTokenizer.from_pretrained(risky_model_path or risky_model.config._name_or_path)
            tcs_risky = ByteConditioning(risky_model, tokenizer=risky_tokenizer)
            if verbose:
                print(f"[INFO] Created ByteConditioning for risky model (from object)")
        elif risky_model_path is not None:
            # Load from path
            tcs_risky = ByteConditioning(risky_model_path, load_kwargs=load_kwargs)
            if verbose:
                print(f"[INFO] Created ByteConditioning for risky model from: {risky_model_path}")
        else:
            raise ValueError("Either risky_model or risky_model_path must be provided")
        
        if verbose:
            print(f"[INFO] BytewiseProximalDecodingFactory initialized with k_radius={k_radius}")
        
        return cls(tcs_safe=tcs_safe, tcs_risky=tcs_risky, k_radius=k_radius, **kwargs)

    def get_bytewise_sampler(self, batch_size):
        sampler = BytewiseProximalDecoding(
            batch_size, 
            tcs_safe=self.tcs_safe, 
            tcs_risky=self.tcs_risky, 
            k_radius=self.k_radius, 
            **self.kwargs
        )
        self._last_sampler = sampler 
        return sampler
    
    def generate(
        self,
        text: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        max_new_tokens: int = 100,
        do_sample: bool = True,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        k_radius: Optional[float] = None,
        stop_strings: tuple = (),
        use_prefix_debt: bool = True,
        prefix_debt_n: int = 5,
        log_kl_stats: bool = False,
        **kwargs,
    ) -> BytewiseGenerateOutput:
        """
        Generate text using bytewise proximal decoding.
        
        Args:
            text: List of prompt strings
            input_ids: Alternative to text - token IDs (will be decoded to text)
            generation_config: HuggingFace GenerationConfig (extracts max_new_tokens, temperature, etc.)
            max_new_tokens: Maximum new tokens to generate
            do_sample: Whether to sample (vs greedy)
            temperature: Sampling temperature
            seed: Random seed
            k_radius: Override factory k_radius for this generation
            stop_strings: Strings to stop generation at
            use_prefix_debt: Whether to use prefix debt
            prefix_debt_n: Number of positions for prefix debt calculation
            log_kl_stats: Whether to log KL statistics
            **kwargs: Additional kwargs passed to generate_byte
            
        Returns:
            BytewiseGenerateOutput with .sequences (token tensors) and .text (decoded strings)
        """
        # Extract settings from generation_config if provided
        if generation_config is not None:
            max_new_tokens = getattr(generation_config, 'max_new_tokens', max_new_tokens) or max_new_tokens
            do_sample = getattr(generation_config, 'do_sample', do_sample)
            temperature = getattr(generation_config, 'temperature', temperature) or temperature
        
        # Handle input_ids -> text conversion
        if text is None:
            if input_ids is None:
                raise ValueError("Either text or input_ids must be provided")
            # Decode input_ids to text using safe model's tokenizer
            text = self.tcs_safe.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Use factory k_radius if not overridden
        if k_radius is None:
            k_radius = self.k_radius
        
        # Temporarily update factory k_radius for this generation
        original_k_radius = self.k_radius
        self.k_radius = k_radius
        
        try:
            # Call generate_byte
            outputs = generate_byte(
                sampler_factory=self,
                prompts=text,
                max_new_bytes=max_new_tokens * TOKEN_TO_BYTE,  # Approximate bytes from tokens
                do_sample=do_sample,
                temperature=temperature,
                seed=seed,
                stop_strings=stop_strings,
                use_prefix_debt=use_prefix_debt,
                prefix_debt_n=prefix_debt_n,
                log_kl_stats=log_kl_stats,
                **kwargs,
            )
        finally:
            self.k_radius = original_k_radius
        
        # Combine prompts with generated text
        full_texts = [prompt + output for prompt, output in zip(text, outputs)]
        
        # Tokenize full outputs for .sequences compatibility
        tokenized = self.tcs_safe.tokenizer(
            full_texts, 
            return_tensors="pt", 
            padding=True,
            truncation=False,
        )
        
        return BytewiseGenerateOutput(
            sequences=tokenized.input_ids,
            text=full_texts,
        )


class BytewiseProximalDecoding:
    def __init__(self, batch_size, tcs_safe, tcs_risky, k_radius=1.0, log_kl_stats=False, **kwargs):
        self.batch_size = batch_size
        self.tcs_safe = tcs_safe
        self.tcs_risky = tcs_risky
        self.k_radius = k_radius
        self.kwargs = kwargs
        self.bs_safe = tcs_safe.get_bytewise_sampler(batch_size=batch_size)
        self.bs_risky = tcs_risky.get_bytewise_sampler(batch_size=batch_size)

        self.bss = [self.bs_safe, self.bs_risky]
        self.kwargs = kwargs
        self.eps_kl = 1e-3
        
        # KL statistics logging
        self.log_kl_stats = log_kl_stats
        self.kl_stats_history = []

    def get_k_radius(self):
        return self.k_radius

    # apply log transforms, then solve for fused distribution
    def get_dists(self, k_radius=None, return_components:bool = False, save_kl_to_safe: bool = False, mask_special_tokens: bool = False, **kwargs):
        if k_radius is None:
            k_radius = self.k_radius # this is the local setting

        if isinstance(k_radius, (int, float)) and not save_kl_to_safe:
            if k_radius == 0.0: # return safe
                safe_logits = self.bs_safe.get_dists(**kwargs)
                return torch.log_softmax(safe_logits, -1)
            elif k_radius == -1.0: # return risky
                risky_logits = self.bs_risky.get_dists(**kwargs)
                risky_logp = torch.log_softmax(risky_logits, -1)
                if return_components:
                    safe_logits = self.bs_safe.get_dists(**kwargs)
                    safe_logp = torch.log_softmax(safe_logits, -1)
                    return risky_logp, safe_logp
                else:
                    return risky_logp


        safe_logits = self.bs_safe.get_dists(**kwargs).float()
        risky_logits = self.bs_risky.get_dists(**kwargs).float()

        if save_kl_to_safe: # always decode w/ risky model only
            log_pd = torch.log_softmax(risky_logits[:, :256], dim=-1)  # [B,256]
            log_pc = torch.log_softmax(safe_logits[:, :256], dim=-1)  # [B,256]
            kl_to_safe = safe_kl_pd_pc(log_pd, log_pc)

            return risky_logits, kl_to_safe

        if safe_logits.dim() == 3:
            safe_logits = safe_logits[:, -1, :]   # [B, V]
            risky_logits = risky_logits[:, -1, :]   # [B, V]
        elif safe_logits.dim() != 2:
            raise ValueError(f"Unsupported logits shape: {safe_logits.shape}")

        # Define a fused device; move computation there
        fuse_device = risky_logits.device
        safe_logits = safe_logits.to(fuse_device)
        risky_logits = risky_logits.to(fuse_device)

        if mask_special_tokens:
            safe_logits[:, 256:] = float("-inf")
            risky_logits[:, 256:] = float("-inf")

        valid = torch.isfinite(safe_logits) & torch.isfinite(risky_logits)

        # If your solve_optimization expects logits, keep them masked
        safe_m = safe_logits.masked_fill(~valid, float("-inf"))
        risky_m = risky_logits.masked_fill(~valid, float("-inf"))

        bc, bd = solve_optimization_newton(safe_m, risky_m, k_radius)

        # Ensure fusion weights bc/bd live with your logits
        bc = torch.as_tensor(bc, device=fuse_device, dtype=risky_m.dtype)
        bd = torch.as_tensor(bd, device=fuse_device, dtype=risky_m.dtype)

        # Make the assertion device/dtype-safe
        assert torch.allclose(bc + bd, torch.ones_like(bc), atol=1e-5, rtol=1e-5)

        bc = bc.view(-1, 1) if bc.dim() == 1 else bc
        bd = bd.view(-1, 1) if bd.dim() == 1 else bd

        # Geometric fusion: logits = w_c * logits_c + w_d * logits_d
        # Handle -inf in logits by temporarily replacing with 0, then re-masking
        safe_safe = torch.nan_to_num(safe_m, neginf=0.0)
        risky_safe = torch.nan_to_num(risky_m, neginf=0.0)
        fused_logits = bc * safe_safe + bd * risky_safe
        fused_logits = fused_logits.masked_fill(~valid, float("-inf"))

        fused_log_probs = torch.log_softmax(fused_logits, dim=-1)
        if return_components:
            safe_log_probs = torch.log_softmax(safe_m, -1)
            risky_log_probs = torch.log_softmax(risky_m, -1)
            return fused_log_probs, safe_log_probs, risky_log_probs, bc, bd
        return fused_log_probs

    def add_context(self, prompts: list[Union[str, bytes]]):
        for bs in self.bss:
            bs.add_context(prompts)


@torch.inference_mode()
def generate_byte(
    sampler_factory,
    prompts: list[str],
    min_new_bytes: int = 0,
    max_new_bytes: int = 100,
    do_sample: bool = True,
    temperature: float = 1,
    top_k: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    display: bool = False,
    stop_strings: tuple[str] = (),
    include_stop_str_in_output: bool = False,
    save_kl_to_safe: bool = False,
    use_prefix_debt: bool = True,
    use_precomputed_llrs: list[dict] = None,
    prefix_debt_n: int = None,
    scale_prefix_debt: bool = True,
    allow_special: bool = True,
    logprob_transforms=None,
    log_kl_stats: bool = False,
):
    assert not isinstance(
        stop_strings, str
    ), "stop_strings should be a sequence of strings"
    stop_strings = tuple(sorted(stop_strings, key=len, reverse=True))
    assert not isinstance(prompts, str)
    assert seed is None or generator is None, "can pass only one of seed/generator"

    bsize = len(prompts)
    assert not (display and bsize > 1)

    try:
        bs = sampler_factory.get_bytewise_sampler(batch_size=bsize)
    except AttributeError:
        bs = sampler_factory

    if log_kl_stats:
        bs.log_kl_stats = True
        bs.kl_stats_history = []

    device = bs.tcs_safe.device
    bs.add_context([prompt.encode() for prompt in prompts])

    unfinished_sequences = torch.ones(bsize, device=device, dtype=torch.long)

    outputs = [[] for _ in range(bsize)]
    decode_bufs = [b"" for _ in range(bsize)]
    stop_found = [False for _ in range(bsize)]

    if display:
        print(prompts[0], end="", flush=True)

    if save_kl_to_safe:
        kl_to_safe_history = []

    ## k
    k_radius = bs.get_k_radius()
    print(f"[INFO] k_radius: {k_radius}")
    assert k_radius != -1.0 and k_radius != 0.0, "Use generate_batched instead for your k-radius"
    cum_kl_spent = torch.zeros(bsize, device=device, dtype=torch.float32)

    if use_prefix_debt and prefix_debt_n is not None and prefix_debt_n > 0:
        print(f"[INFO] Using prefix debt from prompt")
        if use_precomputed_llrs is not None:
            print(f"[INFO] Using precomputed llrs (batch of {len(use_precomputed_llrs)} items)")
            prefix_debt = get_prefix_debt_from_llrs(use_precomputed_llrs, prompts, n=prefix_debt_n)
        else:
            print(f"[INFO] Computing llrs from scratch ")
            prefix_debt = get_prefix_debt_bytewise(bs, prompts, n=prefix_debt_n)
        if scale_prefix_debt:
            prefix_debt *= TOKEN_TO_BYTE
            print(f"[INFO] Scaled prefix debt: {prefix_debt}")
        else:
            print(f"[INFO] prefix debt: {prefix_debt}")
        # assert prefix debt is negative, <0

    else:
        prefix_debt = torch.zeros(bsize, device=device, dtype=torch.float32)

    for t_gen in range(max_new_bytes):

        try:
            if save_kl_to_safe:
                assert k_radius == -1, "kl_to_safe only works with k=-1"
                dists, kl_to_safe_list = bs.get_dists(logprob_transforms=logprob_transforms, save_kl_to_safe=True)
                kl_to_safe_history.append(kl_to_safe_list)
            else:
                # Compute k_t:
                budget_so_far = (float(t_gen +1)*float(k_radius)) - prefix_debt
                remaining = (budget_so_far - cum_kl_spent).clamp(min=0.0) # should be fp32
                k_t = remaining * unfinished_sequences.float()
                # print(f"k_t: {k_t}")
                # Use k_t to solve fused distribution
                # Pass mask_special_tokens so find_lambda enforces KL constraint on byte-only vocab
                should_mask = not allow_special or t_gen < min_new_bytes
                dists, safe_log_probs, risky_log_probs, bc, bd = bs.get_dists(
                    logprob_transforms=logprob_transforms,
                    k_radius=k_t,
                    return_components=True,
                    mask_special_tokens=should_mask
                )
                # Compute KL immediately on the distributions returned by get_dists
                # (before any further modifications)
                kl_step = safe_kl_pd_pc(dists, safe_log_probs).float()
                
                # Log KL statistics if enabled
                if getattr(bs, 'log_kl_stats', False):
                    kl_to_risky = safe_kl_pd_pc(dists, risky_log_probs).float()
                    kl_pd_pc = safe_kl_pd_pc(risky_log_probs, safe_log_probs).float()
                    
                    bs.kl_stats_history.append({
                        'step': t_gen,
                        'kl_to_safe': kl_step.detach().cpu().tolist(),
                        'kl_to_risky': kl_to_risky.detach().cpu().tolist(),
                        'kl_pd_pc': kl_pd_pc.detach().cpu().tolist(),
                        'bc': bc.squeeze(-1).detach().cpu().tolist() if bc.dim() > 1 else bc.detach().cpu().tolist(),
                        'bd': bd.squeeze(-1).detach().cpu().tolist() if bd.dim() > 1 else bd.detach().cpu().tolist(),
                        'k_t': k_t.detach().cpu().tolist(),
                        'cum_kl_spent': (cum_kl_spent + kl_step * unfinished_sequences.float()).detach().cpu().tolist(),
                        'budget_so_far': budget_so_far.detach().cpu().tolist() if isinstance(budget_so_far, torch.Tensor) else [budget_so_far] * bsize,
                    })
        except RecursionError:
            # Tree got too deep, use uniform distribution over bytes and continue
            print(f"[generate_batched_adaptive] RecursionError at step {t_gen}, using fallback distribution")
            dists = torch.zeros(bsize, 259, device=device)
            dists[:, :256] = 0.0  # Uniform over bytes
            dists[:, 256:] = -torch.inf  # Mask special tokens
            if save_kl_to_safe:
                kl_to_safe_list = torch.zeros(bsize, device=device)
                kl_to_safe_history.append(kl_to_safe_list)
            else:
                # For non-save_kl mode, set safe_log_probs and kl_step to safe defaults
                safe_log_probs = dists.clone()
                kl_step = torch.zeros(bsize, device=device, dtype=torch.float32)

        # Apply transformations (e.g., zeroing special tokens) for sampling
        # Note: When mask_special_tokens=True was passed to get_dists, the logits were
        # already masked before log_softmax, so dists is already correct for byte-only.
        # However, we still need to ensure dists has -inf for special tokens for sampling.
        if not allow_special or t_gen < min_new_bytes:
            dists[:, 256:] = -torch.inf

        # init the generator late so we know which device to put it on
        if generator is None and seed is not None:
            generator = torch.Generator(device=dists.device).manual_seed(seed)

        new_bytes = sample_from_logits(
            dists,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        ).tolist()

        for i, new_byte in enumerate(new_bytes):
            if new_byte >= 256 and t_gen >= min_new_bytes:
                stop_found[i] = True

        new_bytes = [
            bytes([b]) if not sf else bytes() for b, sf in zip(new_bytes, stop_found)
        ]

        bs.add_context(new_bytes)

        for i, new_byte in enumerate(new_bytes):
            if stop_found[i]:
                continue
            try:
                decode_bufs[i] += new_byte
                char = decode_bufs[i].decode()
                outputs[i].append(char)
                if display:
                    print(char, end="", flush=True)
                decode_bufs[i] = b""
            except UnicodeDecodeError:
                pass

        if stop_strings:
            for i, output in enumerate(outputs):
                if stop_found[i]:
                    continue

                suffix = "".join(output[-max(map(len, stop_strings)) :])
                if suffix.endswith(stop_strings):
                    if t_gen < min_new_bytes:
                        continue
                    if not include_stop_str_in_output:
                        for stop in stop_strings:
                            if suffix.endswith(stop):
                                outputs[i] = output[: -len(stop)]
                                break

                    stop_found[i] = True

        if all(stop_found):
            break

        # Update cum_kl_spent and BANK the KL spend (only when not in save_kl_to_safe mode)
        # NOTE: kl_step was computed BEFORE modifying dists (zeroing special tokens)
        if not save_kl_to_safe:
            mask = unfinished_sequences.bool()
            assert torch.all(kl_step[mask] <= k_t[mask] + bs.eps_kl), \
                f"Local KL violated: kl_step[mask]={kl_step[mask].tolist()}, k_t[mask]={k_t[mask].tolist()}, diff={(kl_step[mask] - k_t[mask]).tolist()}, eps={bs.eps_kl}"
            cum_kl_spent = cum_kl_spent + kl_step * unfinished_sequences.float()

    output = ["".join(output) for output in outputs]
    if save_kl_to_safe:
        return output, kl_to_safe_history
    else:
        return output


def get_prefix_debt_bytewise(bs, prompts: list[Union[str, bytes]], n: int = 7):
    """
    Computes prefix debt (average top-n LLR) at the byte level for the prompt.

    For each byte position in the prompt, we compute:
      LLR = log p_risky(byte) - log p_safe(byte)

    Then return the average of the top-n LLR values (clamped to 0).

    This is the byte-level analogue of get_prefix_debt_v2: we score each actual
    byte in the prompt, not just the distribution at the final position.

    This version uses the proper ByteConditioning interface (get_dists) which
    handles token-to-byte conversion through BPE tree decomposition. This works
    correctly for any token-level model, not just byte-aligned models.

    Args:
        bs: BytewiseKLAcpFuse sampler
        prompts: List of prompts (strings or bytes) for each batch element
        n: number of top LLR values to average

    Returns:
        tensor of shape [batch_size] with prefix debt for each sequence
    """
    bsize = bs.batch_size
    device = bs.tcs_safe.device

    # Ensure prompts are bytes
    prompts_bytes = [p.encode() if isinstance(p, str) else p for p in prompts]

    # Create fresh samplers to process prompts byte-by-byte
    bs_safe_temp = bs.tcs_safe.get_bytewise_sampler(batch_size=bsize)
    bs_risky_temp = bs.tcs_risky.get_bytewise_sampler(batch_size=bsize)

    # Find max length to iterate
    max_len = max(len(p) for p in prompts_bytes)

    if max_len == 0:
        return torch.zeros(bsize, device=device)

    # Collect per-position LLRs for actual bytes
    all_llrs = []

    for pos in range(max_len):
        # Get byte-level distributions from ByteConditioning
        # get_dists() returns log-probs over 256 bytes + special tokens
        safe_dists = bs_safe_temp.get_dists()
        risky_dists = bs_risky_temp.get_dists()

        # Extract log-probs for bytes only (first 256 positions)
        safe_logprobs = torch.log_softmax(safe_dists[:, :256].float(), dim=-1)
        risky_logprobs = torch.log_softmax(risky_dists[:, :256].float(), dim=-1)

        # For each sample, get LLR for the actual byte
        # Use -inf for invalid positions (will be masked later)
        llr_pos = torch.full((bsize,), -torch.inf, device=device, dtype=torch.float32)
        current_bytes = []

        for i, prompt in enumerate(prompts_bytes):
            if pos < len(prompt):
                byte_idx = prompt[pos]
                llr_pos[i] = risky_logprobs[i, byte_idx] - safe_logprobs[i, byte_idx]
                current_bytes.append(bytes([byte_idx]))
            else:
                current_bytes.append(b'')

        all_llrs.append(llr_pos)

        # Add the bytes as context for next position
        bs_safe_temp.add_context(current_bytes)
        bs_risky_temp.add_context(current_bytes)

    # Stack LLRs: [bsize, max_len]
    llr_matrix = torch.stack(all_llrs, dim=1)

    # Build validity mask: exclude first byte (pos=0) per the algorithm
    # valid[i, j] = True if j >= 1 and j < len(prompts_bytes[i])
    valid = torch.zeros((bsize, max_len), device=device, dtype=torch.bool)
    for i, p in enumerate(prompts_bytes):
        L = len(p)
        if L >= 2:
            valid[i, 1:L] = True

    # Mask invalid positions
    llr_matrix = llr_matrix.masked_fill(~valid, -torch.inf)

    # per-example top-n over valid positions only
    valid_count = valid.sum(dim=-1)  # [B]
    k_i = torch.minimum(valid_count, torch.tensor(n, device=device)).clamp(min=1)
    k_max = int(k_i.max().item())

    if k_max == 0:
        return torch.zeros(bsize, device=device)

    topv, _ = llr_matrix.topk(k_max, dim=-1)  # [B, k_max]
    topv = topv.clamp(min=0.0)

    jj = torch.arange(k_max, device=device).unsqueeze(0)  # [1, k_max]
    keep = jj < k_i.unsqueeze(1)
    topv = topv.masked_fill(~keep, 0.0)

    return topv.sum(dim=-1) / k_i.float()
