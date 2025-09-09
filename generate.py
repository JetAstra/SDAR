import argparse
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def top_k_logits(logits, k):
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                 -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # shape: [batch*block, vocab]
    assert probs.dim() == 2
    token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]
    token_prob = torch.gather(probs, -1, token)     # [batch*block, 1]

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


@torch.no_grad()
def block_diffusion_generate(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=8,
        denoising_steps=8,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='low_confidence_dynamic',
        confidence_threshold=0.85,
        eb_threshold=None,
        stopping_criteria_idx=None
    ):

    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:,
                                                       :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # Store kv cache
                model(cur_x,
                      attention_mask=cur_attn_mask,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True)
                break

            # Denosing
            logits = model(cur_x,
                           attention_mask=cur_attn_mask,
                           position_ids=cur_position_ids,
                           past_key_values=past_key_values,
                           use_cache=True,
                           store_kv=False).logits

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Sampling strategy
            if remasking_strategy == 'sequential':
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        first_mask_index = mask_index[j].nonzero(as_tuple=True)[
                            0].min().item()
                        transfer_index[j, first_mask_index:first_mask_index +
                                       num_transfer_tokens[step]] = True
                    else:
                        raise ValueError(
                            "No mask tokens found in the current block.")

            elif remasking_strategy == 'low_confidence_static':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(
                        confidence[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True

            elif remasking_strategy == 'low_confidence_dynamic':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = high_conf_mask.sum()
                    if num_high_confidence >= num_transfer_tokens[step]:
                        transfer_index[j] = high_conf_mask
                    else:
                        _, idx = torch.topk(
                            confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True
            elif remasking_strategy == "entropy_bounded":
                eps = 1e-12
                entropies = -(x0_p.clamp_min(eps) * (x0_p.clamp_min(eps)).log()).sum(dim=-1)
                entropies = torch.where(mask_index, entropies, torch.inf)
                ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
                cumsum = torch.cumsum(ent_sorted, dim=1)
                for j in range(x0_p.shape[0]):
                    k = torch.searchsorted(cumsum[j], torch.tensor(eb_threshold, device=x0_p.device), right=False).item()
                    k = max(1, min(k, int(mask_index[j].sum().item())))
                    selected_token_indices = order[j, :k]
                    transfer_index[j, selected_token_indices] = True
                
            else:
                raise ValueError(
                    f"Unknown remasking strategy: {remasking_strategy}")

            cur_x[transfer_index] = x0[transfer_index]

        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
        if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    return x


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pretrained model directory")
    parser.add_argument("--trust_remote_code", action='store_true')
    parser.add_argument("--mask_id", type=int, default=None,
                        help="Mask token id for Diffusion")
    parser.add_argument("--prompt_length", type=int, default=4096,
                        help="Maximum prompt length in tokens")
    parser.add_argument("--gen_length", type=int, default=20480,
                        help="Maximum generation length in tokens")
    parser.add_argument("--block_length", type=int, default=4,
                        help="Length of token block to replace each denoising step")
    parser.add_argument("--denoising_steps", type=int, default=4,
                        help="Number of denoising steps (iterations)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-K sampling (0 to disable)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-P sampling probability threshold")
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence_dynamic",
                        choices=["low_confidence_dynamic",
                                 "low_confidence_static",
                                 "sequential",
                                 "entropy_bounded"],
                        help="Strategy for remasking tokens")
    parser.add_argument("--confidence_threshold", type=float, default=0.85,
                        help="Confidence threshold for low-confidence remasking")
    parser.add_argument("--eb_threshold", type=float, default=0.35,
                        help="entropy threshold for entropy bounded sampling")
    parser.add_argument("--stopping_criteria_idx", type=int, nargs="+", default=None,
                        help="List of token IDs that stop generation (e.g. eos_token_id)")

    parser.add_argument("--device", type=str, default="cuda",)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"],)
    if args.remasking_strategy == "low_confidence_dynamic" and args.confidence_threshold is None:
        parser.error(
            "--confidence_threshold is required when --remasking_strategy=low_confidence_dynamic"
        )
    if args.remasking_strategy == "entropy_bounded" and args.eb_threshold is None:
        parser.error(
            "--eb_threshold is required when --remasking_strategy=entropy_bounded"
        )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.dtype,
        device_map=args.device
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
    )

    if args.mask_id is None:
        args.mask_id = tokenizer(tokenizer.mask_token)['input_ids'][0]
    if args.stopping_criteria_idx is None:
        gen_cfg = GenerationConfig.from_pretrained(args.model_dir,)
        args.stopping_criteria_idx = gen_cfg.eos_token_id
    if isinstance(args.stopping_criteria_idx, int):
        args.stopping_criteria_idx = [args.stopping_criteria_idx,]
    args.stop_words = tokenizer.convert_ids_to_tokens(
        args.stopping_criteria_idx)
    print(f"Your Arguments: {args}")

    origin_prompt = [
        # dict(role="user", content="Given the function $f(x) = \\frac{4x^2 - 4x + 4}{x^2 + 2x + 4}$, where $x \\in \\mathbb{R}$, determine its minimum value.\nPlease reason step by step, and put your final answer within \\boxed{}.\n"),
        dict(role="user", content="If the domain of the function $\\log x^2$ is $x < a$ or $x > b$, for some $a$ and $b$, find $a + b$.\nPlease reason step by step, and put your final answer within \\boxed{}.\n")
    ]

    messages = tokenizer.apply_chat_template(
        origin_prompt, add_generation_prompt=True, tokenize=False)
    tokenize_kwargs = dict(
        return_tensors='pt',
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=args.prompt_length
    )

    tokens = tokenizer.batch_encode_plus([messages], **tokenize_kwargs)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    output_ids = block_diffusion_generate(
        model,
        prompt=tokens,
        mask_id=args.mask_id,
        gen_length=args.gen_length,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        remasking_strategy=args.remasking_strategy,
        confidence_threshold=args.confidence_threshold,
        eb_threshold=args.eb_threshold,
        stopping_criteria_idx=args.stopping_criteria_idx
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    cleaned_text = output_text.replace('<|MASK|>', '')
    print(cleaned_text)
