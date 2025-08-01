import torch
import numpy as np
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer

# Helper funcitons
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
    token = torch.multinomial(probs, num_samples=1) # [batch*block, 1]
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
    confidence_threshold = 0.85,
    stopping_criteria_idx=None):

    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x, 
              attention_mask=cur_attn_mask, 
              position_ids=cur_position_ids, 
              past_key_values=past_key_values, 
              use_cache=True, 
              store_kv=True)

    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        cur_position_ids = position_ids[:, num_block*block_length:(num_block+1)*block_length]
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
                        first_mask_index = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                        transfer_index[j, first_mask_index:first_mask_index + num_transfer_tokens[step]] = True
                    else:
                        raise ValueError("No mask tokens found in the current block.")

            elif remasking_strategy == 'low_confidence_static':
                confidence = torch.where(mask_index, x0_p, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True

            elif remasking_strategy == 'low_confidence_dynamic':
                confidence = torch.where(mask_index, x0_p, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = high_conf_mask.sum()
                    if num_high_confidence >= num_transfer_tokens[step]:
                        transfer_index[j] = high_conf_mask
                    else:
                        _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

            cur_x[transfer_index] = x0[transfer_index]

        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
        if stopping_criteria_idx is not None and any(stop_idx in x[:,prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    return x


if __name__ == "__main__":
    print("Loading model...")
    # model_dir = "/xxx/SDAR-1.7B-Chat"
    # model_dir = "/xxx/SDAR-4B-Chat"
    model_dir = "xxx/SDAR-8B-Chat"
    # model_dir = "xxx/SDAR-30B-A3B-Chat"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    origin_prompt = [
        # dict(role="user", content="Given the function $f(x) = \\frac{4x^2 - 4x + 4}{x^2 + 2x + 4}$, where $x \\in \\mathbb{R}$, determine its minimum value.\nPlease reason step by step, and put your final answer within \\boxed{}.\n"),
        dict(role="user", content="If the domain of the function $\\log x^2$ is $x < a$ or $x > b$, for some $a$ and $b$, find $a + b$.\nPlease reason step by step, and put your final answer within \\boxed{}.\n")
    ]

    messages = tokenizer.apply_chat_template(origin_prompt, add_generation_prompt=True, tokenize=False)
    tokenize_kwargs = dict(
                return_tensors='pt',
                padding=True,
                truncation=True,
                add_special_tokens=False,
                max_length=20480)

    tokens = tokenizer.batch_encode_plus([messages], **tokenize_kwargs)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    # SamplingParameters
    block_length = 4
    denoising_steps = 4
    gen_length = 20480
    top_k = 0
    top_p = 1.0
    temperature=1.0
    confidence_threshold = 0.85
    remasking_strategy = 'low_confidence_dynamic' # low_confidence_static, squential

    output_ids = block_diffusion_generate(
        model,
        prompt=tokens,
        mask_id=tokenizer(tokenizer.mask_token)['input_ids'][0],
        gen_length=gen_length,
        block_length=denoising_steps,
        denoising_steps=block_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        remasking_strategy=remasking_strategy,
        stopping_criteria_idx=[151645, 151643]
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    cleaned_text = output_text.replace('<|MASK|>', '')
    print(cleaned_text)
