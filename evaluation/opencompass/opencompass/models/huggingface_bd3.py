# flake8: noqa
# yapf: disable
from typing import Dict, List, Optional, Union

from einops import rearrange
import torch
import numpy as np
from torch.nn import functional as F
from mmengine.device import is_npu_available

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


## bd3 generation
import torch
from torch import block_diag
from torch.nn import functional as F
import numpy as np
from transformers.cache_utils import (
    Cache,
    DynamicCache,
)

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def top_k_logits(logits, k):
    """ 保留概率最大的k个值，其他的设为-inf以屏蔽 """
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

def top_p_logits(logits, p):
    """ nucleus sampling: 保留累积概率大于p的前几个token，其他设为-inf """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # 获得大于p的第一个index并截断
    sorted_mask = cumulative_probs > p
    # 保证至少保留一个token
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

    # 1️⃣ 原始概率：先应用温度，然后 softmax
    assert temperature > 0, "Temperature must be positive"
    logits = logits / temperature if temperature != 1.0 else logits
    ori_probs = F.softmax(logits, dim=-1)  # 用于置信度排序

    # 2️⃣ 再做 top-k / top-p 筛选用于采样
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    assert probs.dim() == 2

    # 3️⃣ multinomial 采样
    token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]

    # 4️⃣ 获取 token 在原始概率分布里的概率
    token_prob = torch.gather(ori_probs, -1, token)

    return token.view(*orig_shape), token_prob.view(*orig_shape)

def get_num_transfer_tokens(block_length, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    base = block_length // steps
    remainder = block_length % steps

    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1

    return num_transfer_tokens

@torch.no_grad()
def block_diffusion_generate(
    model,
    tokenizer,
    prompt,
    mask_id,
    gen_length=128,
    block_length=8,
    denoising_steps=8, 
    temperature=0.,
    top_k=0,
    top_p=1.0,
    remasking='low_confidence',
    threshold=1.0,
    stopping_criteria_idx=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        denoising_steps: Sampling steps, less than or equal to block_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK]
    '''

    # 仅仅支持单个输入，不支持 batch inference
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    attention_mask = prompt['attention_mask']
    tokenizer = tokenizer
    past_key_values = DynamicCache()

    # prepare block_diag and position_ids
    num_blocks = (input_ids.shape[1] + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length
    # 生成块级下三角掩码（num_blocks × num_blocks）
    block_mask = torch.tril(
        torch.ones(num_blocks, num_blocks, device=model.device),
        diagonal=0
    )
    # 扩展到每个块内部（变成 total_length × total_length）
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0) \
                                .repeat_interleave(block_length, dim=1) \
                                .unsqueeze(0)  # 添加 batch 维度

    position_ids = torch.arange(
        0, total_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)

    # prepare input_ids
    x = torch.full((input_ids.shape[0], total_length),
                   mask_id, dtype=torch.long).to(model.device)
    x[:, :input_ids.shape[1]] = input_ids.clone()

    # calculate prefill_length
    prefill_blocks = input_ids.shape[1] // block_length
    prefill_length = prefill_blocks * block_length

    # prefill stage
    if prefill_length > 0:
        # 预填充阶段，计算得到 kv cache
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        _ = model(
            input_ids=cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values, # 这是一个全局变量
            use_cache=True,
            store_kv=True)

    # generate output recursively
    assert block_length % denoising_steps == 0 and block_length >= denoising_steps
    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)
    end_generate = False
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block * block_length: (num_block + 1) * block_length]
        cur_attn_mask = block_diffusion_attention_mask[:, num_block *
            block_length: (num_block + 1) * block_length, :(num_block + 1) * block_length]
        cur_position_ids = position_ids[:, num_block * block_length: (num_block + 1) * block_length]
        for i in range(denoising_steps + 1): # 有一步需要专门用于缓存 kv cache，后续可以优化掉
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                _ = model(
                    input_ids=cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True)
                break
            else:
                logits = model(
                    input_ids=cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False).logits

                logits_with_noise = add_gumbel_noise(
                    logits, temperature=temperature)
                # x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == 'low_confidence':
                    # p = F.softmax(logits, dim=-1)
                    # x0_p = torch.squeeze(
                    #     torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
                    x0, x0_p = sample_with_temperature_topk_topp(
                        logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                elif remasking == 'random':
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0 = torch.where(mask_index, x0, cur_x)
                if threshold < 1.0:
                    # 动态解码过程
                    confidence = torch.where(mask_index, x0_p, -np.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

                    for j in range(confidence.shape[0]):
                        # 找到满足置信度阈值的token
                        high_conf_mask = confidence[j] > threshold
                        num_high_confidence = high_conf_mask.sum()

                        if num_high_confidence >= num_transfer_tokens[i]:
                            # 如果高置信度数量已经超过或等于num_transfer_tokens，直接使用这些token
                            transfer_index[j] = high_conf_mask
                        else:
                            # 当高置信度tokens数量不够时，直接简单地从整体confidence中选取topk即可
                            _, idx = torch.topk(confidence[j], num_transfer_tokens[i])
                            transfer_index[j, idx] = True
                else:
                    # 静态解码过程
                    confidence = torch.where(mask_index, x0_p, -np.inf)

                    transfer_index = torch.zeros_like(
                        x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        _, select_index = torch.topk(
                            confidence[j], k=num_transfer_tokens[i])
                        transfer_index[j, select_index] = True

                cur_x[transfer_index] = x0[transfer_index]

        x[:, num_block * block_length: (num_block + 1) * block_length] = cur_x
        for stop_idx in stopping_criteria_idx:
            if stop_idx in x[:, prompt_length:]:
                end_generate = True
                break
        if end_generate:
            break

    return x


def forward_add_noise(
    input_ids: torch.Tensor, #(b, l)
    eps: float,
    mask_token: int,
    ignore_index: int = -100
):
    bsz, length = input_ids.shape
    t = torch.rand(bsz, device=input_ids.device)  # U[0,1]
    p_mask = (1 - eps) * t + eps  # avoid devision by zero
    p_mask = p_mask[:, None].repeat(1, length)
    # True is masked
    masked_indices = torch.rand(
        (bsz, length), device=input_ids.device) < p_mask
    # `noisy_inputs.shape` is same as `input_ids`, some tokens are replaced by `mask_token`
    noisy_inputs = torch.where(masked_indices, mask_token, input_ids)
    # 需要被预测的 labels, 对应的地方变为 -100
    masked_labels = torch.where(masked_indices, input_ids, ignore_index)
    return noisy_inputs, masked_labels, p_mask  # minibsz, length

def _get_stopping_criteria(stop_words, tokenizer, batch_size):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class MultiTokenEOSCriteria(StoppingCriteria):
        """Criteria to stop on the specified multi-token sequence."""

        def __init__(self, stop_words: List[str], tokenizer, batch_size: int):
            self.done_tracker = [False] * batch_size
            self.stop_words, self.max_sequence_id_len = [], 0
            for s in stop_words:
                self.stop_words.append(s)
                sequence_ids = tokenizer.encode(s, add_special_tokens=False)
                self.max_sequence_id_len = max(self.max_sequence_id_len, len(sequence_ids))
            self.tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            # compare the last len(stop) tokens
            lookback_ids_batch = input_ids[:, -self.max_sequence_id_len:]
            lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
            for i, done in enumerate(self.done_tracker):
                if done:
                    continue
                self.done_tracker[i] = any(s in lookback_tokens_batch[i] for s in self.stop_words)
            return False not in self.done_tracker

    c = MultiTokenEOSCriteria(stop_words, tokenizer, batch_size)
    return StoppingCriteriaList([c])

def _get_possible_max_seq_len(max_seq_len, path):
    if max_seq_len is not None:
        return max_seq_len

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    possible_keys = [
        'max_position_embeddings',
        'seq_length',
        'model_max_length',
        'max_sequence_length'
    ]
    for k in possible_keys:
        if hasattr(config, k):
            return getattr(config, k)
    raise ValueError('max_seq_len is not provided and cannot be inferred from the model config.')


def _convert_chat_messages(inputs, merge_role=True, skip_empty_prompt=True):
    outputs = []
    for _input in inputs:
        messages = []
        if isinstance(_input, str):
            messages.append({'role': 'user', 'content': _input})
        else:
            for item in _input:
                if skip_empty_prompt and not item['prompt']:
                    continue
                role = {
                    'HUMAN': 'user',
                    'BOT': 'assistant',
                    'SYSTEM': 'system',
                }[item['role']]
                messages.append({'role': role, 'content': item['prompt']})

        if merge_role:
            merged_messages = []
            for item in messages:
                if merged_messages and merged_messages[-1]['role'] == item['role']:
                    merged_messages[-1]['content'] += '\n' + item['content']
                else:
                    merged_messages.append(item)
            messages = merged_messages

        outputs.append(messages)
    return outputs


def _format_with_fast_chat_template(inputs: List[str], name: str='vicuna'):
    try:
        from fastchat.model import get_conversation_template
    except ImportError:
        raise ModuleNotFoundError('fastchat not found. Please install with\npip install "fschat[model_worker,webui]"')

    outputs = []
    for _input in inputs:
        template = get_conversation_template(name)
        for item in _input:
            if item['role'] == 'user':
                template.append_message(template.roles[0], item['content'])
            elif item['role'] == 'assistant':
                template.append_message(template.roles[1], item['content'])
            elif item['role'] == 'system':
                continue
            else:
                raise ValueError(f"Unknown role {item['role']}")
        template.append_message(template.roles[1], None)
        outputs.append(template.get_prompt())
    return outputs


def _get_meta_template(meta_template):
    default_meta_template = dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            # XXX: all system roles are mapped to human in purpose
            dict(role='SYSTEM', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ]
    )
    return APITemplateParser(meta_template or default_meta_template)


def _set_model_kwargs_torch_dtype(model_kwargs):
    import torch
    if 'torch_dtype' not in model_kwargs:
        torch_dtype = torch.float16
    else:
        torch_dtype = {
            'torch.float16': torch.float16,
            'torch.bfloat16': torch.bfloat16,
            'torch.float': torch.float,
            'auto': 'auto',
            'None': None,
        }.get(model_kwargs['torch_dtype'])
    if torch_dtype is not None:
        model_kwargs['torch_dtype'] = torch_dtype
    return model_kwargs


@MODELS.register_module()
class BD3withChatTemplate(BaseModel):
    """Model wrapper for bd3 models designed for chat.

    Args:
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
    """

    def __init__(self,
                 path: str,
                 model_kwargs: dict = dict(),
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 peft_kwargs: dict = dict(),
                 tokenizer_only: bool = False,
                 generation_kwargs: dict = dict(),
                 max_seq_len: Optional[int] = None,
                 meta_template: Optional[Dict] = None,
                 pad_token_id: Optional[int] = None,
                 fastchat_template: Optional[str] = None,
                 stop_words: Optional[str] = [],
                 mode: str = 'none',
                 **other_kwargs):

        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        self._load_tokenizer(tokenizer_path or path, tokenizer_kwargs, pad_token_id)
        if not tokenizer_only:
            self._load_model(path=path, kwargs=model_kwargs, peft_path=peft_path, peft_kwargs=peft_kwargs)
        self.generation_kwargs = generation_kwargs
        self.fastchat_template = fastchat_template
        self.stop_words = list(set(stop_words + self._get_potential_stop_words(path)))
        assert mode in ['none', 'mid']
        self.mode = mode
        self.logger.info(f'using stop words: {self.stop_words}')

        for k, v in other_kwargs.items():
            if v is not None:
                self.logger.warning(f'Unused argument {k}={v}')

    def _load_tokenizer(self, path: Optional[str], kwargs: dict, pad_token_id: Optional[int] = None):
        from transformers import AutoTokenizer, GenerationConfig

        DEFAULT_TOKENIZER_KWARGS = dict(padding_side='left', truncation_side='left', trust_remote_code=True)
        tokenizer_kwargs = DEFAULT_TOKENIZER_KWARGS
        tokenizer_kwargs.update(kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)

        # A patch for some models without pad_token_id
        if pad_token_id is not None:
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f'Using {pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != pad_token_id:
                self.logger.warning(f'pad_token_id is not consistent. Using {pad_token_id} as pad_token_id')
            self.tokenizer.pad_token_id = pad_token_id
            return
        if self.tokenizer.pad_token_id is not None:
            return
        self.logger.warning('pad_token_id is not set for the tokenizer.')
        generation_config = GenerationConfig.from_pretrained(path)
        if generation_config.pad_token_id is not None:
            self.logger.warning(f'Using {generation_config.pad_token_id} as pad_token_id.')
            self.tokenizer.pad_token_id = generation_config.pad_token_id
            return
        if self.tokenizer.eos_token_id is not None:
            self.logger.warning(f'Using eos_token_id {self.tokenizer.eos_token_id} as pad_token_id.')
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            return
        raise ValueError('pad_token_id is not set for this tokenizer. Please set `pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        from transformers import AutoModel, AutoModelForCausalLM

        DEFAULT_MODEL_KWARGS = dict(device_map='cuda', trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS
        model_kwargs.update(kwargs)
        model_kwargs = _set_model_kwargs_torch_dtype(model_kwargs)
        self.logger.debug(f'using model_kwargs: {model_kwargs}')
        if is_npu_available():
            model_kwargs['device_map'] = 'npu'

        try:
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
            # self.model = self.model.to(torch.float16)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            peft_kwargs['is_trainable'] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        print(f"model configs:\n{self.model.config}")
        print(f"model 参数类型: {next(self.model.parameters()).dtype}")
        self.model.eval()
        self.model.generation_config.do_sample = False

    def get_ppl_tokenwise(self, inputs: List[str], label: List[List[int]], mask_length: Optional[List[int]] = None) -> List[float]:
        """Get inference-ppl per token given a list of inputs and label.

        Args:
            inputs (List[str]): A list of strings.
            label (List[List[int]]): A list of list of label, each label is a tuple of (start, end, 1)
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        assert self.tokenizer.pad_token
        import torch
        import torch.nn.functional as F
        pad_token_id = self.tokenizer.pad_token_id
        messages = _convert_base_messages(inputs)

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len,
        )

        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'

        tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        outputs = self.model(**tokens)[0]

        batch_size, seq_len, vocab_size = outputs.shape
        shift_logits = outputs[:, :-1, :].contiguous().float()
        shift_labels = tokens['input_ids'][:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
            reduction='none').view(batch_size, seq_len - 1)
        lens = (tokens['input_ids'] != pad_token_id).sum(-1).cpu().numpy()

        if mask_length is not None:
            import numpy as np
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask
            lens -= np.array(mask_length)

        loss = loss.cpu().numpy()

        decode_messages = [[self.tokenizer.decode([input_id]) for input_id in token] for token in tokens['input_ids']]
        char_messages = [[ch for ch in message] for message in messages]

        # shifted to align label and loss
        for i in range(len(decode_messages)):
            decode_messages[i] = decode_messages[i][1:]

        aggregated_label_list = [[] for _ in range(len(decode_messages))]

        tag_list = [[] for _ in range(len(decode_messages))]

        for tmp_index, label_list in enumerate(label):
            for single_label in label_list:
                left = single_label[0]
                right = single_label[1]
                for i in range(left, right):
                    aggregated_label_list[tmp_index].append(i)


        def align_sequences(seq1, seq2, sep_len):
            """
            seq1: decoded sequence from token, one token may contain multiple characters
            seq2: original separate character sequence
            """
            i, j = 0, 0
            matched_pairs = []
            while i < len(seq1) and j < len(seq2):
                word = seq1[i]
                if len(word) == 0:
                    matched_pairs.append((word, []))
                    i += 1
                    continue

                if '\ufffd' in word:
                    for _ in range(sep_len):
                        matched_pairs.append((word, [j]))
                        i += 1
                    j += 1
                    continue

                char_sequence = ''
                while j < len(seq2) and (char_sequence != word):
                    char_sequence += seq2[j]
                    if char_sequence == word:
                        matched_pairs.append((word, [k for k in range(j - len(word) + 1, j+1)]))
                        j += 1
                        break
                    elif len(char_sequence) > len(word):
                        if word == char_sequence[-len(word):]:
                            matched_pairs.append((word, [k for k in range(j - len(word) + 1, j+1)]))
                            j += 1
                            break
                        else:
                            j += 1
                    else:
                        j += 1
                i += 1

            return matched_pairs


        if 'qwen' in self.path or 'Qwen' in self.path:
            sep_len = 2
        elif 'Llama-3' in self.path:
            sep_len = 2
        elif 'Yi' in self.path:
            sep_len = 3
        elif 'Llama-2' in self.path:
            sep_len = 3
        elif 'deepseek' in self.path:
            sep_len = 2
        else:
            sep_len = 3


        matched_pairs_list = [align_sequences(decode_messages[i], char_messages[i], sep_len) for i in range(len(decode_messages))]
        for match_index, matched_pairs in enumerate(matched_pairs_list):
            for i, (word, indices) in enumerate(matched_pairs):
                for j in indices:
                    if j in aggregated_label_list[match_index]:
                        tag_list[match_index].append(i)
                        break

        inference_loss_list = []
        token_len_list = []
        for i in range(len(loss)):
            inference_loss = 0
            token_len = 0
            for j in range(len(loss[i])):
                if j in tag_list[i]:

                    inference_loss += loss[i][j]
                    print(loss[i][j])
                    token_len += 1
            inference_loss_list.append(inference_loss)
            token_len_list.append(token_len)

        return inference_loss_list, token_len_list

    def _get_potential_stop_words(self, path: Optional[str]):
        from transformers import GenerationConfig
        potential_stop_words = []
        try:
            generation_config = GenerationConfig.from_pretrained(path)
        except:
            generation_config = None
        if generation_config and hasattr(generation_config, 'eos_token_id'):
            if isinstance(generation_config.eos_token_id, int):
                potential_stop_words.append(self.tokenizer.decode(generation_config.eos_token_id))
            else:
                assert isinstance(generation_config.eos_token_id, list)
                for token_id in generation_config.eos_token_id:
                    potential_stop_words.append(self.tokenizer.decode(token_id))
        if self.tokenizer.eos_token is not None:
            potential_stop_words.append(self.tokenizer.eos_token)
        potential_stop_words = list(set(potential_stop_words))
        potential_stop_words = [s for s in potential_stop_words if s]
        return potential_stop_words

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                #  diffusion_kwargs: dict = dict(),
                 **kwargs) -> List[str]:
        messages = _convert_chat_messages(inputs)
        batch_size = len(messages)

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len
        )

        if self.fastchat_template:
            print("进入 fastchat_template")
            messages = _format_with_fast_chat_template(messages, self.fastchat_template)
            tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)
        else:
            messages = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]
            tokenize_kwargs['add_special_tokens'] = False
            tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)

        # list of tensors
        # inputs_ids = [torch.tensor(t, device=self.model.device) for t in tokens['input_ids']]

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

        if self.mode == 'mid':
            # Reserve space for the tokens to be generated in the future.
            max_prompt_len = self.max_seq_len - max_out_len

            # Retain the first 0.5 * max_prompt_len tokens and the last 0.5 * max_prompt_len tokens, discarding the middle ones,
            # because the prompts' questions are usually at the beginning or the end.
            # To avoid the warning:
            # This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length.
            # Depending on the model, you may observe exceptions, performance degradation, or nothing at all.
            half_max_prompt_len = max_prompt_len // 2
            if half_max_prompt_len > 0 and tokens['input_ids'].shape[1] > max_prompt_len:
                for key in tokens.keys():
                    if tokens[key].shape[1] > max_prompt_len:
                        field_values = tokens[key]
                        tokens[key] = torch.cat(
                            (field_values[:, :half_max_prompt_len], field_values[:, -half_max_prompt_len:]), dim=1
                        )

        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update(kwargs)
        # generation_kwargs.update(diffusion_kwargs)
        stopping_criteria = list(set(stopping_criteria + self.stop_words))
        if stopping_criteria:
            generation_kwargs['stopping_criteria'] = _get_stopping_criteria(stopping_criteria, self.tokenizer, batch_size)
        if max_out_len is not None:
            generation_kwargs['max_new_tokens'] = max_out_len
        if min_out_len is not None:
            generation_kwargs['min_new_tokens'] = min_out_len
        generation_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        stopping_criteria_idx = [self.tokenizer.encode(stop, add_special_tokens=False)[0] for stop in stopping_criteria]
        outputs = block_diffusion_generate(
            self.model,
            self.tokenizer,
            tokens,
            mask_id=generation_kwargs['mask_id'],
            denoising_steps=generation_kwargs['denoising_steps'],
            gen_length=generation_kwargs['gen_length'],
            block_length=generation_kwargs['block_length'],
            temperature=generation_kwargs['temperature'],
            top_k=generation_kwargs['top_k'],
            top_p=generation_kwargs['top_p'],
            # cfg_scale=generation_kwargs['cfg_scale'],
            remasking=generation_kwargs['remasking'],
            threshold=generation_kwargs['threshold'],
            stopping_criteria_idx=stopping_criteria_idx,
            )[:, tokens["input_ids"].shape[1]:]

        decodeds = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        decodeds = [t.replace('<|MASK|>', '') for t in decodeds]
        for stop in stopping_criteria:
            decodeds = [t.split(stop)[0] for t in decodeds]
        return decodeds

    def get_token_len(self, prompt: str) -> int:
        m = _convert_chat_messages([prompt])[0]
        t = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_dict=True)
        return len(t['input_ids'])

    def generate_from_template(self, templates: List[PromptType],
                               max_out_len: int, **kwargs):
        """Generate completion from a list of templates.

        Args:
            templates (List[PromptType]): A list of templates.
            max_out_len (int): The maximum length of the output.
        """
        inputs = self.parse_template(templates, mode='gen')
        if hasattr(self, 'sync_rank') and self.sync_rank:
            inputs = self.sync_inputs(inputs)
        return self.generate(inputs, max_out_len=max_out_len, **kwargs)

def  _convert_base_messages(inputs):
    outputs = []
    for _input in inputs:
        if isinstance(_input, str):
            outputs.append(_input)
        else:
            messages = []
            for item in _input:
                messages.append(item['prompt'])
            outputs.append(''.join(messages))
    return outputs
