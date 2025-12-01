# flake8: noqa
# yapf: disable
from typing import Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer

from opencompass.models.base import BaseModel
from opencompass.utils import get_logger

from .huggingface_above_v4_33 import (_convert_chat_messages,
                                      _format_with_fast_chat_template,
                                      _get_meta_template,
                                      _get_possible_max_seq_len)

try:
    from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
except ImportError:
    pipeline, PytorchEngineConfig, GenerationConfig = None, None, None


class LMDeploywithChatTemplate(BaseModel):

    def __init__(
        self,
        path: str,
        model_kwargs: dict = dict(),
        tokenizer_only: bool = False,
        generation_kwargs: dict = dict(),
        max_seq_len: int = None,
        meta_template: Optional[Dict] = None,
        fastchat_template: Optional[str] = None,
        stop_words: List[str] = [],
    ):
        assert pipeline, ('Please install lmdeploy with `pip install lmdeploy`. note: torch==2.1.2 is required.')

        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        if not tokenizer_only:
            self._load_model(path, model_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        self.generation_kwargs = generation_kwargs
        self.fastchat_template = fastchat_template
        self.stop_words = list(set(stop_words + self._get_potential_stop_words(path)))

    def _load_model(self, path: str, added_model_kwargs: dict = dict()):
        import ray

        if ray.is_initialized():
            self.logger.info('shutdown ray instance to avoid "Calling ray.init() again" error.')
            ray.shutdown()

        tp = added_model_kwargs.pop("tp")
        dtype = added_model_kwargs.pop("dtype", "auto")
        cache_max_entry_count = added_model_kwargs.pop("cache_max_entry_count", 0.6)
        dllm_block_length = added_model_kwargs.pop("dllm_block_length")
        dllm_denoising_steps = added_model_kwargs.pop("dllm_denoising_steps")
        dllm_unmasking_strategy = added_model_kwargs.pop("dllm_unmasking_strategy")
        dllm_confidence_threshold = added_model_kwargs.pop("dllm_confidence_threshold", 0.9)
        max_prefill_token_num = added_model_kwargs.pop("max_prefill_token_num", 4096)
        backend_config = PytorchEngineConfig(
            tp=tp,
            dtype=dtype,
            max_prefill_token_num=max_prefill_token_num,
            cache_max_entry_count=cache_max_entry_count,
            dllm_block_length=dllm_block_length,
            dllm_denoising_steps=dllm_denoising_steps,
            dllm_unmasking_strategy=dllm_unmasking_strategy,
            dllm_confidence_threshold=dllm_confidence_threshold,
        )
        self.logger.info('Backend Config of LMDeploy: ')
        self.logger.info(backend_config)
        self.pipe = pipeline(path, backend_config=backend_config)

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

    def generate(self, inputs: List[str], max_out_len: int, stopping_criteria: List[str] = [], **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        messages = _convert_chat_messages(inputs)
        if self.fastchat_template:
            messages = _format_with_fast_chat_template(messages, self.fastchat_template)
        else:
            messages = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]
            # vLLM tokenize prompts by AutoTokenizer with its default parameter "add_special_token=True"
            # OC add bos_token in the prompt, which requires tokenizing prompts using "add_speicial_token=False"
            # But vLLM doesn't have "add_speicial_token" in the pipeline API. So, we remove bos_token
            # from messages as a workaround
            if self.tokenizer.bos_token:
                bos_token = self.tokenizer.bos_token
                messages = [message.removeprefix(bos_token) if message.startswith(bos_token) else message for message in messages]
        DEFAULT_GENERATION_KWARGS = {
            'temperature': 0,
            'max_new_tokens': max_out_len,
            'stop_words': list(set(self.stop_words + stopping_criteria))
        }
        sampling_kwargs = DEFAULT_GENERATION_KWARGS.copy()
        sampling_kwargs.update(self.generation_kwargs)
        sampling_kwargs.update(kwargs)
        sampling_kwargs = GenerationConfig(**sampling_kwargs)
        self.logger.info('Sampling Params of LMDeploy: ')
        self.logger.info(sampling_kwargs)

        outputs = self.pipe(messages, gen_config=sampling_kwargs)

        prompt_list, output_strs = [], []
        for message, output in zip(messages, outputs):
            prompt = message
            generated_text = output.text
            prompt_list.append(prompt)
            output_strs.append(generated_text)

        return output_strs

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        m = _convert_chat_messages([prompt])[0]
        t = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_dict=True)
        return len(t['input_ids'])
