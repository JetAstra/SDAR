"""Offline tokenizer regressions; no model weights, GPU, or remote code needed."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from transformers import AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parents[1] / "training/model/SDAR-8B-Chat"
ASSISTANT_START = "<|im_start|>assistant\n"


def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, use_fast=True
    )


class SDARChatTemplateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer(MODEL_DIR)

    def assert_switches_match(
        self, messages, *, expected=None, tools=None, add_generation_prompt=True
    ):
        options = {"add_generation_prompt": add_generation_prompt}
        if tools is not None:
            options["tools"] = tools
        default = self.tokenizer.apply_chat_template(
            messages, tokenize=False, **options
        )
        if expected is not None:
            self.assertEqual(default, expected)
        default_ids = self.tokenizer.encode(default, add_special_tokens=False)
        for mode in (None, False, True):
            with self.subTest(enable_thinking=mode):
                kwargs = {} if mode is None else {"enable_thinking": mode}
                rendered = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, **options, **kwargs
                )
                ids = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, **options, **kwargs
                )
                self.assertEqual(rendered, default)
                self.assertEqual(ids, default_ids)
                if add_generation_prompt:
                    # Check the new assistant turn, not think tags in history.
                    self.assertTrue(rendered.endswith(ASSISTANT_START))
        return default

    def test_single_turn_generation_prefix(self):
        self.assert_switches_match(
            [{"role": "user", "content": "What is 2 + 2?"}],
            expected=(
                "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n"
                + ASSISTANT_START
            ),
        )

    def test_multi_turn_generation_prefix(self):
        self.assert_switches_match(
            [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "Name a primary color."},
                {"role": "assistant", "content": "Blue."},
                {"role": "user", "content": "Name another."},
            ],
            expected=(
                "<|im_start|>system\nAnswer briefly.<|im_end|>\n"
                "<|im_start|>user\nName a primary color.<|im_end|>\n"
                "<|im_start|>assistant\nBlue.<|im_end|>\n"
                "<|im_start|>user\nName another.<|im_end|>\n"
                + ASSISTANT_START
            ),
        )

    def test_without_generation_prompt(self):
        self.assert_switches_match(
            [{"role": "user", "content": "Say hello."}],
            add_generation_prompt=False,
            expected="<|im_start|>user\nSay hello.<|im_end|>\n",
        )

    def test_recent_assistant_reasoning_is_preserved(self):
        expected = (
            "<|im_start|>user\nName a primary color.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\nA primary color fits.\n</think>\n\n"
            "Blue.<|im_end|>\n"
        )
        assistants = [
            {
                "role": "assistant",
                "reasoning_content": "A primary color fits.",
                "content": "Blue.",
            },
            {
                "role": "assistant",
                "content": "<think>\nA primary color fits.\n</think>\n\nBlue.",
            },
        ]
        for assistant in assistants:
            for add_prompt in (False, True):
                with self.subTest(assistant=assistant, add_prompt=add_prompt):
                    self.assert_switches_match(
                        [{"role": "user", "content": "Name a primary color."}, assistant],
                        add_generation_prompt=add_prompt,
                        expected=expected + (ASSISTANT_START if add_prompt else ""),
                    )

    def test_prior_turn_reasoning_keeps_existing_history_policy(self):
        # The existing template omits reasoning preceding the latest user turn.
        self.assert_switches_match(
            [
                {"role": "user", "content": "Name a primary color."},
                {
                    "role": "assistant",
                    "content": "<think>\nA primary color fits.\n</think>\n\nBlue.",
                },
                {"role": "user", "content": "Name another."},
            ],
            expected=(
                "<|im_start|>user\nName a primary color.<|im_end|>\n"
                "<|im_start|>assistant\nBlue.<|im_end|>\n"
                "<|im_start|>user\nName another.<|im_end|>\n"
                + ASSISTANT_START
            ),
        )

    def test_tool_calls_and_grouped_responses_are_preserved(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "read_temperature",
                "description": "Read a room temperature.",
                "parameters": {
                    "type": "object",
                    "properties": {"room": {"type": "string"}},
                    "required": ["room"],
                },
            },
        }]
        calls = [
            {"name": "read_temperature", "arguments": {"room": "room_a"}},
            {"name": "read_temperature", "arguments": {"room": "room_b"}},
        ]
        messages = [
            {"role": "user", "content": "Compare the two room temperatures."},
            {
                "role": "assistant", "content": "",
                "reasoning_content": "Read both sensor values.",
                "tool_calls": [{"type": "function", "function": call} for call in calls],
            },
            {"role": "tool", "content": "21"},
            {"role": "tool", "content": "22"},
        ]
        rendered = self.assert_switches_match(messages, tools=tools)
        self.assertIn("<think>\nRead both sensor values.\n</think>", rendered)
        tool_definition = rendered.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]
        self.assertEqual(json.loads(tool_definition), tools[0])
        assistant_turn = rendered.split(ASSISTANT_START, 1)[1].split("<|im_end|>\n", 1)[0]
        tool_calls = [
            json.loads(part.split("\n</tool_call>", 1)[0])
            for part in assistant_turn.split("<tool_call>\n")[1:]
        ]
        self.assertEqual(tool_calls, calls)
        self.assertIn(
            "<|im_start|>user\n<tool_response>\n21\n</tool_response>"
            "\n<tool_response>\n22\n</tool_response><|im_end|>\n",
            rendered,
        )
        self.assert_switches_match(
            messages, tools=tools, add_generation_prompt=False,
            expected=rendered.removesuffix(ASSISTANT_START),
        )

    def test_standalone_and_embedded_templates_match(self):
        standalone = (MODEL_DIR / "chat_template.jinja").read_text(encoding="utf-8")
        embedded = json.loads((MODEL_DIR / "tokenizer_config.json").read_text(encoding="utf-8"))[
            "chat_template"
        ]
        self.assertEqual(standalone, embedded)
        self.assertEqual(self.tokenizer.chat_template, standalone)

    def test_embedded_template_fallback_without_jinja_file(self):
        messages = [{"role": "user", "content": "Say hello."}]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            # The fast tokenizer needs no weights, model configuration, or
            # repository Python code. Leave out Jinja to exercise config fallback.
            for name in (
                "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "added_tokens.json",
            ):
                shutil.copyfile(MODEL_DIR / name, target / name)
            fallback = load_tokenizer(target)
            self.assertEqual(fallback.chat_template, self.tokenizer.chat_template)
            for mode in (None, False, True):
                with self.subTest(enable_thinking=mode):
                    kwargs = {} if mode is None else {"enable_thinking": mode}
                    self.assertEqual(
                        fallback.apply_chat_template(messages, add_generation_prompt=True, **kwargs),
                        self.tokenizer.apply_chat_template(messages, add_generation_prompt=True),
                    )


if __name__ == "__main__":
    unittest.main()
