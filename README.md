<p align="center">
  <img src="assets/SDAR_doc_head.png" style="max-width:75%; height:auto;">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue" />
  <a href="https://jetastra.github.io/SDAR/">
    <img src="https://img.shields.io/badge/Website-SDAR-blue" />
  </a>
  <a href="https://huggingface.co/collections/JetLM/sdar-689b1b6d392a4eeb2664f8ff">
    <img src="https://img.shields.io/badge/HuggingFace-Models-yellow" />
  </a>
  <img src="https://img.shields.io/badge/Technical%20Report-Coming%20Soon-orange" />
</p>


**SDAR**(**S**ynergy of **D**iffusion and **A**uto**R**egression)-model is a new large language model that integrates autoregressive (AR) and discrete diffusion modeling strategies. It combines the efficient training paradigm of AR models with the highly parallel inference capability of diffusion models, while delivering performance fully on par with SOTA opensource AR models. At the same time, SDAR sets a new benchmark as the most powerful diffusion language model to date.

## 🗞️ News
 [2025-07-20] — We’ve open-sourced the weights for our [1.7B](https://huggingface.co/JetLM/SDAR-1.7B-Chat), [4B](https://huggingface.co/JetLM/SDAR-4B-Chat), [8B](https://huggingface.co/JetLM/SDAR-8B-Chat) dense models, along with our [30B](https://huggingface.co/JetLM/SDAR-30B-A3B-Chat) MoE model — now available on Hugging Face 🤗.


## ⚙️ Usage

### Environment Setup

```
transformers>=4.52.4
```

### Training

The training code will be released soon.

### Inference

#### 1. Using the built-in inference script

```bash
python generate.py \
  --model_dir=JetLM/SDAR-1.7B-Chat \
  --trust_remote_code
```

#### 2. Using the third-party inference engine [JetEngine](https://github.com/Labman42/JetEngine)

JetEngine enables more efficient inference compared to the built-in implementation.

```bash
git clone https://github.com/JetAstra/SDAR.git
cd SDAR
git submodule update --init --recursive
cd third_party/JetEngine
pip install .
```

The following example shows how to quickly load a model with JetEngine and run a prompt end-to-end.

```python
import os
from jetengine import LLM, SamplingParams
from transformers import AutoTokenizer

model_path = os.path.expanduser("/path/to/your/model")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# Initialize the LLM
llm = LLM(
    model_path,
    enforce_eager=True,
    tensor_parallel_size=1,
    mask_token_id=151669,   # Optional: only needed for masked/diffusion models
    block_length=4
)

# Set sampling/generation parameters
sampling_params = SamplingParams(
    temperature=1.0,
    topk=0,
    topp=1.0,
    max_tokens=256,
    remasking_strategy="low_confidence_dynamic",
    block_length=4,
    denoising_steps=4,
    dynamic_threshold=0.9
)

# Prepare a simple chat-style prompt
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain what reinforcement learning is in simple terms."}],
    tokenize=False,
    add_generation_prompt=True
)

# Generate text
outputs = llm.generate_streaming([prompt], sampling_params)
```


## 📊 Benchmarks
### Scaling the Qwen3 Series with SDAR for General (Non-Reasoning) Tasks
#### Settings

We use Qwen3-1.7B-Base, Qwen3-4B-Base, Qwen3-8B-Base, and Qwen3-30B-A3B-Base as base models. Each model undergoes continued pretraining on 0.14% (50B) tokens of relatively low quality data (opensource data), followed by fine-tuning on the general SFT dataset.

- SDAR-1.7B-Chat, SDAR-4B-Chat, SDAR-8B-Chat, and SDAR-30B-A3B-Chat are trained using the **SDAR training scheme**.
- Qwen3-1.7B-AR-SFT and Qwen3-30B-AR-SFT are trained using the **autoregressive (AR) training scheme**.

#### Performance

For **SDAR** models, inference hyperparameters are set to: `block_length = 4`, `denoising_steps = 4`, greedy decoding.

For **Qwen3-1.7B-AR-SFT** and **Qwen3-30B-AR-SFT**, we use *greedy decoding*, and the base models **Qwen3-1.7B-Base** and **Qwen3-30B-Base** are derived from the [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).

<p align="center">

| Scale | Model | MMLU | GSM8K | Math500 | MathBench | HumanEval | MBPP | IFEval |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1.7B | SDAR-1.7B-Chat | 62.9 | 80.1 | 63.2 | 63.6 | 61.6 | 61.1 | 43.4 |
|  | Qwen3-1.7B-AR-SFT | 63.8 | 81.1 | 62.0 | 60.5 | 65.9 | 61.9 | 43.3 |
|  | Qwen3-1.7B-Base | 62.6 | 75.4 | 43.5 | - | - | 55.4 | - |
| 4B | SDAR-4B-Chat | 74.9 | 89.9 | 72.8 | 74.7 | 72.0 | 65.4 | 56.6 |
| 8B | SDAR-8B-Chat | 78.6 | 91.3 | 78.6 | 76.9 | 78.7 | 72.0 | 61.4 |
| 30B | SDAR-30B-A3B-Chat | 82.8 | 91.4 | 77.8 | 79.3 | 87.2 | 71.6 | 60.6 |
|  | Qwen3-30B-AR-SFT | 82.2 | 92.7 | 76.8 | 78.4 | 84.8 | 75.1 | 57.7 |
|  | Qwen3-30B-Base | 81.4 | 91.8 | 59.0 | - | - | 74.4 | - |

</p>

> **Key observations:**
> - SDAR-1.7B-Chat achieves comparable performance to Qwen3-1.7B-AR-SFT across most benchmarks.
> - SDAR-30B-A3B-Chat performs on par with Qwen3-30B-AR-SFT on the evaluated benchmarks.

#### Efficiency

We compare the performance of **SDAR-30B-A3B-Chat** and **Qwen3-30B-AR-SFT** under both *dynamic* and *static* inference settings.
Additionally, we evaluate how varying the threshold in dynamic inference affects speed relative to static inference.

<p align="center">
  <img src="assets/Performace_and_speed.svg" width="100%">
</p>

> **Key observations:**
> - SDAR achieves **over 2× faster inference speed** compared to static inference almost **without any loss in accuracy**, with its static inference speed being comparable to that of AR models.
> - The speedup effect tends to become more pronounced with increasing model size.

### Applying SDAR to Qwen3-30B-MoE for Reasoning Benchmarks
#### Settings

Starting from **Qwen3-30B-A3B-Base**, we trained on 500B tokens (including scientific data) using the NTP strategy, followed by 500B tokens of annealing, resulting in the **AR-30B-A3B-Sci-Base** model. Based on AR-30B-A3B-Sci-Base, we then performed continued training with the SDAR strategy using 50B tokens randomly sampled from the 500B annealing dataset, producing the **SDAR-30B-A3B-Sci-Base** model. Finally, both AR-30B-A3B-Sci-Base and SDAR-30B-A3B-Sci-Base were further fine-tuned on reasoning datasets to obtain the **AR-30B-A3B-Sci** and **SDAR-30B-A3B-Sci** models, respectively.


#### Performance

For the **AR-30B-A3B-Sci** model, we use decoding parameters `temperature=0.6`, `top_p=0.95`, and `top_k=20`.
For the **SDAR-30B-A3B-Sci** model, we set `block_length=4` and `denoising_steps=4`, and perform decoding with both greedy and sampling strategies, where the sampling parameters are `temperature=1.0`, `top_p=1.0`, and `top_k=0`. 



##### 1. Strict Experimental Comparison

This table presents a **controlled comparison** between AR and SDAR under the same backbone and dataset settings.
The results are averaged over 8 runs for GPQA, and over 32 runs each for AIME 2024, AIME 2025, and LiveMathBench.

<p align="center">

| Task               | AR-30B-A3B-Sci | SDAR-30B-A3B-Sci(greedy) | SDAR-30B-A3B-Sci(sample) |
| ------------------ | :------------: | :----------------------: | :----------------------: |
| MMLU-pro           |      78.3      |           80.2           |           80.6           |
| GPQA-diamond       |      61.2      |           73.7           |           71.8           |
| AIME2024           |      74.9      |           73.3           |           76.2           |
| AIME2025           |      60.7      |           63.3           |           62.2           |
| LiveMathBench-hard |      55.4      |           60.7           |           57.9           |
| LiveCodeBench-v5   |      51.5      |           40.7           |           49.1           |
| LiveCodeBench-v6   |      46.3      |           42.3           |           51.4           |
| ChemBench          |      60.5      |           75.1           |           75.1           |
| PHYSICS            |      39.0      |           52.9           |           55.6           |
| ProteinLMBench     |      59.5      |           60.7           |           60.0           |
</p>

> **Key observations:**
> * **Strict experimental comparison (AR vs. SDAR):** Under identical settings, **SDAR-30B-A3B-Sci** consistently outperforms **AR-30B-A3B-Sci**, with especially notable gains on science-focused tasks such as **GPQA**, **ChemBench**, and **PHYSICS**.

##### 2. Comparison with Other Open/Closed Models

This table positions **SDAR-30B-A3B-Sci(sample)** against leading open-source and closed-source LLMs.
Scores for external models are sourced from the [InternLM/Intern-S1](https://github.com/InternLM/Intern-S1) repository.
<p align="center">

| Model                    | MMLU-pro | AIME2025 | GPQA-diamond | ChemBench | PHYSICS | ProteinLMBench |
| ------------------------ | :------: | :------: | :----------: | :-------: | :-----: | :------------: |
| SDAR-30B-A3B-Sci(sample) |   80.6   |   62.2   |     71.8     |    75.1   |   55.6  |      60.0      |
| Intern-S1(235B-A22B)     |   83.5   |   86.0   |     77.3     |    83.4   |   44.0  |      63.1      |
| InternVL3-78B            |   73.0   |   10.7   |     49.9     |    61.3   |   23.1  |      61.6      |
| Qwen2.5-VL-72B           |   72.1   |   10.9   |     49.0     |    61.6   |   15.7  |      61.0      |
| DS-R1-0528               |   83.4   |   87.5   |     80.6     |    75.6   |    –    |      61.4      |
| Qwen3-235B-A22B          |   82.2   |   81.5   |     71.1     |    75.8   |    –    |      59.8      |
| Kimi-K2-Instruct         |   82.7   |   51.4   |     77.8     |    75.3   |    –    |      66.7      |
| Gemini-2.5 Pro           |   86.0   |   83.0   |     83.8     |    82.8   |   40.0  |      62.9      |
| o3                       |   85.0   |   88.9   |     83.3     |    81.6   |   47.9  |      67.7      |
| Grok-4                   |   85.9   |   91.7   |     87.5     |    83.3   |   42.8  |      66.2      |
</p>

> **Key observations:**
>   * On **GPQA**, SDAR surpasses strong open-source baselines like **Qwen3-235B-A22B** and approaches the performance of top closed-source models including **Gemini-2.5 Pro**, **o3**, and **Grok-4**.
>   * On **ChemBench**, SDAR matches or outperforms most open-source systems and narrows the gap to state-of-the-art proprietary models, demonstrating strong scientific reasoning ability.
>   * On **PHYSICS**, **SDAR achieves the best score across all listed models (55.6)**, surpassing much larger open- and closed-source models such as **Intern-S1(235B-A22B)**, **Gemini-2.5 Pro**, **o3**, and **Grok-4** — establishing a new **SOTA** on this benchmark.

## 🔥 Highlight

1. **Innovation** — Alleviates the inefficiency of autoregressive (AR) models during inference while preserving their training efficiency. Additionally, it leverages diffusion mechanisms to reduce causal inductive bias in relevant scenarios.

2. **Efficient Computation** — Integrates the training efficiency of AR models with the fast inference capabilities of diffusion models, achieving up to **2× faster inference**.

3. **Strong Performance** — Demonstrates state-of-the-art results across diverse benchmarks:
   - The **non-reasoning variant** matches or outperforms the strongest existing AR models of the same scale.
   - The **reasoning variant** performs competitively with top-tier **Qwen3** models.
   - Unlike most current diffusion-based LMs—often limited to small-scale experiments or niche tasks (e.g., code generation)—**our models are production-ready and widely applicable**.


## 🚩 Roadmap

- [x] release non-reasoning models
- [x] release naive inference code
- [x] release inference engine
- [ ] release training code
- [ ] release reasoning models

## 🔬 Citation

```
@misc{JetAstra2025,
  title={SDAR: A Synergistic Diffusion–AutoRegression Paradigm for Scalable Sequence Generation},
  author={Shuang Cheng and Yihan Bian and Dawei Liu and Biqing Qi and Qipeng Guo and Kai Chen and Bowen Zhou},
  year={2025},
  institution={Shanghai AI Lab},
  url={https://github.com/JetAstra/SDAR}
}
```
## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=JetAstra/SDAR&type=Date)](https://www.star-history.com/#JetAstra/SDAR&Date)
