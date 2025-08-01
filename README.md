<div align="center" style="display: flex; align-items: center; justify-content: center;">
  <img src="./asset/logo.jpg" width="80" style="margin-right: 15px;">
  <h1><b>SDAR</b> (<b>S</b>ynergy of <b>D</b>iffusion and <b>A</b>uto<b>R</b>egression)</h1>
</div>

**SDAR**(**S**ynergy of **D**iffusion and **A**uto**R**egression)-model is a new large language model that integrates autoregressive (AR) and discrete diffusion modeling strategies. It combines the efficient training paradigm of AR models with the highly parallel inference capability of diffusion models, while delivering performance fully on par with SOTA opensource AR models. At the same time, SDAR sets a new benchmark as the most powerful diffusion language model to date.

## 🗞️ Updates

 **2025-07-20** — We’ve open-sourced the weights for our **1.7B**, **4B**, and **8B** dense models, along with our **30B** MoE model. [Explore on Hugging Face 🤗](https://huggingface.co/collections/JetLM/sdar-689b1b6d392a4eeb2664f8ff)

## ⚙️ Usage

### Environment Setup

```
transformers>=4.52.4
```

### Training

The training code will be released soon.

### Inference

```
python generate.py 
```

## 📊 Benchmarks

### Settings

We use Qwen3-1.7B-Base, Qwen3-4B-Base, Qwen3-8B-Base, and Qwen3-30B-A3B-Base as base models. Each model undergoes continued pretraining on 0.14% (50B) tokens of relatively low quality data (opensource data), followed by fine-tuning on the general SFT dataset.

- SDAR-1.7B-Chat, SDAR-4B-Chat, SDAR-8B-Chat, and SDAR-30B-A3B-Chat are trained using the **SDAR training scheme**.
- Qwen3-1.7B-AR-SFT and Qwen3-30B-AR-SFT are trained using the **autoregressive (AR) training scheme**.

### Performance

For **SDAR** models, inference hyperparameters are set to:
`block_length = 4`, `denoising_steps = 4`, greedy decoding.

For Qwen3-1.7B-AR-SFT and Qwen3-30B-AR-SFT, we use greedy decoding.

| Model             | MMLU | Math500 | GSM8K | HumanEval | Sanitized_MBPP | IFEval | MathBench |
|:-----------------:|:----:|:-------:|:-----:|:---------:|:--------------:|:------:|:---------:|
| SDAR-1.7B-Chat    | 62.9 |  63.2   | 80.06 | 61.59     | 61.09          | 43.44  | 63.55     |
| SDAR-4B-Chat      | 74.85|  72.8   | 89.92 | 71.95     | 65.37          | 56.56  | 74.71     |
| SDAR-8B-Chat      | 78.61|  78.6	 | 91.28 | 78.66     | 71.98          | 61.37  | 76.94     |
| SDAR-30B-A3B-Chat | 82.80|  77.8   | 92.27 | 83.54     | 74.32          | 58.60  | 78.33     |
| Qwen3-1.7B-AR-SFT | 63.82|  62.00  | 81.12 | 65.85     | 61.87          | 43.25  | 60.53     |
| Qwen3-30B-AR-SFT  | 82.23|  76.8   | 92.65 | 84.76     | 75.10          | 57.67  | 78.41     |
| Qwen3-1.7B-Base   | 62.6 |  43.5   | 75.44 |   -       | 55.40          |   -    |  -        |
| Qwen3-30B-Base    | 81.38|  59.04  | 91.81 |   -       | 74.40          |   -    |  -        |


> SDAR-1.7B-Chat matches Qwen3-1.7B-AR-SFT across all benchmarks.
> SDAR-30B-A3B-Chat matches Qwen3-30B-AR-SFT across all benchmarks.

### Efficiency

SDAR provides over 2× faster inference speed losslessly compared to the same-size AR models.

## 🔥 Highlight

- **Innovation** — Pioneers a new language model training and inference paradigm that **integrates the strengths of autoregressive (AR) and diffusion models**.  
  - Mitigates the inefficiency of AR models during inference while retaining their efficient training characteristics, and leverages diffusion to alleviate **causal inductive bias** in certain scenarios.

- **Efficient Computation** — Combines the efficient training paradigm of AR models with the efficient inference of diffusion models, achieving **2× faster inference speed**.

- **Strong Performance** — On multiple benchmarks:  
  - Non-reasoning model **matches or surpasses** the strongest existing non-reasoning AR models of the same scale.  
  - Reasoning model **performs on par** with same-tier **Qwen3** models.  
  - Unlike most current diffusion-based LMs (often limited to small-scale experiments or niche tasks, e.g., code generation), our models are fully production-ready.

## 🚩 Roadmap
- [x] release non-reasoning models
- [x] release naive inference code
- [ ] release training code
- [ ] release reasoning models

## 🔬 Citation

```
@misc{JetAstra2025,
  title={SDAR: Synergy of Diffusion and AutoRegression},
  author={Shuang Cheng and Dawei Liu and Yihan Bian},
  year={2025},
  institution={Shanghai AI Lab},
  url={https://github.com/JetAstra/SDAR}
}
```
## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=JetAstra/SDAR&type=Date)](https://www.star-history.com/#JetAstra/SDAR&Date)
