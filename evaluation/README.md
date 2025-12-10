# **SDAR Model Evaluation Guide (OpenCompass)**

This guide explains how to evaluate SDAR models (such as  **SDAR-1.7B-Chat**, **SDAR-4B-Chat** or **SDAR-8B-Chat**) using the [OpenCompass](https://github.com/open-compass/opencompass) framework.

---

## 1. Environment Setup

1. Make sure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed.
2. In your project directory, create the Conda environment and install dependencies:

```bash
conda env create -f environment.yml
```

3. Activate the newly created environment:

```bash
conda activate <your_env_name>
```

---

## 2. Evaluation Script

A sample evaluation script is provided at:

```
evaluation/opencompass/configs/eval_sdar_lmdeploy.py
```

You can customize this file to change:

- Model path or name
- Evaluation datasets
- Output directory and logging options

---

## 3. Run Evaluation

From the project root, run:

Using the lmdeploy inference engine:
```bash
python run.py configs/eval_sdar_lmdeploy.py
```

Using the Hugging Face inference script:
```bash
python run.py configs/eval_sdar_hf.py
```

---

## Notes & Recommendations

- To evaluate multiple models, add their configurations to `eval_sdar_lmdeploy.py`.
- OpenCompass supports various datasets; adjust the dataset list as needed based on your evaluation goals.