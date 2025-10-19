# Fine-tuning SDAR Models with LlamaFactory

This guide provides step-by-step instructions for fine-tuning SDAR models (e.g., SDAR-4B-Chat, SDAR-8B-Chat) using the LlamaFactory framework. The process involves a specific environment setup, model preparation with custom code, and a specialized data configuration to leverage Flex Attention.

## 1. Environment Setup

First, create the Conda environment using the provided requirements file. This will install all necessary dependencies for LlamaFactory and the SDAR model.

```bash
conda env create -f llamafactory_full_env.yml
```

## 2. Model Preparation

The SDAR models in this repository require custom `modeling` and `config` files to function correctly.

1.  **Prepare Model Directories**:
    Create local directories for the models you intend to train, such as:
    *   `./model/SDAR-4B-Chat`
    *   `./model/SDAR-8B-Chat`

2.  **Add Custom Files**:
    Place the modified `modeling_*.py` and `config.json` files from this project into the corresponding model directory. These files are essential for enabling the model's unique architecture.

3.  **Download Model Weights**:
    Download the official `safetensors` weight files from their Hugging Face repositories and copy them into the same local directories.

After this step, your directory structure should look something like this:

```
./model/SDAR-4B-Chat/
├── config.json          # Custom config file
├── modeling_sdar.py     # Custom modeling file
├── model-*.safetensors  # Official weights from Hugging Face
└── ...                  # Other model files
```

## 3. Training Configuration

Next, define a YAML file to specify the training parameters. Below is an example based on `./examples/train_full_sdar/sdar_4b/sdar_4b_math_cot_full.yaml`.

### Model Configuration

In the `model` section, you must point to your local model path and enable `trust_remote_code`. This is mandatory for loading the custom modeling scripts you prepared in Step 2.

```yaml
### model
model_name_or_path: /path/to/your/model/SDAR-4B-Chat
train_from_scratch: false
trust_remote_code: true
```

### Dataset Configuration

SDAR models use **Flex Attention** as their attention backend, which performs most efficiently with fixed-shape inputs. Therefore, training must be done using a data packing method (`neat_packing: true`).

```yaml
### dataset
dataset: open_r1_math
template: qwen3
block_length: 4         # Corresponds to the model's block size for packing
cutoff_len: 20480
truncate_mode: drop     # Recommended: 'drop' or 'cut'
overwrite_cache: false
tokenized_path: /cache_dir/for/tokenized_data
preprocessing_num_workers: 96
dataloader_num_workers: 4
neat_packing: true      # Must be true for SDAR training
```

**Key Parameters for SDAR:**

*   `neat_packing: true`: This enables the data packing strategy required by the model.
*   `block_length`: Defines the block size for packing, which should align with the model's architectural design.
*   `truncate_mode: drop`: This field handles sequences that exceed `cutoff_len`.
    *   `drop`: Discards the entire sequence. This is recommended to maintain fixed input shapes for Flex Attention.
    *   `cut`: Truncates the sequence to `cutoff_len`.

## 4. Launch Training

Finally, use `torchrun` to launch the distributed training job. The command executes the LlamaFactory launcher script with your specified YAML configuration file.

The following command starts a training job on a single machine (`nnodes 1`) with 8 GPUs (`nproc_per_node 8`).

```bash
torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node 8 \
    --master_addr 127.0.0.1 \
    --master_port 12345 \
    ./src/llamafactory/launcher.py \
    ./examples/train_full_sdar/sdar_4b/sdar_4b_sb_sal_v_full.yaml
```

Make sure to replace the final argument with the path to your own YAML configuration file.
