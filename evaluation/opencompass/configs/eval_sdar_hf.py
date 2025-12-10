import torch
from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import BD3withChatTemplate


with read_base():
    # datasets setting
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    # math
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets
    from opencompass.configs.datasets.math.math_prm800k_500_0shot_cot_gen_11c4b5 import math_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416 import sanitized_mbpp_datasets

    from opencompass.configs.datasets.MathBench.mathbench_2024_gen_50a320 import (
        mathbench_datasets,
    )
    # Instruction Following
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import (
        ifeval_datasets,
    )
    # summarizer
    from opencompass.configs.summarizers.internlm2_keyset import summarizer
    from opencompass.configs.summarizers.groups.mathbench_v1_2024 import (
        mathbench_2024_summary_groups,
    )
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups

# summarizer
summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)

summary_groups.append(
    {
        'name': 'Mathbench',
        'subsets': ['mathbench-a (average)', 'mathbench-t (average)'],
    },
)

# Summarizer
summarizer = dict(
    dataset_abbrs=[
        'Instruction Following',
        ['IFEval', 'Prompt-level-strict-accuracy'],
        '',
        "Math Calculation",
        ["gsm8k", "accuracy"],
        ['Mathbench', 'naive_average'],
        ['math_prm800k_500', 'accuracy'],
        '',
        'Knowledge',
        ['mmlu', 'naive_average'],
        '',
        'Code',
        ['openai_humaneval', 'humaneval_pass@1'],
        ['sanitized_mbpp', 'score'],
        '',
        'mmlu',
        'mmlu-stem',
        'mmlu-social-science',
        'mmlu-humanities',
        'mmlu-other',
        '',
        '###### MathBench-A: Application Part ######',
        'college',
        'high',
        'middle',
        'primary',
        'arithmetic',
        'mathbench-a (average)',
        '###### MathBench-T: Theory Part ######',
        'college_knowledge',
        'high_knowledge',
        'middle_knowledge',
        'primary_knowledge',
        'mathbench-t (average)',
    ],
    summary_groups=summary_groups,
)

# datasets = [*mmlu_datasets, *gsm8k_datasets, *humaneval_datasets, *sanitized_mbpp_datasets, *math_datasets, *mathbench_datasets, *ifeval_datasets]
datasets = [*gsm8k_datasets]
for dataset in datasets:
    dataset['infer_cfg']['inferencer']['batch_size'] = 1 # only support batchsize=1 up to now

# model
model_configs = [
    # ("SDAR-1.7B-Chat-b4-thr0_95", "/xxx/Models/SDAR/SDAR-1.7B-Chat", 4, 0.95, 1),
    ("SDAR-1.7B-Chat-b4-thr1_00", "xxx/Models/SDAR/SDAR-1.7B-Chat", 4, 1.0, 1),
    # ("SDAR-4B-Chat-b4-thr0_95", "/xxx/Models/SDAR/SDAR-4B-Chat", 4, 0.95, 1),
    # ("SDAR-4B-Chat-b4-thr1_00", "/xxx/Models/SDAR/SDAR-4B-Chat", 4, 1.0, 1),
    # ("SDAR-8B-Chat-b4-thr0_95", "/xxx/Models/SDAR/SDAR-8B-Chat", 4, 0.95, 1),
    # ("SDAR-8B-Chat-b4-thr1_00", "/xxx/Models/SDAR/SDAR-8B-Chat", 4, 1.0, 1),
    # ("SDAR-30B-A3B-Chat-b4-thr0_95", "/xxx/Models/SDAR/SDAR-30B-A3B-Chat", 4, 0.95, 1),
    # ("SDAR-30B-A3B-Chat-b4-thr1_00", "/xxx/Models/SDAR/SDAR-30B-A3B-Chat", 4, 1.0, 1)
]
models = []
for abbr, path, block_length, threshold, num_gpus in model_configs:

    models.append(
        dict(
        type=BD3withChatTemplate,
        abbr=abbr,
        path=path,
        run_cfg=dict(num_gpus=num_gpus),
        generation_kwargs=dict(
            mask_id=151669,
            gen_length=4096,
            block_length=block_length,
            denoising_steps=block_length,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            cfg_scale=0.0,
            remasking='low_confidence',
            threshold=threshold
        ),
        model_kwargs=dict(
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ),
    )
    )

GPUS = 8
infer = dict(
    # 同时启动num_workers个任务并行
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=GPUS,  # 划分完成后的任务数 / 预期能有的 worker 数
        # force_rebuild=True
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=GPUS,  # 最大并行运行进程数
        keep_tmp_file=True,
        task=dict(type=OpenICLInferTask),
        retry=5
    )
)
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=16),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLEvalTask, dump_details=True)),
)

work_dir = f'./outputs/eval-chat-sdar'
