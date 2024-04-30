# AMEFT
AMEFT: Aggressive Memory Efficient Fine Tuning

## How to run experiments with Profiling and Memory Consumption Analysis

1. Go to LLaMA-Factory and install the requirements
```bash
cd LLaMA-Factory
conda create -n llama_factory python=3.10
conda activate llama_factory
pip install -e .[metrics,deepspeed]
```

2. Install torch_tb_profiler
```bash
pip install torch_tb_profiler
```

3. Configure `PROFILER_LOG_DIR`, `MODEL_NAME_OR_PATH`, and `OUTPUT_DIR` in `train_sft_lora_ds.sh`

    note: `PROFILER_LOG_DIR` if not set, the profiler will be disabled.

4. Configure `--num_gpus` and `--deepspeed` in the script.

4. Run the script
```bash
bash train_sft_lora_ds.sh
```
5. After finishing the training, you can visualize the profiling results using tensorboard
```bash
tensorboard --logdir=PROFILER_LOG_DIR
```

Note if errors with threads being unavailable occur, you may try to run the following command

```bash
export OPENBLAS_NUM_THREADS='1'
tensorboard --logdir=PROFILER_LOG_DIR
```