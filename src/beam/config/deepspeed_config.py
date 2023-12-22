deepspeed_config = {
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 1,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 0.001,
            "warmup_num_steps": 1000
        }
    },

    "fp16": {
        "enabled": False,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "bf16": {
        "enabled": False
    },

    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "cpu_offload": True
    },

    "gradient_clipping": 1.0,
    "steps_per_print": 1000,
    "wall_clock_breakdown": False,
    "memory_breakdown": False,
    "dump_state": False,

    "flops_profiler": {
        "enabled": False,
        "profile_step": 200,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": False,
        "output_file": None
    },

    "tensorboard": {
        "enabled": False,
        "output_path": "./runs",
        "job_name": "deepspeed_job"
    },

    "activation_checkpointing": {
        "partition_activations": False,
        "cpu_checkpointing": False,
        "synchronize_checkpoint_boundary": False,
        "contiguous_memory_optimization": False
    },

    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True
    },

    "offload_param": {
        "device": "cpu",
        "pin_memory": True
    },

    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "thread_count": 1,
        "single_submit": False,
        "overlap_events": True
    },

    "elasticity": {
        "enabled": False,
        "min_gpus": 1,
        "max_gpus": 4
    }
}