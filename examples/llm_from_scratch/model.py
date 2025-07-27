from beam import beam_device

def get_hf_model(tokenizer, hparams):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hparams.model_name, trust_remote_code=True,
                                                token=hparams.huggingface_token,
                                                torch_dtype=hparams.dtype)
    model.config.use_cache = False  # Disable caching for training
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(beam_device(hparams.device))
    return model