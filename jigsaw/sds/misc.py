import math
import torch
import os
import hashlib

from jigsaw.sds.df_guidance_b import DeepFloydGuidance, DeepFloydPromptProcessor
from jigsaw.sds.sdi_guidance import StableDiffusionSDIGuidance, StableDiffusionSDIPromptProcessor


def get_text_embeds(prompt, model_id, proc_class, device="cuda", use_saved=True, save_path="text_embeds", neg_prompt=""):
    prompt_hash = hashlib.md5((prompt + "-" + model_id + "-" + neg_prompt).encode()).hexdigest()
    full_path = os.path.join(save_path, f"{prompt_hash}.pt")
    if use_saved and os.path.exists(full_path):
        print(f"Loading text embeddings from {full_path}...")
        return torch.load(full_path).to(device)
    else:
        print(f"Computing text embeddings for prompt: '{prompt}' and saving to {full_path}...")
        prompt_processor = proc_class(model_id=model_id, device=device, device_map="auto")
        text_embeddings = prompt_processor.get_text_embeddings(prompt, negative_prompt=neg_prompt)
        torch.save(text_embeddings, full_path)
        prompt_processor.destroy_text_encoder()
        return text_embeddings

def get_guidance_and_text_embeds(model_type, prompts, guidance_scale=50, device="cuda", use_saved=True, save_path="text_embeds", neg_prompt=""):
    if model_type in ["M", "L", "XL"]:
        model_id = f"DeepFloyd/IF-I-{model_type}-v1.0"
        guidance = DeepFloydGuidance(model_id=model_id, guidance_scale=guidance_scale, device=device)
        prompt_proc = DeepFloydPromptProcessor
    elif model_type == "SDI":
        model_id = "runwayml/stable-diffusion-v1-5"
        guidance = StableDiffusionSDIGuidance(guidance_scale=guidance_scale, device=device)
        prompt_proc = StableDiffusionSDIPromptProcessor
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if isinstance(prompts, str):
        text_embeddings = get_text_embeds(prompts, model_id, prompt_proc, device, use_saved=use_saved, save_path=save_path, neg_prompt=neg_prompt)
        return guidance, text_embeddings
    elif isinstance(prompts, list):
        text_embeddings = []
        for prompt in prompts:
            text_embeddings.append(get_text_embeds(prompt, model_id, prompt_proc, device, use_saved=use_saved, save_path=save_path, neg_prompt=neg_prompt))
        return guidance, torch.stack(text_embeddings, dim=0)
    else:
        raise TypeError("Prompts must be either a string or a list of strings.")



def C(value, epoch: int, global_step: int, interpolation="linear") -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        if len(value) >= 6:
            select_i = 3
            for i in range(3, len(value) - 2, 2):
                if global_step >= value[i]:
                    select_i = i + 2
            if select_i != 3:
                start_value, start_step = value[select_i - 3], value[select_i - 2]
            else:
                start_step, start_value = value[:2]
            end_value, end_step = value[select_i - 1], value[select_i]
            value = [start_step, start_value, end_value, end_step]
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
        elif isinstance(end_step, float):
            current_step = epoch
        t = max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
        if interpolation == "linear":
            value = start_value + (end_value - start_value) * t
        elif interpolation == "exp":
            value = math.exp(math.log(start_value) * (1 - t) + math.log(end_value) * t)
        else:
            raise ValueError(
                f"Unknown interpolation method: {interpolation}, only support linear and exp"
            )
    return value
