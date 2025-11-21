import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import tqdm

import helpers as H

import gc

from ts_simple.df_guidance import DeepFloydGuidance, DeepFloydPromptProcessor

DEVICE = "cuda"

if __name__ == "__main__":
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
        return LambdaLR(optimizer, lr_lambda, -1)

    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

    guidance = DeepFloydGuidance(device=DEVICE)
    prompt_processor = DeepFloydPromptProcessor(device=DEVICE)

    prompt = "a realistic picture of the a cute and adorable kitten 4k hdr photography unreal engine beautiful high-quality"
    text_embeddings = prompt_processor.get_text_embeddings(prompt)
    prompt_processor.destroy_text_encoder()
    print(prompt)

    w = h = 128
    lr = 1e-1
    n_iters = 1000
    target = nn.Parameter(torch.rand(1, h, w, 1, device=guidance.device))

    optimizer = torch.optim.AdamW([target], lr=lr, weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(n_iters * 1.5))

    pbar = tqdm.trange(n_iters)

    for it in pbar:
        optimizer.zero_grad()

        target_rgb = torch.cat([target, target, target], dim=3)
        loss = guidance(target_rgb, text_embeddings)
        loss['loss_sds'].backward()

        grad = target.grad
        if grad.norm() > 0:
            target.grad /= grad.norm()
            target.grad *= 10

        optimizer.step()

        pbar.set_postfix_str(f"Grad norm: {grad.norm():.6f}")
        if scheduler is not None:
            scheduler.step()
        
        # guidance.update_step(epoch=0, global_step=it)

    rgb = target
    img_rgb = rgb.clamp(0, 1).detach().squeeze(0).cpu().numpy()
    H.save_images("2ddf.png", [img_rgb], auto_grid=True)