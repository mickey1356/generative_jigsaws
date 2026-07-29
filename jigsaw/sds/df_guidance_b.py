import os
import gc
import numpy as np

import torch
import torch.nn.functional as F
from diffusers import IFPipeline
from transformers import T5EncoderModel, BitsAndBytesConfig

from jigsaw.sds.ts_typing import *

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, -1)

# DEEP_FLOYD_MODEL = "DeepFloyd/IF-I-XL-v1.0"
DEEP_FLOYD_MODEL = "DeepFloyd/IF-I-L-v1.0"
# DEEP_FLOYD_MODEL = "DeepFloyd/IF-I-M-v1.0"

class DeepFloydGuidance:
    def __init__(self, model_id=DEEP_FLOYD_MODEL, min_pct = 0.02, max_pct = 0.98, guidance_scale=7, grad_clip_val=None, device="cuda"):
        self.device = device

        self.weights_dtype = torch.float16

        self.pipe = IFPipeline.from_pretrained(
                    model_id,
                    force_download=False,
                    text_encoder=None,
                    safety_checker=None,
                    watermarker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                    variant="fp16",
                    torch_dtype=self.weights_dtype,
                )
        self.pipe.enable_model_cpu_offload(device=device)
        
        self.pipe.unet.to(memory_format=torch.channels_last)

        self.unet = self.pipe.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = self.pipe.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.min_step_percent = min_pct
        self.max_step_percent = max_pct
        self.set_min_max_steps(self.min_step_percent, self.max_step_percent)

        self.guidance_scale = guidance_scale

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)

        self.grad_clip_val = grad_clip_val

    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)


    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        text_embeddings: Float[Tensor, "2 D E"] | Float[Tensor, "B 2 D E"],
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        rgb_BCHW = rgb_BCHW * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        latents = F.interpolate(
            rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
        )

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

            # repeat text embeddings for batch if necessary
            if text_embeddings.ndim == 3 and text_embeddings.shape[0] == 2 and batch_size > 1:
                current_text_embeddings = text_embeddings.repeat_interleave(batch_size, dim=0)
            elif text_embeddings.ndim == 4 and text_embeddings.shape[0] == batch_size:
                # text embeddings are in the form (B, 2, D, E) and we need to reshape to (2B, D, E)
                # but we need to put embed[0, ...] first and then embed[1, ...]
                current_text_embeddings = torch.cat([text_embeddings[:, 0, ...], text_embeddings[:, 1, ...]], dim=0)
            else:
                current_text_embeddings = text_embeddings

            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=current_text_embeddings,
            )  # (2B, 6, 64, 64)

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
        noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        noise_pred = noise_pred_text + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        """
        # thresholding, experimental
        if self.cfg.thresholding:
            assert batch_size == 1
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
            noise_pred = custom_ddpm_step(self.scheduler,
                noise_pred, int(t.item()), latents_noisy, **self.pipe.prepare_extra_step_kwargs(None, 0.0)
            )
        """

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        return guidance_out

    # def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
    #     # clip grad for stable training as demonstrated in
    #     # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
    #     # http://arxiv.org/abs/2303.15413
    #     # if self.cfg.grad_clip is not None:
    #     #     self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

    #     self.set_min_max_steps(
    #         min_step_percent=C(self.min_step_percent, epoch, global_step),
    #         max_step_percent=C(self.max_step_percent, epoch, global_step),
    #     )

    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)


class DeepFloydPromptProcessor:
    ### these functions are unused, kept for debugging ###
    def __init__(self, model_id=DEEP_FLOYD_MODEL, device="cuda", device_map="auto"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            device_map=device_map
            # quantization_config=cfg,
        )
        # self.text_encoder.enable
        # print('text encoder loaded')
        self.pipe = IFPipeline.from_pretrained(
            model_id,
            text_encoder=self.text_encoder,  # pass the previously instantiated 8bit text encoder
            unet=None,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        self.pipe.enable_sequential_cpu_offload(device=device)
        # self.pipe.enable_model_cpu_offload(device=device)
        self.device = device

    def destroy_text_encoder(self) -> None:
        del self.text_encoder
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()


    def get_text_embeddings(self, prompt, negative_prompt=""):
        te, ut = self.pipe.encode_prompt(prompt=prompt, negative_prompt=negative_prompt, device=self.device)
        text_embeddings = te.expand(1, -1, -1)  # type: ignore
        uncond_text_embeddings = ut.expand(1, -1, -1)
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)

