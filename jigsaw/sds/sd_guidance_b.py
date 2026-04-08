import os
import gc

import numpy as np

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModel

from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

from jigsaw.sds.ts_typing import *

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, -1)

STABLE_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"
# STABLE_DIFFUSION_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

class StableDiffusionGuidance:

    def __init__(self, model_id=STABLE_DIFFUSION_MODEL, min_pct = 0.02, max_pct = 0.98, guidance_scale=100, grad_clip_val=None, device="cuda"):
        self.device = device

        self.weights_dtype = torch.float16

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            tokenizer=None,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            torch_dtype=self.weights_dtype,
        ).to(self.device)
        # self.pipe.enable_model_cpu_offload(device=device)

        self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        gc.collect()
        torch.cuda.empty_cache()

        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        
        self.min_step_percent = min_pct
        self.max_step_percent = max_pct
        self.set_min_max_steps(self.min_step_percent, self.max_step_percent)

        self.guidance_scale = guidance_scale

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)
        self.grad_clip_val: Optional[float] = grad_clip_val

    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)


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


    @torch.amp.autocast("cuda", enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)


    @torch.amp.autocast("cuda", enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)


    def __call__(self, rgb: Float[Tensor, "B H W C"], text_embeddings):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )

        # encode image into latents with vae
        latents = self.encode_images(rgb_BCHW_512)

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
            if text_embeddings.shape[0] == 2 and batch_size > 1:
                current_text_embeddings = text_embeddings.repeat_interleave(batch_size, dim=0)
            else:
                current_text_embeddings = text_embeddings

            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=current_text_embeddings,
            )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

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


class StableDiffusionPromptProcessor:
    def __init__(self, model_id=STABLE_DIFFUSION_MODEL, device="cuda", **kwargs):
        self.device = device

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder"
        ).to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        gc.collect()
        torch.cuda.empty_cache()

    def get_text_embeddings(self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]] = ""):
        if isinstance(prompt, str):
            prompt = [prompt]
        
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompt)
        
        if len(negative_prompt) < len(prompt):
            negative_prompt = negative_prompt + [""] * (len(prompt) - len(negative_prompt))

        # Tokenize text and get embeddings
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
            uncond_text_embeddings = self.text_encoder(
                uncond_tokens.input_ids.to(self.device)
            )[0]

        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)
