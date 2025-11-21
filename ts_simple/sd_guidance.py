import os
import gc

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModel

from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

from ts_simple.ts_typing import *

STABLE_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"

class StableDiffusionGuidance:

    def __init__(self, min_pct = 0.02, max_pct = 0.98, guidance_scale=100, grad_clip_val=None, device="cuda"):
        self.device = device

        self.weights_dtype = torch.float32

        self.pipe = StableDiffusionPipeline.from_pretrained(
            STABLE_DIFFUSION_MODEL,
            tokenizer=None,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            torch_dtype=self.weights_dtype,
        ).to(self.device)

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
            STABLE_DIFFUSION_MODEL,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * min_pct)
        self.max_step = int(self.num_train_timesteps * max_pct)
        self.guidance_scale = guidance_scale

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)
        self.grad_clip_val: Optional[float] = grad_clip_val


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


    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        image: Float[Tensor, "B 3 512 512"],
        t: Int[Tensor, "B"],
        text_embeddings,
    ):
        batch_size = latents.shape[0]

        neg_guidance_weights = None
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # w(t), sigma_t^2
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)


        alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
        sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
        latents_denoised = (latents_noisy - sigma * noise_pred) / alpha
        image_denoised = self.decode_latents(latents_denoised)

        grad = w * (noise_pred - noise)
        
        # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        # if self.cfg.use_img_loss:
        #     grad_img = w * (image - image_denoised) * alpha / sigma
        # else:
        grad_img = None

        # guidance_eval_utils = {
        #     "use_perp_neg": prompt_utils.use_perp_neg,
        #     "neg_guidance_weights": neg_guidance_weights,
        #     "text_embeddings": text_embeddings,
        #     "t_orig": t,
        #     "latents_noisy": latents_noisy,
        #     "noise_pred": noise_pred,
        # }

        return grad, grad_img

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        text_embeddings,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )

        # if rgb_as_latents:
        #     latents = F.interpolate(
        #         rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
        #     )
        # else:
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

        grad, grad_img = self.compute_grad_sds(
            latents,
            rgb_BCHW_512,
            t,
            text_embeddings
        )

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

        # if self.cfg.use_img_loss:
        #     grad_img = torch.nan_to_num(grad_img)
        #     if self.grad_clip_val is not None:
        #         grad_img = grad_img.clamp(-self.grad_clip_val, self.grad_clip_val)
        #     target_img = (rgb_BCHW_512 - grad_img).detach()
        #     loss_sds_img = (
        #         0.5 * F.mse_loss(rgb_BCHW_512, target_img, reduction="sum") / batch_size
        #     )
        #     guidance_out["loss_sds_img"] = loss_sds_img

        return guidance_out


class StableDiffusionPromptProcessor:
    def __init__(self, device="cuda"):
        self.device = device

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.tokenizer = AutoTokenizer.from_pretrained(
            STABLE_DIFFUSION_MODEL, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            STABLE_DIFFUSION_MODEL, subfolder="text_encoder"
        ).to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        gc.collect()
        torch.cuda.empty_cache()

    def get_text_embeddings(self, prompt: Union[str, List[str]], negative_prompt="") -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
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
