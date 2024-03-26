# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

This code has been changed in the following ways:
1) for all samples in a batch same noise can be used
2) tqdm bar can be turned on and off 

"""


import inspect
from typing import List, Optional, Tuple, Union

import torch

from diffusers.models import UNet2DModel, VQModel
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from tqdm import tqdm


class ConditionalPipeline(DiffusionPipeline):

    def __init__(self, unet: UNet2DModel, scheduler: DDIMScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        input_cond,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        same_noise_mode = False,
        disable_tqdm=True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        if same_noise_mode: # model performs better when same noise is used in every frame of the video (all video frames is a batch)
            latents = randn_tensor(
                (1, 3, self.unet.config.sample_size, self.unet.config.sample_size),
                generator=generator,device=self.device
            ).repeat(batch_size,1,1,1)
        else:
            latents = randn_tensor(
                (batch_size, 3, self.unet.config.sample_size, self.unet.config.sample_size),
                generator=generator,device=self.device
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in tqdm(self.scheduler.timesteps,disable=disable_tqdm):
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            # predict the noise residual
            input = torch.cat((latent_model_input,input_cond),axis=1)
            
            noise_prediction = self.unet(input, t).sample #original LDMPipeline does not use class labels
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_prediction, t, latents, **extra_kwargs).prev_sample


        image = (latents / 2 + 0.5).clamp(0, 1)
        if output_type != "tensor":
            image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
