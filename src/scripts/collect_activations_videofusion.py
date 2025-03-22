import os
import sys

from simple_parsing import parse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from accelerate import Accelerator

from diffusers import DPMSolverMultistepScheduler
from src.hooked_model.hooked_model_videofusion import HookedDiffusionModel

from src.sae.cache_activations_runner_videofusion import CacheActivationsRunner
from src.sae.config import CacheActivationsRunnerConfig


def run():
    args = parse(CacheActivationsRunnerConfig)
    accelerator = Accelerator()
    # define model
    pipe = HookedDiffusionModel.from_pretrained(
        args.model_name, torch_dtype=args.dtype, variant="fp16"
    ).to(accelerator.device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    CacheActivationsRunner(args, pipe, accelerator).run()


if __name__ == "__main__":
    run()