import argparse
import logging
import math
import os
import copy
import shutil
from tqdm.auto import tqdm

import accelerate
import datasets
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms

import diffusers
from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
    is_accelerate_version,
)
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.1")

logger = get_logger(__name__, log_level="INFO")


def log_validation(
    unet: UNet2DModel,
    noise_scheduler: DDPMScheduler,
    args: DictConfig,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    step: int,
    epoch: int,
):
    logger.info("Running validation... ")

    pipeline = DDPMPipeline(
        unet=unet,
        scheduler=noise_scheduler,
    )
    pipeline = pipeline.to(accelerator.device, weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # run pipeline in inference (sample random noise and denoise)
    images = pipeline(
        generator=generator,
        batch_size=args.eval_batch_size,
        num_inference_steps=args.num_inference_steps,
        output_type="np",
    ).images

    # denormalize the images and save to tensorboard
    images_processed = (images * 255).round().astype("uint8")

    if args.report_to == "tensorboard":
        if is_accelerate_version(">=", "0.17.0.dev0"):
            tracker = accelerator.get_tracker("tensorboard", unwrap=True)
        else:
            tracker = accelerator.get_tracker("tensorboard")

        tracker.add_images(
            "test_samples", images_processed.transpose(0, 3, 1, 2), epoch
        )
    elif args.report_to == "wandb":
        # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
        accelerator.get_tracker("wandb").log(
            {
                "test_samples": [wandb.Image(img) for img in images_processed],
                "epoch": epoch,
            },
            step=step,
        )

    del pipeline
    torch.cuda.empty_cache()

    return images_processed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training the unconditional diffusion model using ddpm."
    )

    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    args = OmegaConf.load(args.config)

    # Logging, seed, config and checkpointing
    args.setdefault("output_dir", None)
    args.setdefault("logging_dir", "logs")
    args.setdefault("report_to", "tensorboard")
    args.setdefault("seed", None)
    args.setdefault("output_dir", None)
    args.setdefault("resume_from_checkpoint", None)
    args.setdefault("checkpointing_steps", 1000)
    args.setdefault("checkpoints_total_limit", None)
    args.setdefault("upcast_before_saving", False)

    # Model loading and setting
    args.setdefault("model_config_name_or_path", None)
    args.setdefault("use_ema", False)
    args.setdefault("foreach_ema", False)
    args.setdefault("gradient_checkpointing", False)
    args.setdefault("prediction_type", "epsilon")
    args.setdefault("num_train_timesteps", 1000)
    args.setdefault("ddpm_beta_schedule", "linear")
    args.setdefault("model_config_name_or_path", None)
    args.setdefault("scheduler_config_name_or_path", None)

    # Data loading
    args.setdefault("cache_dir", None)
    args.setdefault("train_data_dir", None)
    args.setdefault("max_train_samples", None)
    args.setdefault("resolution", 28)

    # Training
    args.setdefault("learning_rate", 5e-5)
    args.setdefault("adam_beta1", 0.9)
    args.setdefault("adam_beta2", 0.999)
    args.setdefault("adam_weight_decay", 0.01)
    args.setdefault("train_batch_size", 128)
    args.setdefault("lr_warmup_steps", 500)
    args.setdefault("dataloader_num_workers", 0)
    args.setdefault("max_train_steps", None)
    args.setdefault("lr_scheduler", "constant")
    args.setdefault("num_train_epochs", 100)
    args.setdefault("max_grad_norm", 1.0)

    # Validation
    args.setdefault("eval_batch_size", None)
    args.setdefault("validation_epochs", 1)
    args.setdefault("num_inference_steps", 50)

    return args


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        log_with=args.report_to, project_config=accelerator_project_config
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    if args.scheduler_config_name_or_path is not None:
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.scheduler_config_name_or_path
        )
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            prediction_type=args.prediction_type,
        )
    # Initialize vae and model
    if args.model_config_name_or_path is None:
        unet = UNet2DModel(
            sample_size=args.resolution,
            in_channels=1,
            out_channels=1,
            block_out_channels=(32, 64, 64),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )
    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        unet = UNet2DModel.from_config(config)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, weight_dtype)
    unet.train()
    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(
            unet.parameters(),
            model_cls=UNet2DModel,
            model_config=unet.config,
            foreach=args.foreach_ema,
        )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DModel
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
    )

    data_files = {}
    if args.train_data_dir is not None:
        data_files["train"] = os.path.join(args.train_data_dir, "**")
    else:
        raise ValueError("No directory path for the training data.")
    dataset = load_dataset(
        "parquet",
        data_files=data_files,
        cache_dir=args.cache_dir,
    )

    # Data preprocessing transformations
    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    def preprocess_train(examples):
        examples["pixel_values"] = [
            train_transforms(image) for image in examples["image"]
        ]
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=args.seed)
                .select(range(args.max_train_samples))
            )
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {
            "pixel_values": pixel_values,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(
            len(train_dataloader) / accelerator.num_processes
        )
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding
            / accelerator.gradient_accumulation_steps
        )
        num_training_steps_for_scheduler = (
            args.num_train_epochs
            * num_update_steps_per_epoch
            * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = (
            args.max_train_steps * accelerator.num_processes
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / accelerator.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if (
            num_training_steps_for_scheduler
            != args.max_train_steps * accelerator.num_processes
        ):
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_args = {}
        for k, v in OmegaConf.to_container(args).items():
            if isinstance(v, (int, float, str, bool, torch.Tensor)):
                tracker_args[k] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    tracker_args[f"{k}.{kk}"] = str(vv)
            else:
                tracker_args[k] = str(v)

        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, config=tracker_args)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(
                    accelerator.device, weight_dtype
                )

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(pixel_values)
                bsz = pixel_values.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=pixel_values.device,
                )
                timesteps = timesteps.long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(pixel_values, noise, timesteps)

                # Predict the noise residual
                model_pred = unet(noisy_images, timesteps, return_dict=False)[0]
                if args.prediction_type == "epsilon":
                    loss = F.mse_loss(
                        model_pred.float(), noise.float()
                    )  # this could have different weights!
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {args.prediction_type}"
                    )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / accelerator.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if (
                    global_step % args.checkpointing_steps == 0
                    and accelerator.is_main_process
                ):
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")
                        ]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1])
                        )

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = (
                                len(checkpoints) - args.checkpoints_total_limit + 1
                            )
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}"
                            )

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    args.output_dir, removing_checkpoint
                                )
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if (
            args.eval_batch_size is not None
            and accelerator.is_main_process
            and epoch % args.validation_epochs == 0
        ):
            if args.use_ema:
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())

            log_validation(
                unwrap_model(unet),
                copy.deepcopy(noise_scheduler),
                args,
                accelerator,
                weight_dtype,
                global_step,
                epoch,
            )
            if args.use_ema:
                # Switch back to the original UNet parameters.
                ema_unet.restore(unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = DDPMPipeline(
            scheduler=noise_scheduler,
            unet=unet,
        )
        if args.upcast_before_saving:
            pipeline.to(torch.device("cpu"), torch.float32)

        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
