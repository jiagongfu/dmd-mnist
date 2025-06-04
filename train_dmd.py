import argparse
import math
import os
import copy
import shutil
from tqdm.auto import tqdm

import accelerate
import datasets
import torch
import torch.nn.functional as F
import diffusers
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from diffusers import UNet2DModel
from diffusers.schedulers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available, is_accelerate_version
from diffusers.utils.torch_utils import is_compiled_module

from pipeline_dmd import DMDPipeline, get_x0_from_noise

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.1")

logger = get_logger(__name__, log_level="INFO")

if is_wandb_available():
    import wandb


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

    pipeline = DMDPipeline(
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


def compute_distribution_matching_loss(
    args: DictConfig,
    scheduler: DDPMScheduler,
    generated_x0: torch.Tensor,
    fake_score_model: UNet2DModel,
    real_score_model: UNet2DModel,
):
    bsz = generated_x0.shape[0]
    with torch.no_grad():
        timestep = torch.randint(
            args.min_timesteps,
            min(args.max_timesteps + 1, scheduler.config.num_train_timesteps),
            (bsz,),
            device=generated_x0.device,
            dtype=torch.long,
        )

        noise = torch.randn_like(generated_x0)

        noisy_image = scheduler.add_noise(generated_x0, noise, timestep)

        # run at full precision as autocast and no_grad doesn't work well together
        pred_fake_noise = fake_score_model(noisy_image, timestep, return_dict=False)[0]

        pred_fake_x0 = get_x0_from_noise(
            noisy_image.double(),
            pred_fake_noise.double(),
            scheduler.alphas_cumprod.double(),
            timestep,
        )

        pred_real_noise = real_score_model(noisy_image, timestep, return_dict=False)[0]

        pred_real_x0 = get_x0_from_noise(
            noisy_image.double(),
            pred_real_noise.double(),
            scheduler.alphas_cumprod.double(),
            timestep,
        )

        p_real = -pred_real_x0
        p_fake = -pred_fake_x0

        grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
        grad = torch.nan_to_num(grad)

    # Stop gradient
    loss = 0.5 * F.mse_loss(
        generated_x0.float(),
        (generated_x0 - grad).detach().float(),
        reduction="mean",
    )

    return loss


def compute_score_loss(
    pixel_values: torch.Tensor, scheduler: DDPMScheduler, fake_score_model: UNet2DModel
):
    bsz = pixel_values.shape[0]
    noise = torch.randn_like(pixel_values)

    timestep = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (bsz,),
        device=pixel_values.device,
        dtype=torch.long,
    )
    noisy_image = scheduler.add_noise(pixel_values, noise, timestep)
    fake_noise_pred = fake_score_model(noisy_image, timestep, return_dict=False)[0]

    loss = F.mse_loss(fake_noise_pred, noise, reduction="mean")

    return loss


def log_grad_norm(model: torch.nn.Module, accelerator: Accelerator, global_step: int):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    args = OmegaConf.load(args.config)

    # Logging, seed and config
    args.setdefault("output_dir", None)
    args.setdefault("logging_dir", "logs")
    args.setdefault("report_to", "tensorboard")
    args.setdefault("seed", None)
    args.setdefault("output_dir", None)
    args.setdefault("resume_from_checkpoint", None)
    args.setdefault("checkpoints_total_limit", None)
    args.setdefault("log_grad_norm_steps", 500)
    args.setdefault("log_steps", 50)

    # Model loading and setting
    args.setdefault("pretrained_model_name_or_path", None)
    args.setdefault("use_ema", False)
    args.setdefault("dfake_gen_update_ratio", 1)
    args.setdefault("conditioning_timestep", 999)
    args.setdefault("min_timesteps", 20)
    args.setdefault("max_timesteps", 980)

    # Data loading
    args.setdefault("cache_dir", None)
    args.setdefault("train_data_file", None)
    args.setdefault("resolution", 28)
    args.setdefault("max_train_samples", None)

    # Training
    args.setdefault("learning_rate", 1e-5)
    args.setdefault("score_learning_rate", 1e-5)
    args.setdefault("adam_beta1", 0.9)
    args.setdefault("adam_beta2", 0.999)
    args.setdefault("adam_weight_decay", 0.01)
    args.setdefault("max_train_steps", None)
    args.setdefault("lr_scheduler", "constant_with_warmup")
    args.setdefault("lr_warmup_steps", 500)
    args.setdefault("num_train_epochs", 100)
    args.setdefault("max_grad_norm", 1.0)
    args.setdefault("train_batch_size", 1)

    return args


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        log_with=args.report_to, project_config=accelerator_project_config
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################

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

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.pretrained_model_name_or_path is not None:
        # Load generator
        model = UNet2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet"
        )
        # Load real and fake score models
        real_score_model = UNet2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet"
        )
        fake_score_model = UNet2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet"
        )
    else:
        raise NotImplementedError("Training from scratch is not supported.")

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(), model_cls=UNet2DModel, model_config=model.config
        )

    model.to(accelerator.device, weight_dtype)
    fake_score_model.to(accelerator.device, weight_dtype)
    real_score_model.to(accelerator.device, weight_dtype)

    model.train()
    fake_score_model.train()
    real_score_model.requires_grad_(False)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))
                unet = models[0]
                fake_score_model = models[1]
                unet.save_pretrained(os.path.join(output_dir, "unet"))
                fake_score_model.save_pretrained(
                    os.path.join(output_dir, "fake_score_model")
                )
                weights.pop()
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DModel
                )
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model
            fake_score_model = models.pop()
            load_model = UNet2DModel.from_pretrained(
                input_dir, subfolder="fake_score_model"
            )
            fake_score_model.register_to_config(**load_model.config)
            fake_score_model.load_state_dict(load_model.state_dict())
            del load_model
            unet = models.pop()
            load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
            unet.register_to_config(**load_model.config)
            unet.load_state_dict(load_model.state_dict())
            del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
    )
    score_optimizer = torch.optim.AdamW(
        list(fake_score_model.parameters()),
        lr=args.score_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
    )

    ##################################
    # DATLOADER and LR-SCHEDULER     #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")
    # DataLoaders creation
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
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
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

    # DataLoaders creation
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=args.max_train_steps,
        num_warmup_steps=args.lr_warmup_steps,
    )
    score_lr_scheduler = get_scheduler(
        args.score_lr_scheduler,
        optimizer=score_optimizer,
        num_training_steps=args.max_train_steps,
        num_warmup_steps=args.lr_warmup_steps,
    )

    # Prepare everything with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    (
        model,
        fake_score_model,
        optimizer,
        score_optimizer,
        lr_scheduler,
        score_lr_scheduler,
    ) = accelerator.prepare(
        model,
        fake_score_model,
        optimizer,
        score_optimizer,
        lr_scheduler,
        score_lr_scheduler,
    )
    if args.use_ema:
        ema_model.to(accelerator.device)

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
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / accelerator.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

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

    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    avg_dm_loss, avg_score_loss = None, None
    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        fake_score_model.train()

        for i, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"]
            generator_step = i % args.dfake_gen_update_ratio == 0

            noise = torch.randn_like(pixel_values).to(accelerator.device, weight_dtype)
            timestep = (
                torch.ones(noise.shape[0], device=noise.device, dtype=torch.long)
                * args.conditioning_timestep
            )
            # Train Step
            # The behavior of accelerator.accumulate is to
            # 1. Check if gradients are synced(reached gradient-accumulation_steps)
            # 2. If so sync gradients by stopping the not syncing process
            if generator_step:
                optimizer.zero_grad()
            else:
                score_optimizer.zero_grad()

            noise_pred = model(noise, timestep.long(), return_dict=False)[0]
            x0_pred = get_x0_from_noise(
                noise, noise_pred, noise_scheduler.alphas_cumprod, timestep
            )

            # Genertor's turn
            if generator_step:
                with accelerator.accumulate(model):
                    loss = compute_distribution_matching_loss(
                        args,
                        noise_scheduler,
                        x0_pred,
                        fake_score_model,
                        real_score_model,
                    )
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_dm_loss = (
                        accelerator.gather(loss.repeat(args.train_batch_size))
                        .float()
                        .mean()
                    )
                    accelerator.backward(loss)

                    if args.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    optimizer.step()
                    lr_scheduler.step()

                    # log gradient norm before zeroing it
                    if (
                        accelerator.sync_gradients
                        and global_step % args.log_grad_norm_steps == 0
                        and accelerator.is_main_process
                    ):
                        log_grad_norm(model, accelerator, global_step)
            # Score model's turn
            else:
                with accelerator.accumulate(fake_score_model):
                    x0_pred.detach_()
                    loss = compute_score_loss(
                        x0_pred, noise_scheduler, fake_score_model
                    )

                    avg_score_loss = accelerator.gather(
                        loss.repeat(args.train_batch_size)
                    ).mean()
                    accelerator.backward(loss)

                    if args.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            fake_score_model.parameters(), args.max_grad_norm
                        )

                    score_optimizer.step()
                    score_lr_scheduler.step()
                    if (
                        accelerator.sync_gradients
                        and global_step % args.log_grad_norm_steps == 0
                        and accelerator.is_main_process
                    ):
                        log_grad_norm(fake_score_model, accelerator, global_step)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                if args.use_ema:
                    ema_model.step(model.parameters())

            if (
                accelerator.sync_gradients
                and not generator_step
                and accelerator.is_main_process
            ):
                # wait for both generator and score_model to settle
                # Log metrics
                if global_step % args.log_steps == 0:
                    logs = {
                        "step_score_loss": avg_score_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    if avg_dm_loss is not None:
                        logs["step_dm_loss"] = avg_dm_loss.item()

                    accelerator.log(logs, step=global_step)

                # Save model checkpoint
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

                # Generate images
                if global_step % args.validation_steps == 0:
                    if args.use_ema:
                        # Store the generator parameters temporarily and load the EMA parameters to perform inference.
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())
                    log_validation(
                        unwrap_model(model),
                        copy.deepcopy(noise_scheduler),
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                        epoch,
                    )
                    if args.use_ema:
                        # Switch back to the original VQGAN parameters.
                        ema_model.restore(model.parameters())

            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = unwrap_model(model)
        fake_score_model = unwrap_model(fake_score_model)
        if args.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained(os.path.join(args.output_dir, "unet"))
        fake_score_model.save_pretrained(
            os.path.join(args.output_dir, "fake_score_model")
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
