
from .ddpm_trainer import DDPM_Trainer
from .ddpmpp_trainer import DDPMPP_Trainer
from .recflow_trainer import Reflow_Trainer




def build_trainer(cfgs, output_dir, clip_model, diffusion_model, dataloader_pakages, accelerator=None, device='cuda'):
    if cfgs.learn_obj == 'recflow':
        trainer = Reflow_Trainer(cfgs.model.diffusion.version,
                                cfgs.task,
                                output_dir, 
                                clip_model, 
                                diffusion_model,
                                dataloader_pakages,
                                cfgs.trainer.learning_rate,
                                cfgs.trainer.gt_type,
                                device,
                                cfgs.trainer.use_ema,
                                cfgs.load_checkpoint,
                                cfgs.trainer.checkpoint_name,
                                cfgs.trainer.num_timesteps,
                                cfgs.trainer.num_iterations,
                                cfgs.trainer.save_interval,
                                accelerator,
                                cfgs.log_method,
                                cfgs.trainer.start_point,
                                clip_grads=cfgs.trainer.clip_grads)
    elif cfgs.learn_obj == 'ddpm':
        trainer = DDPM_Trainer(cfgs.model.diffusion.version,
                                cfgs.task,
                                output_dir, 
                                clip_model, 
                                diffusion_model,
                                dataloader_pakages,
                                cfgs.trainer.learning_rate,
                                cfgs.trainer.gt_type,
                                device,
                                cfgs.trainer.use_ema,
                                cfgs.load_checkpoint,
                                cfgs.trainer.checkpoint_name,
                                cfgs.trainer.num_timesteps,
                                cfgs.trainer.num_iterations,
                                cfgs.trainer.save_interval,
                                accelerator,
                                cfgs.log_method,
                                cfgs.trainer.start_point,
                                clip_grads=cfgs.trainer.clip_grads)
    elif cfgs.learn_obj == 'ddpmpp':
        trainer = DDPMPP_Trainer(cfgs.model.diffusion.version,
                                 cfgs.task,
                                 output_dir, 
                                 clip_model, 
                                 diffusion_model,
                                 dataloader_pakages,
                                 cfgs.trainer.learning_rate,
                                 cfgs.trainer.gt_type,
                                 device,
                                 cfgs.trainer.use_ema,
                                 cfgs.load_checkpoint,
                                 cfgs.trainer.checkpoint_name,
                                 cfgs.trainer.num_timesteps,
                                 cfgs.trainer.num_iterations,
                                 cfgs.trainer.save_interval,
                                 accelerator,
                                 cfgs.log_method,
                                 cfgs.trainer.start_point,
                                 clip_grads=cfgs.trainer.clip_grads,
                                 infer_algo=cfgs.infer_algo
                                )
    else:
        raise ValueError(f"Unsupported learning objective: {cfgs.learn_obj}, what do you wanna do ???")
    return trainer
    