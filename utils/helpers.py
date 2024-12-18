import os 
import torch 
import matplotlib
import numpy as np
import importlib.util
import blobfile as bf

from torch import nn
from matplotlib import pyplot as plt
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from . import logger


INITIAL_LOG_LOSS_SCALE = 20.0


def import_or_skip(module_name):
    if importlib.util.find_spec(module_name) is None:
        print(f"Skipping import of {module_name}")
        return None
    return importlib.import_module(module_name)



def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)



def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()



def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None



def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)



def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module



def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16):
    if use_fp16:
        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters()
        ]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_master_grads(master_params):
    for param in master_params:
        param.grad = None


def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return torch.zeros_like(param)
    

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
            


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0
    

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None




class MixedPrecisionTrainer:
    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()

    def zero_grad(self):
        zero_grad(self.model_params)

    def backward(self, loss: torch.Tensor):
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize(self, opt: torch.optim.Optimizer):
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: torch.optim.Optimizer):
        logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)
        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2 ** self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_master_grads(self.master_params)
            return False

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        opt.step()
        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: torch.optim.Optimizer):
        grad_norm, param_norm = self._compute_norms()
        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)
        opt.step()
        return True

    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with torch.no_grad():
                param_norm += torch.norm(p, p=2, dtype=torch.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += torch.norm(p.grad, p=2, dtype=torch.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)



def process_checkpoints(checkpoint):
    """convert the checkpoint from parallel to single gpu:

    Args:
        checkpoint (_type_): _description_
    """
    new_state_dict_model = {}
    for key in checkpoint['model'].keys():
        new_key = key.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict_model[new_key] = checkpoint['model'][key]
    checkpoint['model'] = new_state_dict_model
    
    new_state_dict_model = {}
    if checkpoint.get('model_ema', None) is not None:
        for key in checkpoint['model_ema'].keys():
            new_key = key.replace("module.", "")  # Remove 'module.' prefix
            new_state_dict_model[new_key] = checkpoint['model_ema'][key]
        checkpoint['model_ema'] = new_state_dict_model
    
    return checkpoint


def mix_images_with_masks(images, masks, alpha_heatmap=0.5, colormap='jet'):
    """
    Mixes images with unsigned distance functions (USDFs) using a specified colormap and alpha blending.
    Args:
        images (torch.Tensor): A batch of images with shape (N, C, H, W).
        masks (torch.Tensor): A batch of masks with shape (N, 1, H, W).
        alpha_heatmap (float, optional): The blending factor for the heatmap overlay. Default is 0.5.
        colormap (str, optional): The colormap to use for the SDFs. Default is 'jet'.
    Returns:
        numpy.ndarray: The resulting images with the SDF heatmap overlay, with shape (N, H, W, C).
    """

    cmap = matplotlib.colormaps.get_cmap(colormap)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    masks = masks.squeeze(1).cpu().numpy()
    
    # normalize masks
    masks = min_max_normalize(masks)
    
    rgb_heatmaps_np = cmap(masks)[..., :3]
    overlayed_images = (1 - alpha_heatmap) * images + alpha_heatmap * rgb_heatmaps_np
    return np.clip(overlayed_images, a_min=0., a_max=1.)


def compute_metrics(preds, gts, mask_name, metric, thresh=126, gt_type='sdf_map'):
    preds = preds.squeeze(1).cpu().numpy()
    gts = gts.squeeze(1).cpu().numpy()
    preds = min_max_normalize(preds)
    # gts = min_max_normalize(gts)
    if gt_type == 'mask':
        thresh = 127
    
    preds = (255 * preds >= thresh)
    gts = (255 * gts > 0)
    # visualization_for_debug(preds, gts, mask_name)
    if metric == 'iou':
        intersection = np.logical_and(preds, gts)
        union = np.logical_or(preds, gts)
        intersection_batch = np.sum(intersection.astype('float32'), axis=(1, 2))
        union_batch = np.sum(union.astype('float32'), axis=(1, 2))
        
        mask = np.where(union_batch > 0, 1., 0.)
        union_batch = np.where(union_batch > 0, union_batch, 1e-3)
        outcome = intersection_batch * mask / union_batch
    elif metric == 'dice':
        intersection = np.logical_and(preds, gts)
        intersection_batch = np.sum(intersection.astype('float32'), axis=(1, 2))
        pred_batch = np.sum(preds.astype('float32'), axis=(1, 2))
        gt_batch = np.sum(gts.astype('float32'), axis=(1, 2))
        denorminztor = pred_batch + gt_batch
        
        mask = np.where(denorminztor > 0., 1., 0.)
        denorminztor = np.where(denorminztor > 0., denorminztor, 1e-3)
        outcome = (2 * intersection_batch * mask) / denorminztor
        
    return outcome
    
    
    
def save_batch(mixed_img_predits_I, mixed_img_predits_II, mixed_img_gts, mask_names, vis_path):
    num_examples = mixed_img_predits_I.shape[0]
    for i in range(num_examples):
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].imshow(mixed_img_predits_I[i])
        ax[0].axis('off')
        ax[0].set_title('Predictions I')   
        ax[1].imshow(mixed_img_predits_II[i])
        ax[1].axis('off')
        ax[1].set_title('Predictions II')   
        ax[2].imshow(mixed_img_gts[i])
        ax[2].axis('off')
        ax[2].set_title('Ground truths')   
        plt.savefig(os.path.join(vis_path, mask_names[i]))
        plt.close(fig)
        
        
        

def min_max_normalize(usdfs):
    
    mim_usdfs = np.min(usdfs, axis=(1, 2), keepdims=True)
    max_usdfs = np.max(usdfs, axis=(1, 2), keepdims=True)
    nonzero_max = (max_usdfs > 0).astype('float32')
    max_usdfs = np.where(max_usdfs > 0, max_usdfs, 1e-6)
    return nonzero_max * (usdfs - mim_usdfs) / (max_usdfs - mim_usdfs)


def visualization_for_debug(preds, gts, mask_name, save_dir='experiments/check_res'):
    
    B = preds.shape[0]
    for i in range(B):
        pred = 255 * preds[i].astype('float32')
        gt = gts[i].astype('float32')
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(pred)
        ax[0].axis('off')
        ax[0].set_title('Predictions')  
        ax[1].imshow(gt)
        ax[1].axis('off')
        ax[1].set_title('GroundTruth')  
        plt.savefig(os.path.join(save_dir, mask_name[i]))
        plt.close(fig)
    
    


