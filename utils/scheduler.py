import math
import types
import tqdm
import torch 
import pickle
import warnings
import numpy as np 

from torch import nn
from functools import partial 
from typing import Any, Callable
from torch.optim import Optimizer
from monai.utils import StateCacher
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from monai.networks.utils import eval_mode
from torch.serialization import DEFAULT_PROTOCOL
from torch.optim.lr_scheduler import _LRScheduler 
from monai.optimizers.lr_scheduler import ExponentialLR, LinearLR



class LearningRateFinder:
    """Learning rate range test.

    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.

    Example (fastai approach):
    >>> lr_finder = LearningRateFinder(net, optimizer, criterion)
    >>> lr_finder.range_test(data_loader, end_lr=100, num_iter=100)
    >>> lr_finder.get_steepest_gradient()
    >>> lr_finder.plot() # to inspect the loss-learning rate graph

    Example (Leslie Smith's approach):
    >>> lr_finder = LearningRateFinder(net, optimizer, criterion)
    >>> lr_finder.range_test(train_loader, val_loader=val_loader, end_lr=1, num_iter=100, step_mode="linear")

    Gradient accumulation is supported; example:
    >>> train_data = ...    # prepared dataset
    >>> desired_bs, real_bs = 32, 4         # batch size
    >>> accumulation_steps = desired_bs // real_bs     # required steps for accumulation
    >>> data_loader = torch.utils.data.DataLoader(train_data, batch_size=real_bs, shuffle=True)
    >>> acc_lr_finder = LearningRateFinder(net, optimizer, criterion)
    >>> acc_lr_finder.range_test(data_loader, end_lr=10, num_iter=100, accumulation_steps=accumulation_steps)

    By default, image will be extracted from data loader with x["image"] and x[0], depending on whether
    batch data is a dictionary or not (and similar behaviour for extracting the label). If your data loader
    returns something other than this, pass a callable function to extract it, e.g.:
    >>> image_extractor = lambda x: x["input"]
    >>> label_extractor = lambda x: x[100]
    >>> lr_finder = LearningRateFinder(net, optimizer, criterion)
    >>> lr_finder.range_test(train_loader, val_loader, image_extractor, label_extractor)

    References:
    Modified from: https://github.com/davidtvs/pytorch-lr-finder.
    Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: torch.nn.Module,
        device: str | torch.device | None = None,
        memory_cache: bool = True,
        cache_dir: str | None = None,
        amp: bool = False,
        pickle_module: types.ModuleType = pickle,
        pickle_protocol: int = DEFAULT_PROTOCOL,
        verbose: bool = True,
    ) -> None:
        """Constructor.

        Args:
            model: wrapped model.
            optimizer: wrapped optimizer.
            criterion: wrapped loss function.
            device: device on which to test. run a string ("cpu" or "cuda") with an
                optional ordinal for the device type (e.g. "cuda:X", where is the ordinal).
                Alternatively, can be an object representing the device on which the
                computation will take place. Default: None, uses the same device as `model`.
            memory_cache: if this flag is set to True, `state_dict` of
                model and optimizer will be cached in memory. Otherwise, they will be saved
                to files under the `cache_dir`.
            cache_dir: path for storing temporary files. If no path is
                specified, system-wide temporary directory is used. Notice that this
                parameter will be ignored if `memory_cache` is True.
            amp: use Automatic Mixed Precision
            pickle_module: module used for pickling metadata and objects, default to `pickle`.
                this arg is used by `torch.save`, for more details, please check:
                https://pytorch.org/docs/stable/generated/torch.save.html#torch.save.
            pickle_protocol: can be specified to override the default protocol, default to `2`.
                this arg is used by `torch.save`, for more details, please check:
                https://pytorch.org/docs/stable/generated/torch.save.html#torch.save.
            verbose: verbose output
        Returns:
            None
        """
        # Check if the optimizer is already attached to a scheduler
        self.optimizer = optimizer
        self._check_for_scheduler()

        self.model = model
        self.criterion = criterion
        self.history: dict[str, list] = {"lr": [], "loss": []}
        self.memory_cache = memory_cache
        self.cache_dir = cache_dir
        self.amp = amp
        self.verbose = verbose

        # Save the original state of the model and optimizer so they can be restored if
        # needed
        self.model_device = next(self.model.parameters()).device
        self.state_cacher = StateCacher(
            in_memory=memory_cache, cache_dir=cache_dir, pickle_module=pickle_module, pickle_protocol=pickle_protocol
        )
        self.state_cacher.store("model", self.model.state_dict())
        self.state_cacher.store("optimizer", self.optimizer.state_dict())

        # If device is None, use the same as the model
        self.device = device if device else self.model_device

    def reset(self) -> None:
        """Restores the model and optimizer to their initial states."""

        self.model.load_state_dict(self.state_cacher.retrieve("model"))
        self.optimizer.load_state_dict(self.state_cacher.retrieve("optimizer"))
        self.model.to(self.model_device)

    def range_test(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_training_sample: int | None = None,
        num_validation_sample: int | None = None,
        start_lr: float | None = None,
        end_lr: float = 10.0,
        num_iter: int = 100,
        step_mode: str = "exp",
        smooth_f: float = 0.05,
        diverge_th: int = 5,
        accumulation_steps: int = 1,
        auto_reset: bool = True,
    ) -> None:
        """Performs the learning rate range test.

        Args:
            train_loader: training set data loader.
            val_loader: validation data loader (if desired).
            image_extractor: callable function to get the image from a batch of data.
                Default: `x["image"] if isinstance(x, dict) else x[0]`.
            label_extractor: callable function to get the label from a batch of data.
                Default: `x["label"] if isinstance(x, dict) else x[1]`.
            start_lr : the starting learning rate for the range test.
                The default is the optimizer's learning rate.
            end_lr: the maximum learning rate to test. The test may stop earlier than
                this if the result starts diverging.
            num_iter: the max number of iterations for test.
            step_mode: schedule for increasing learning rate: (`linear` or `exp`).
            smooth_f: the loss smoothing factor within the `[0, 1[` interval. Disabled
                if set to `0`, otherwise loss is smoothed using exponential smoothing.
            diverge_th: test is stopped when loss surpasses threshold:
                `diverge_th * best_loss`.
            accumulation_steps: steps for gradient accumulation. If set to `1`,
                gradients are not accumulated.
            non_blocking_transfer: when `True`, moves data to device asynchronously if
                possible, e.g., moving CPU Tensors with pinned memory to CUDA devices.
            auto_reset: if `True`, returns model and optimizer to original states at end
                of test.
        Returns:
            None
        """

        # Reset test results
        self.history = {"lr": [], "loss": []}
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_training_sample = num_training_sample
        self.num_validation_sample = num_validation_sample
        best_loss = -float("inf")

        # Move the model to the proper device
        self.model.to(self.device)

        # Check if the optimizer is already attached to a scheduler
        self._check_for_scheduler()

        # Set the starting learning rate
        if start_lr:
            self._set_learning_rate(start_lr)

        # Check number of iterations
        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")

        # Initialize the proper learning rate policy
        lr_schedule: ExponentialLR | LinearLR
        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
        else:
            raise ValueError(f"expected one of (exp, linear), got {step_mode}")

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1[")

        # Create an iterator to get data batch by batch
        # train_iter = TrainDataLoaderIter(train_loader, image_extractor, label_extractor)
        train_iter = iter(train_loader)

        trange: partial[tqdm.trange] | type[range]
        if self.verbose:
            trange = partial(tqdm.trange, desc="Computing optimal learning rate")
            tprint = tqdm.tqdm.write
        else:
            trange = range
            tprint = print

        for iteration in trange(num_iter):
            
            # Train on batch and retrieve loss
            loss = self._train_batch(train_iter, accumulation_steps)
            if val_loader:
                loss = self._validate()

            # Update the learning rate
            self.history["lr"].append(lr_schedule.get_lr()[0])
            lr_schedule.step()

            # Track the best loss and smooth it if smooth_f is specified
            if iteration == 0:
                best_loss = loss
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self.history["loss"][-1]
                if loss < best_loss:
                    best_loss = loss

            # Check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(loss)
            if loss > diverge_th * best_loss:
                if self.verbose:
                    tprint("Stopping early, the loss has diverged")
                break

        if auto_reset:
            if self.verbose:
                print("Resetting model and optimizer")
            self.reset()

    def _set_learning_rate(self, new_lrs: float | list) -> None:
        """Set learning rate(s) for optimizer."""
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
        if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "Length of `new_lrs` is not equal to the number of parameter groups " + "in the given optimizer"
            )

        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr

    def _check_for_scheduler(self):
        """Check optimizer doesn't already have scheduler."""
        for param_group in self.optimizer.param_groups:
            if "initial_lr" in param_group:
                raise RuntimeError("Optimizer already has a scheduler attached to it")

    def _train_batch(self, train_iter, accumulation_steps: int) -> float:
        self.model.concept_head.train()
        total_loss = 0

        self.optimizer.zero_grad()
        for i in range(accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            images = batch['pixel_values'].to(self.device)
            cams = batch['inter_map'].to(self.device)
            # sdf_maps = batch['sdf_map'].to(self.device)
            onehot_maps = batch['mask'].to(self.device)
            
            # Forward pass
            concept_weights = self.model(images)
            preds = torch.sum(concept_weights[..., None, None] * cams, dim=1, keepdim=True)
            loss = self.criterion(preds, onehot_maps, concept_weights)

            # Loss should be averaged in each step
            loss /= accumulation_steps

            # Backward pass
            if self.amp and hasattr(self.optimizer, "_amp_stash"):
                # For minor performance optimization, see also:
                # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
                delay_unscale = ((i + 1) % accumulation_steps) != 0

                with torch.cuda.amp.scale_loss(loss, self.optimizer, delay_unscale=delay_unscale) as scaled_loss:  # type: ignore
                    scaled_loss.backward()
            else:
                loss.backward()

            total_loss += loss.item()

        self.optimizer.step()

        return total_loss

    def _validate(self) -> float:
        # Set model to evaluation mode and disable gradient computation
        running_loss = 0
        with eval_mode(self.model):
            for batch in self.val_loader:
                # Copy data to the correct device
                images = batch['pixel_values'].to(self.device)
                cams = batch['inter_map'].to(self.device)
                # sdf_maps = batch['sdf_map'].to(self.device)
                onehot_maps = batch['mask'].to(self.device)

                # Forward pass and loss computation
                concept_weights = self.model(images)
                preds = torch.sum(concept_weights[..., None, None] * cams, dim=1, keepdim=True)
                # preds = postprocess_pred(preds)
                loss = self.criterion(preds, onehot_maps, concept_weights)
                
                running_loss += loss.item() * onehot_maps.shape[0]

        return running_loss / self.num_validation_sample

    def get_lrs_and_losses(self, skip_start: int = 0, skip_end: int = 0) -> tuple[list, list]:
        """Get learning rates and their corresponding losses

        Args:
            skip_start: number of batches to trim from the start.
            skip_end: number of batches to trim from the end.
        """
        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        lrs = self.history["lr"]
        losses = self.history["loss"]
        end_idx = len(lrs) - skip_end - 1
        lrs = lrs[skip_start:end_idx]
        losses = losses[skip_start:end_idx]

        return lrs, losses

    def get_steepest_gradient(self, skip_start: int = 0, skip_end: int = 0) -> tuple[float, float] | tuple[None, None]:
        """Get learning rate which has steepest gradient and its corresponding loss

        Args:
            skip_start: number of batches to trim from the start.
            skip_end: number of batches to trim from the end.

        Returns:
            Learning rate which has steepest gradient and its corresponding loss
        """
        lrs, losses = self.get_lrs_and_losses(skip_start, skip_end)

        try:
            min_grad_idx = np.gradient(np.array(losses)).argmin()
            return lrs[min_grad_idx], losses[min_grad_idx]
        except ValueError:
            print("Failed to compute the gradients, there might not be enough points.")
            return None, None

    def plot(
        self,
        skip_start: int = 0,
        skip_end: int = 0,
        log_lr: bool = True,
        ax: Any | None = None,
        steepest_lr: bool = True,
    ) -> Any | None:
        """Plots the learning rate range test.

        Args:
            skip_start: number of batches to trim from the start.
            skip_end: number of batches to trim from the start.
            log_lr: True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale.
            ax: the plot is created in the specified matplotlib axes object and the
                figure is not be shown. If `None`, then the figure and axes object are
                created in this method and the figure is shown.
            steepest_lr: plot the learning rate which had the steepest gradient.

        Returns:
            The `matplotlib.axes.Axes` object that contains the plot. Returns `None` if
            `matplotlib` is not installed.
        """

        lrs, losses = self.get_lrs_and_losses(skip_start, skip_end)

        # Create the figure and axes object if axes was not already given
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)

        # Plot the LR with steepest gradient
        if steepest_lr:
            lr_at_steepest_grad, loss_at_steepest_grad = self.get_steepest_gradient(skip_start, skip_end)
            if lr_at_steepest_grad is not None and loss_at_steepest_grad is not None:
                ax.scatter(
                    lr_at_steepest_grad,
                    loss_at_steepest_grad,
                    s=75,
                    marker="o",
                    color="red",
                    zorder=3,
                    label="steepest gradient",
                )
                ax.legend()

        if log_lr:
            ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")

        # Show only if the figure was created internally
        if fig is not None:
            plt.show()

        return ax




class WarmupExponentialLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, gamma, last_epoch=-1):
        
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        super(WarmupExponentialLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / float(self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # ExponentialLR 
        decay_step = self.last_epoch - self.warmup_steps
        return [base_lr * (self.gamma ** decay_step) for base_lr in self.base_lrs]



class MultiStageLRScheduler(_LRScheduler):
    def __init__(self, optimizer, milestones, factors, last_epoch=-1):
        """
        Multi-stage learning rate scheduler.
        :param optimizer: Optimizer
        :param milestones: List of epochs where learning rate changes
        :param factors: List of scaling factors for each milestone
        """
        
        self.milestones = milestones
        self.factors = factors
        assert len(milestones) == len(factors), "Milestones and factors must have the same length."
        super(MultiStageLRScheduler, self).__init__(optimizer, last_epoch)
        

    def get_lr(self):
        # factor accoring to the current learning rate
        factor = 1.0
        for milestone, f in zip(self.milestones, self.factors):
            if self.last_epoch >= milestone:
                factor = f
            else:
                break

        # return the learning rate after modification
        return [base_lr * factor for base_lr in self.base_lrs]



class WarmupCosineLRScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super(WarmupCosineLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warm-up stage：learning rate growing
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            return [
                base_lr * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]
            
