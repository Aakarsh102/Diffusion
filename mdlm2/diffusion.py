import itertools
import math
import os
import json
import typing
from dataclasses import dataclass

import hydra.utils
from hydra.core.hydra_config import HydraConfig
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

import dataloader
import models
import noise_schedule
import utils

LOG2 = math.log(2)


def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _gather_token_probabilities(q_xs: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
  """Return per-position probability for chosen tokens.

  Args:
    q_xs: [B, L, V] nonnegative weights (will be normalized across V)
    tokens: [B, L] chosen token ids

  Returns:
    probs: [B, L] probability mass assigned to the chosen token at each position
  """
  return torch.gather(q_xs, dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    # In __init__
    self.esm_model = None
    self.esm_tokenizer = None
    esm_path = '/scratch/gilbreth/rai53/esm2_650M'
    print("Loading ESM-2...")
    self.esm_model = transformers.EsmForMaskedLM.from_pretrained(esm_path).eval()
    self.esm_tokenizer = transformers.AutoTokenizer.from_pretrained(esm_path)
    print("✓ ESM-2 loaded")

    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.sampling.predictor
    self.gen_ppl_eval_model_name_or_path = self.config.eval.\
      gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    # elif self.config.backbone == 'dimamba':
    #   self.backbone = models.dimamba.DiMamba(
    #     self.config,
    #     vocab_size=self.vocab_size,
    #     pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'ar':
      self.backbone = models.autoregressive.AR(
        self.config,
        vocab_size=self.vocab_size,
        mask_index=self.mask_index)
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.T = self.config.T
    self.subs_masking = self.config.subs_masking
    # Use ordering-based masking if enabled in config
    # self.ordering_masking = getattr(self.config, 'ordering_masking', False)

    self.softplus = torch.nn.Softplus()
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')

    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    self.eval_model_tokenizer = transformers.AutoTokenizer.\
      from_pretrained(self.gen_ppl_eval_model_name_or_path)
    if self.eval_model_tokenizer.pad_token is None:
      self.eval_model_tokenizer.pad_token =\
          self.eval_model_tokenizer.eos_token
      self.eval_model_tokenizer.pad_token_id =\
          self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

    # Added by Ziyi:
    self.generated_samples = []
    self.confidence = None

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'
    if self.config.mask_order.name != 'random':
      # Requires discrete time so that steps align with ordering list
      assert self.T > 0, 'order mask requires T > 0 (discrete steps)'
  @torch.no_grad()
  def compute_esm2_perplexity(self, sequences):
    """Use ESM-2 (masked LM) to score naturalness"""
    if self.esm_model is None or self.esm_tokenizer is None:
        print("ESM-2 not loaded, skipping perplexity")
        return 0.0

    # Move to device if needed
    if next(self.esm_model.parameters()).device != self.device:
        self.esm_model = self.esm_model.to(self.device)

    nlls = []
    for seq in sequences:
        inputs = self.esm_tokenizer(seq, return_tensors='pt').to(self.device)
        outputs = self.esm_model(**inputs, labels=inputs['input_ids'])
        nlls.append(outputs.loss.item())

    return np.exp(np.mean(nlls))  # perplexity
  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    try:
      self.print(
        f"[debug] accumulate_grad_batches={self.trainer.accumulate_grad_batches}, "
        f"log_every_n_steps={self.trainer.log_every_n_steps}, "
        f"devices={self.trainer.num_devices}, num_nodes={self.trainer.num_nodes}, "
        f"per_device_batch_size={self.config.loader.batch_size}, global_batch_size={self.config.loader.global_batch_size}")
    except Exception:
      pass
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      print("[debug train start] distributed sampler is invoked.")
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      print("[debug train start] random sampler is invoked.")
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []

    for dl in self.trainer.fit_loop._combined_loader.flattened:
      original_collate = getattr(dl, 'collate_fn', None)
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      
      print("[debug train start] what dl: ", dl)
      if isinstance(dl.dataset, dataloader.SyntheticDatasetWithPredefinedMaskOrder):
        updated_dls.append(
          torch.utils.data.DataLoader(
            dl.dataset,
            batch_size=self.config.loader.batch_size,
            num_workers=self.config.loader.num_workers,
            pin_memory=self.config.loader.pin_memory,
            sampler=dl_sampler,
            shuffle=False,
            collate_fn=dataloader.collate_synthetic_with_order,
            persistent_workers=False))
      else:
        updated_dls.append(
          torch.utils.data.DataLoader(
            dl.dataset,
            batch_size=self.config.loader.batch_size,
            num_workers=self.config.loader.num_workers,
            pin_memory=self.config.loader.pin_memory,
            sampler=dl_sampler,
            shuffle=False,
            collate_fn=original_collate,
            persistent_workers=False))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    try:
      self.print(f"[debug] optimizer_step used | global_step={self.global_step}")
    except Exception:
      pass
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)

    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _d3pm_parameterization(self, logits):
    if self.subs_masking:
      logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    #with torch.cuda.amp.autocast(dtype=torch.float32):
    logits = self.backbone(x, sigma)
    logits = logits.float()

    
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    elif self.parameterization == 'd3pm':
      return self._d3pm_parameterization(logits=logits)
    return logits

  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T
    #eps = 1e-8
    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)
#    alpha_s = 1 - (t - dt).clamp(min=eps) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
   # term_1_coef = dt / t.clamp(min=eps)
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
#    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t.clamp(min=eps) + 1)  # ADD .clamp(min=eps)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
#    term_2_coef = 1 - dt / t.clamp(min=eps)
    term_2_log_nr = term_1_log_nr
#    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / ((t - dt).clamp(min=eps)) + 1)  # ADD .clamp(min=eps)

    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)
    
    # utils.debug_tensor(term_2_log_dr, 'term_2_log_dr')
    # print('[debug loss] T: ', self.T)
    # print('[debug loss] t-dt: ', t-dt)

    # Debug: print diagnostics if non-finite values appear
    if not torch.isfinite(L_vb).all():
      try:
        print("[debug] _d3pm_loss detected non-finite L_vb")
        utils.debug_tensor(t if torch.is_tensor(t) else torch.tensor(t, device=model_output.device), 't')
        utils.debug_tensor(torch.tensor(dt, device=model_output.device), 'dt')
        utils.debug_tensor(alpha_t, 'alpha_t')
        utils.debug_tensor(alpha_s, 'alpha_s')
        utils.debug_tensor(model_output, 'model_output')
        utils.debug_tensor(x0.to(model_output.dtype), 'x0')
        utils.debug_tensor(xt.to(model_output.dtype), 'xt')
        utils.debug_tensor(log_x_theta_at_x0, 'log_x_theta_at_x0')
        utils.debug_tensor(log_x_theta_at_m, 'log_x_theta_at_m')
        utils.debug_tensor(x_theta_at_m, 'x_theta_at_m')
        utils.debug_tensor(term_1_coef, 'term_1_coef')
        utils.debug_tensor(term_1_log_nr, 'term_1_log_nr')
        utils.debug_tensor(term_1_log_dr, 'term_1_log_dr')
        utils.debug_tensor(term_2_coef, 'term_2_coef')
        utils.debug_tensor(term_2_log_nr, 'term_2_log_nr')
        utils.debug_tensor(term_2_log_dr, 'term_2_log_dr')
        utils.debug_tensor(L_vb_masked, 'L_vb_masked')
        utils.debug_tensor(L_vb, 'L_vb')
      except Exception as e:
        print(f"[debug] _d3pm_loss debug failed with {e}")

    return self.T * L_vb

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    # Optional: pull unmask_order (list of lists per sample) when ordering_masking is on
    unmask_order = batch.get('unmask_order', None)
    flat_order = batch.get('unmask_order_flat', None)
    # if self.ordering_masking:
    #   if unmask_order is None:
    #     raise ValueError('ordering_masking enabled but batch lacks unmask_order')
    #   # Sanity: each sample's order length must equal T
    #   for i, order_i in enumerate(unmask_order):
    #     if not isinstance(order_i, (list, tuple)):
    #       raise ValueError(f'unmask_order[{i}] must be list of lists')
    #     if len(order_i) != self.T:
    #       raise AssertionError(f'For sample {i}, len(unmask_order)={len(order_i)} != T={self.T}')
    losses = self._loss(batch['input_ids'], attention_mask, unmask_order=unmask_order, flat_order=flat_order)
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    try:
      accum = batch_idx % max(1, self.trainer.accumulate_grad_batches)
      self.print(
        f"[debug] batch_idx={batch_idx}, accum_step={accum}/{self.trainer.accumulate_grad_batches}, global_step={self.global_step}")
    except Exception:
      pass
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

 # def on_validation_epoch_end(self):
 #   if ((self.config.eval.compute_perplexity_on_sanity
 #        or not self.trainer.sanity_checking)
 #        and self.config.eval.generate_samples
 #        and not self.parameterization == 'ar'):
 #     # TODO(justin): implement sampling and kv cache for AR
 #     samples, text_samples = None, None
 #     for _ in range(
 #       self.config.sampling.num_sample_batches):
 #       samples = self._sample()
 #       # Decode the samples to be re-tokenized by eval model
 #       text_samples = self.tokenizer.batch_decode(samples)
 #       if self.config.eval.compute_generative_perplexity:
 #         self.compute_generative_perplexity(text_samples)
 #     if self.trainer.global_rank == 0 and hasattr(
 #       self.trainer.logger, 'log_table'):
 #       # Log the last generated samples
 #       text_samples = text_samples[
 #         : self.config.sampling.num_sample_log]
 #       self.trainer.logger.log_table(
 #         key=f'samples@global_step{self.global_step}',
 #         columns=['Generated Samples'],
 #         data=[[s] for s in text_samples])
 #     if self.config.eval.compute_generative_perplexity:
 #       self.log('val/gen_ppl',
 #                self.gen_ppl_metric,
 #                on_epoch=True,
 #                on_step=False,
 #                sync_dist=True)
 #   if self.global_step % 100 == 0 and self.global_step > 0:
 #       metrics, _ = self.evaluate_generation_quality(num_samples=25)
 #       for k, v in metrics.items():
 #           self.log(f'gen/{k}', v, on_epoch=True, sync_dist=True)
 #       print(f"\n[Step {self.global_step}] Generation Metrics:")
 #       for k, v in metrics.items():
#            print(f"  {k}: {v:.2f}")
#    if self.ema:
#      self.ema.restore(
#        itertools.chain(self.backbone.parameters(),
#                        self.noise.parameters()))

  def on_validation_epoch_end(self):
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.parameterization == 'ar'):
      # TODO(justin): implement sampling and kv cache for AR
      samples, text_samples = None, None
      for _ in range(
        self.config.sampling.num_sample_batches):
        samples = self._sample()
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.compute_generative_perplexity(text_samples)
      if self.trainer.global_rank == 0 and hasattr(
        self.trainer.logger, 'log_table'):
        # Log the last generated samples
        text_samples_log = text_samples[
          : self.config.sampling.num_sample_log]
        self.trainer.logger.log_table(
          key=f'samples@global_step{self.global_step}',
          columns=['Generated Samples'],
          data=[[s] for s in text_samples_log])

      # ESM-2 perplexity logging
      if text_samples is not None and self.global_step % 100 == 0 and self.global_step > 0:
        try:
          esm_ppl = self.compute_esm2_perplexity(text_samples[:40])
          self.log('gen/esm2_ppl', esm_ppl,
                   on_step=False, on_epoch=True,
                   sync_dist=True, prog_bar=True)
          print(f"\n[Step {self.global_step}] ESM-2 Perplexity: {esm_ppl:.2f}")
        except Exception as e:
          print(f"ESM-2 perplexity computation failed: {e}")

      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))
  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def compute_generative_perplexity(
    self,
    text_samples: typing.List[str],
    retokenize: bool = True,
    max_length: typing.Optional[int] = None) -> None:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
    
    Returns:
        Perplexity of the generated text under a different
        pre-trained AR model (e.g., GPT2).
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path).eval()
    if max_length is None:
      max_length = self.config.model.length
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(self.device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(
         text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    batch_size = min(
      self.config.eval.perplexity_batch_size,
      samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
        logits = eval_model(
          sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],
                               sample_chunk[..., 1:],
                               reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer\
                     .eos_token_id).cumsum(-1) == 1
        token_mask = (
          sample_chunk
          != self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(
          nlls, first_eos[..., 1:] + token_mask[..., 1:])

  def q_xt(self, x, t_or_move, unmask_order=None, flat_order=None):
    """Compute xt given either random-move mask or ordering-based mask.

    - If ordering_masking is enabled: t_or_move is the discrete time t in {1/T,..,1} (tensor [B]),
      and unmask_order is a list per batch element with T lists of indices. The masking at step k
      is the complement of indices revealed in steps [k..T].
    - Else: t_or_move is move_chance and we sample mask positions at random.
    """
    if self.config.mask_order.name == 'denoise':
      assert unmask_order is not None, 'ordering_masking requires unmask_order'
      t = t_or_move
      if torch.is_tensor(t) and t.ndim > 1:
        t = t.squeeze(-1)
      # map t in {1/T,2/T,...,1} to integer step k in [1..T]
      k = (t * self.T).clamp(min=1.0, max=float(self.T)).to(torch.int64).tolist()
      if self.config.mask_order.get('reverse_curriculum', False):
        k = [self.T - k_i + 1 for k_i in k]
      xt = x.clone()
      B, L = x.shape
      if hasattr(self, 'global_step') and self.global_step % 500 == 0:
            k0 = int(k[0])
            mask_pos = self._mask_positions_from_order(unmask_order[0], k0, L, x.device)
            n_masked = mask_pos.sum().item()
            print(f"[denoise-debug] step_k={k0}, n_masked={n_masked}/{L} ({100*n_masked/L:.1f}%)")
      for b in range(B):
        if self.config.mask_order.type == 'consecutive':
          mask_positions = self._mask_positions_from_flat_consecutive(flat_order[b], int(k[b]), L, x.device)
        else:
          mask_positions = self._mask_positions_from_order(unmask_order[b], int(k[b]), L, x.device)
        if mask_positions is None:
          xt[b] = self.mask_index
        else:
          xt[b, mask_positions] = self.mask_index
      # After the masking loop in q_xt:
      if self.config.mask_order.name == 'denoise' and hasattr(self, 'global_step') and self.global_step % 500 == 0:
        # Check first sample in batch
        n_masked = (xt[0] == self.mask_index).sum().item()
        n_total = xt.shape[1]
        print(f"[mask-check] k={int(k[0])}, masked={n_masked}/{n_total} ({100*n_masked/n_total:.1f}%)")
      return xt
    elif self.config.mask_order.name == "ar":
      t = t_or_move
      if torch.is_tensor(t) and t.ndim > 1:
        t = t.squeeze(-1)
      # map t in {1/T,2/T,...,1} to integer step k in [1..T]
      k = (t * self.T).clamp(min=1.0, max=float(self.T)).to(torch.int64).tolist()
      xt = x.clone()
      B, L = x.shape
      for b in range(B):
        kb = int(k[b])
        kb = max(1, min(kb, L))
        if kb >= L:
          xt[b] = self.mask_index
        else:
          xt[b, L - kb:] = self.mask_index
      # Optional debug print
      try:
        if getattr(self.config.training, 'debug_values', False):
          if hasattr(self, 'global_step') and (self.global_step % 1 == 0):
            kb0 = int(k[0]) if isinstance(k, list) else int(k.item())
            kb0 = max(1, min(kb0, L))
            masked_idx = list(range(max(0, L - kb0), L))
            masked_tokens = x[0, max(0, L - kb0): L].detach().cpu().tolist()
            try:
              masked_pieces = self.ids_to_pieces(masked_tokens)
            except Exception:
              masked_pieces = []
            print(
              f"[ar-mask-debug] step_k={kb0} L={L} masked_idx={masked_idx} "
              f"masked_tokens={masked_tokens} masked_pieces={masked_pieces}")
      except Exception:
        pass
      return xt
    else:
      move_chance = t_or_move
      move_indices = torch.rand(*x.shape, device=x.device) < move_chance
      xt = torch.where(move_indices, self.mask_index, x)
      return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)

  def _mask_positions_from_order(self, order, step_k: int, seq_len: int, device):
    """Return boolean mask (length seq_len) of positions to set to MASK at step_k.
    Reverse order rule: mask the union of the last k steps in `order`.
    Example: k=1 -> mask indices in order[-1]; k=2 -> mask order[-2] U order[-1].
    """
    if not isinstance(order, (list, tuple)) or len(order) == 0:
      return None
    step_k = max(1, int(step_k))
    # Union of last k steps to mask
    to_mask = set()
    start = max(0, len(order) - step_k)
    for step in range(start, len(order)):
      step_indices = order[step] or []
      for idx in step_indices:
        ii = int(idx)
        if 0 <= ii < seq_len:
          to_mask.add(ii)
    # Build mask (True = mask)
    mask_positions = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if to_mask:
      idx_tensor = torch.tensor(sorted(to_mask), dtype=torch.long, device=device)
      mask_positions[idx_tensor] = True
    # Optional sanity-debug print (set training.debug_values: True)
    try:
      if getattr(self.config.training, 'debug_values', False):
        # Print sparsely to reduce I/O
        if hasattr(self, 'global_step') and (self.global_step % 1 == 0):
          _sample_mask = sorted(list(to_mask))[:10]
          _n_masked = int(mask_positions.sum().item())
          print(
            f"[order-mask-debug] step_k={step_k} seq_len={seq_len} "
            f"n_to_mask={len(to_mask)} sample_to_mask={_sample_mask} "
            f"n_masked={_n_masked}")
    except Exception:
      pass
    return mask_positions

  def _mask_positions_from_flat_consecutive(self, flat_order_b: torch.Tensor, step_k: int, seq_len: int, device):
    """Return boolean mask (length seq_len) using reversed flat order.

    Takes the last k indices from flat_order_b (by reversing and slicing) and
    builds a boolean mask of length seq_len with those indices set to True.
    """
    if flat_order_b is None or not torch.is_tensor(flat_order_b):
      return None
    if flat_order_b.ndim != 1:
      flat_order_b = flat_order_b.view(-1)
    kb = int(step_k)
    kb = max(0, min(kb, flat_order_b.shape[0]))
    mask_positions = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if kb == 0:
      return mask_positions
    idx = torch.flip(flat_order_b, dims=[-1])[:kb].to(dtype=torch.long, device=device)
    if idx.numel() > 0:
      valid_idx = idx[(idx >= 0) & (idx < seq_len)]
      if valid_idx.numel() > 0:
        mask_positions[valid_idx] = True
    return mask_positions

  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    if p_x0 is None:
      p_x0 = self.forward(x, sigma_t).exp()
    
    assert move_chance_t.ndim == p_x0.ndim
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]

    _x = _sample_categorical(q_xs)
    
    # Added by Ziyi to implement self-confidence denoising schedule
    if self.config.denoise_scheduler.name == 'confidence':

      masked_indices = (_x != self.mask_index)
      confidence = _gather_token_probabilities(q_xs, _x)
      confidence = torch.where(masked_indices, confidence, -np.inf)

      _, select_index = torch.topk(confidence, k=self.config.denoise_scheduler.k)
      decoded_indices = torch.zeros_like(x, dtype=torch.bool)
      decoded_indices.scatter_(-1, select_index, True)
      # Ziyi: Here I only want to keep the selected indices
      _x = torch.where(decoded_indices, _x, self.mask_index)

    else:
      if self.config.denoise_scheduler.save_confidence_order:
        masked_indices = (_x != self.mask_index)
        self.confidence = _gather_token_probabilities(q_xs, _x)
        self.confidence = torch.where(masked_indices, self.confidence, -np.inf)
      else:
        self.confidence = None

    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _ddpm_update(self, x, t, dt):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x

  def _ar_sampler(self, bsz):
    # precompute token buffer
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (bsz, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((bsz, num_pred_tokens, self.vocab_size))
             .to(self.device))
    for i in range(num_pred_tokens):
      next_logits = self.forward(x[:, :i + 1], None)[:, -1]
      y = (next_logits + noise[:, i]).argmax(-1)
      x[:, i + 1] = y
    return x

  @torch.no_grad()
  def ids_to_pieces(self, ids: list[int]) -> list[str]:
    # int -> str
    if isinstance(ids, int):
        return self.tokenizer.convert_ids_to_tokens([ids])[0]
    # List/tuple
    if isinstance(ids, (list, tuple)):
        if not ids:
            return []  # empty
        # Nested: List[List[int]]
        if isinstance(ids[0], (list, tuple)):
            return [self.tokenizer.convert_ids_to_tokens(step_ids) if step_ids else []
                    for step_ids in ids]
        # Flat: List[int]
        return self.tokenizer.convert_ids_to_tokens(list(ids))
    raise TypeError(f"Unsupported type for ids_to_pieces: {type(ids)}")

  @torch.no_grad()
  def debug_print_ar_mask(self, x, k):
    """Utility to preview AR masking of the last k tokens.

    Args:
      x: [B, L] token id tensor
      k: int number of tokens to mask at the end
    """
    try:
      if not torch.is_tensor(x) or x.ndim != 2:
        print("[ar-mask-debug] expected x as [B, L] tensor")
        return
      B, L = x.shape
      kb = int(k)
      kb = max(1, min(kb, L))
      max_print = min(2, B)
      for b in range(max_print):
        start = max(0, L - kb)
        idx = list(range(start, L))
        toks = x[b, start:L].detach().cpu().tolist()
        try:
          pieces = self.ids_to_pieces(toks)
        except Exception:
          pieces = []
        print(
          f"[ar-mask-debug] sample={b} step_k={kb} L={L} masked_idx={idx} "
          f"masked_tokens={toks} masked_pieces={pieces}")
    except Exception as e:
      print(f"[ar-mask-debug] debug_print_ar_mask failed: {e}")

  @torch.no_grad()
  def gather_unmasked_indices_and_tokens(
    self,
    x_prev: torch.Tensor, 
    x_next: torch.Tensor, 
    mask_id: int
):
    """
    For each sample in the batch, return (indices, token_ids) revealed at this step.
      x_prev, x_next: [B, L] token-id tensors
      mask_id: int id of the [MASK] token
    Returns: list of length B; each element is (indices_list, token_ids_list)
    """
    B, L = x_next.shape
    changed = (x_prev == mask_id) & (x_next != mask_id)   # [B, L] bool
    rows, cols = torch.nonzero(changed, as_tuple=True)    # 1D

    # Pre-allocate empty for all rows
    out = [([], []) for _ in range(B)]
    if rows.numel() == 0:
        return out

    counts = torch.bincount(rows, minlength=B)            # how many per row
    splits = counts.tolist()
    grouped_cols = torch.split(cols, splits)              # tuple length B

    for b in range(B):
        idxs_b = grouped_cols[b]
        if idxs_b.numel() == 0:
            continue
        # If enabled, order indices by descending confidence (do not modify confidence values)
        if (getattr(self.config, "denoise_scheduler", None) is not None
            and getattr(self.config.denoise_scheduler, "save_confidence_order", False)
            and (self.confidence is not None)):
            conf_vals = self.confidence[b, idxs_b]
            # Ensure confidence values are numeric and finite for safe sorting
            if not torch.is_floating_point(conf_vals):
                conf_vals = conf_vals.float()
            conf_vals = torch.nan_to_num(conf_vals, nan=-float('inf'), posinf=float('inf'), neginf=-float('inf'))
            order = torch.argsort(conf_vals, descending=True)
            idxs_b = idxs_b[order]
        idxs_list = idxs_b.detach().cpu().tolist()
        toks_list = x_next[b, idxs_b].detach().cpu().tolist()
        out[b] = (idxs_list, toks_list)

    return out

  # @torch.no_grad()
  # def ids_to_pieces(self, ids: list[int]) -> list[str]:
  #   if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "convert_ids_to_tokens"):
  #       return self.tokenizer.convert_ids_to_tokens(ids)
  #   # Fallback: strings of ints
  #   return [str(i) for i in ids]


  @torch.no_grad()
  def gather_unmasked_indices_per_step(self,
                                      x_prev: torch.Tensor,
                                      x_next: torch.Tensor,
                                      mask_id: int) -> list[list[int]]:

    """
    Returns: a Python list of length B; each element is a list[int] of positions
             that changed from MASK -> non-MASK at this step (S_t per sample).
    Shapes:
      x_prev, x_next: [B, L] (token ids)
    """
    B, L = x_next.shape
    changed = (x_prev == mask_id) & (x_next != mask_id)  # [B, L] bool
    rows, cols = torch.nonzero(changed, as_tuple=True)   # 1D tensors

    if rows.numel() == 0:
        return [[] for _ in range(B)]

    counts = torch.bincount(rows, minlength=B)  # how many indices per row
    splits = counts.tolist()
    grouped_cols = torch.split(cols, splits)    # tuple of length B

    # Convert to CPU Python lists (cheap; only ~L ints across all rows)
    return [gc.detach().cpu().tolist() for gc in grouped_cols]

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None
    
    # The object contains the full decoded text and the masked tokend indicies
    B, L = x.shape
    unmasked_steps = [[] for _ in range(B)]
    unmasked_tokens = [[] for _ in range(B)]

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x = self._ddpm_update(x, t, dt)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None

        # Modify here by Ziyi to store the masked tokens indicies
        per_b = self.gather_unmasked_indices_and_tokens(x, x_next, self.mask_index)
        for b in range(B):
          idxs_b, toks_b = per_b[b]
          unmasked_steps[b].append(idxs_b)   # may be []
          unmasked_tokens[b].append(toks_b)

        x = x_next
      else:
        x = self._analytic_update(x, t, dt)


    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        x = self.forward(x, unet_conditioning).argmax(dim=-1)


    # Added by Ziyi for adding the sample to the buffer
    for b in range(B):
        # decode final sequence -> x0
      x0_text = self.tokenizer.decode(x[b].tolist(), skip_special_tokens=False)

      record = {
          "x0": x0_text,
          # shape (T, L), 0/1 int tensor; keep on CPU to reduce GPU memory
          "x0_tokens": x[b].tolist(),
          "unmasked_indices": unmasked_steps[b],
          "unmasked_tokens":  unmasked_tokens[b],
          "unmasked_tokens_text": self.ids_to_pieces(unmasked_tokens[b]),
          "T": num_steps,
          "L": L
      }

      self.generated_samples.append(record)

    return x

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(num_steps=num_steps, eps=eps)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':
      # score(x, t) = p_t(y) / p_t(x)
      # => log score(x, t) = log p_t(y) - log p_t(x)
      
      # case 1: x = masked
      #   (i) y = unmasked
      #     log score(x, t) = log p_\theta(x)|_y + log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      #   (ii) y = masked
      #     log score(x, t) = 0

      # case 2: x = unmasked
      #   (i) y != masked, y != x
      #     log score(x_i, t) = - inf
      #   (ii) y = x 
      #     log score(x_i, t) = 0
      #   (iii) y = masked token
      #     log score(x_i, t) = - log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device):
    _eps_t = torch.rand(n, device=device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    t = t.clamp(min=2/self.T, max=1.0)
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(self, x0, unmask_order=None, flat_order=None):
    if self.config.mask_order.name == 'denoise':
        if unmask_order is None:
            print("[ERROR] denoise mode but unmask_order is None!")
        elif self.global_step % 100 == 0:
            print(f"[OK] unmask_order present, len={len(unmask_order[0])}")
    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    if self.config.mask_order.name != "random" and self.T > 0:
      xt = self.q_xt(x0, t, unmask_order=unmask_order, flat_order=flat_order)
    else:
      xt = self.q_xt(x0, move_chance)

    # Optional debug logging of t, move_chance, xt, x0
    try:
      if getattr(self.config.training, 'debug_values', False):
        # Log only every ~256 steps to reduce I/O
        if (self.global_step % 256 == 0) and (self.trainer is not None):
          with torch.no_grad():
            b = min(2, x0.shape[0])
            dbg = {
              't': t[:b].detach().float().cpu().tolist(),
              'move_chance_mean': move_chance[:b].detach().float().mean().item() if torch.is_tensor(move_chance) else float(move_chance),
              'x0_tokens': x0[:b].detach().cpu().tolist(),
              'xt_tokens': xt[:b].detach().cpu().tolist(),
            }
            # Use Lightning logger if available; else print
            if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'log_text'):
              self.trainer.logger.log_text(key='debug/values', columns=['json'], data=[[json.dumps(dbg)]])
            else:
              print('[debug] diffusion', json.dumps(dbg))
    except Exception:
      # Never break training on debug errors
      pass

    
    model_output = self.forward(xt, unet_conditioning)
    if utils.print_nans(model_output, 'model_output'):
      print("[debug forward pass] xt:", xt)
      print("[debug forward pass] unet_conditioning:", unet_conditioning)
      print("[debug forward pass] t:", t)
      print("[debug forward pass] move chance:", move_chance)


    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(
        model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)
      if self.parameterization == 'd3pm':
        reconstruction_loss = self._reconstruction_loss(x0)
      elif self.parameterization == 'subs':
        reconstruction_loss = 0
      return reconstruction_loss + diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]

  def _loss(self, x0, attention_mask, unmask_order=None, flat_order=None):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(input_tokens, unmask_order=unmask_order, flat_order=flat_order)
    
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def sample_subs_guidance(
    self, n_samples, stride_length, num_strides, dt=0.001):
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self._sample_prior(
        n_samples,
        self.config.model.length).to(self.device)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ddpm_caching_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(
        x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]
      == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append(
        self.tokenizer.batch_decode(
          np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples,
            sequence_lengths)

  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    (sampling_steps, samples,
     sequence_lengths) = self.sample_subs_guidance(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,
      num_strides=num_strides, 
      dt=dt)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths


  def save_generated_samples(self):
    try:
      # Prefer saving under the Hydra run's date folder
      output_path = os.path.join(self.config.data.cache_dir, "{}_{}_{}_generated_samples.jsonl".format(self.config.sampling.steps, self.config.model.length, self.config.denoise_scheduler.name))
    except Exception:
      # Fallback to current working directory if Hydra runtime is unavailable
      output_path = os.path.join(os.getcwd(), "{}_{}_{}_generated_samples.jsonl".format(self.config.sampling.steps, self.config.model.length, self.config.denoise_scheduler.name))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
      for sample in self.generated_samples:
        f.write(json.dumps(sample) + "\n")
    return
