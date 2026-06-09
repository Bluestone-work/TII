import torch

from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
try:
    from ray.rllib.algorithms import registry as rllib_registry
    from ray.rllib.algorithms.registry import POLICIES
except Exception:  # pragma: no cover
    rllib_registry = None
    POLICIES = None
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import ActionDistribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.typing import TensorType


class CounterfactualPPOTorchPolicy(PPOTorchPolicy):
    """PPO with leave-one-out counterfactual credit mixed into advantages."""

    @override(PPOTorchPolicy)
    def loss(
        self,
        model: ModelV2,
        dist_class: type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> TensorType:
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        if state:
            batch_size = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // batch_size
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS],
            model,
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        advantages = train_batch[Postprocessing.ADVANTAGES]
        cf_bonus = torch.zeros_like(advantages)
        cf_credit = torch.zeros_like(advantages)
        cf_credit_mean = torch.tensor(0.0, device=advantages.device)
        cf_credit_std = torch.tensor(0.0, device=advantages.device)

        cf_coef = float(self.config.get("counterfactual_advantage_coef", 0.0))
        cf_clip = float(self.config.get("counterfactual_credit_clip", 0.0))
        if (
            cf_coef > 0.0
            and hasattr(model, "compute_counterfactual_values")
            and SampleBatch.AGENT_INDEX in train_batch
        ):
            with torch.no_grad():
                cf_values = model.compute_counterfactual_values(
                    train_batch[SampleBatch.CUR_OBS],
                    train_batch[SampleBatch.AGENT_INDEX],
                )
                cf_credit = model.value_function().detach() - cf_values.detach()

                valid_credit = cf_credit[mask] if mask is not None else cf_credit
                if valid_credit.numel() > 0:
                    cf_credit_mean = valid_credit.mean()
                    cf_credit_std = valid_credit.std(unbiased=False)
                    cf_credit_std = torch.clamp(cf_credit_std, min=1e-6)
                    cf_bonus = (cf_credit - cf_credit_mean) / cf_credit_std
                    if cf_clip > 0.0:
                        cf_bonus = torch.clamp(cf_bonus, -cf_clip, cf_clip)
                    cf_bonus = cf_coef * cf_bonus
                else:
                    cf_bonus = torch.zeros_like(cf_credit)

        mixed_advantages = advantages + cf_bonus

        surrogate_loss = torch.min(
            mixed_advantages * logp_ratio,
            mixed_advantages
            * torch.clamp(
                logp_ratio,
                1 - self.config["clip_param"],
                1 + self.config["clip_param"],
            ),
        )

        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS],
                2.0,
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        else:
            value_fn_out = torch.tensor(0.0, device=surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(
                0.0, device=surrogate_loss.device
            )

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            value_fn_out,
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
        model.tower_stats["counterfactual_credit_mean"] = cf_credit_mean
        model.tower_stats["counterfactual_credit_std"] = cf_credit_std
        model.tower_stats["counterfactual_bonus_mean"] = reduce_mean_valid(cf_bonus)
        model.tower_stats["counterfactual_bonus_abs_mean"] = reduce_mean_valid(
            torch.abs(cf_bonus)
        )
        model.tower_stats["counterfactual_mixed_adv_mean"] = reduce_mean_valid(
            mixed_advantages
        )

        return total_loss

    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch):
        stats = super().stats_fn(train_batch)
        extra_keys = [
            "counterfactual_credit_mean",
            "counterfactual_credit_std",
            "counterfactual_bonus_mean",
            "counterfactual_bonus_abs_mean",
            "counterfactual_mixed_adv_mean",
        ]
        for key in extra_keys:
            tower_vals = self.get_tower_stats(key)
            if tower_vals:
                stats[key] = convert_to_numpy(torch.mean(torch.stack(tower_vals)))
        return stats


POLICY_REGISTRY_NAME = CounterfactualPPOTorchPolicy.__name__


def register_counterfactual_policy() -> str:
    """Register the custom policy with RLlib so checkpoints use a durable name."""
    if POLICIES is not None and POLICY_REGISTRY_NAME not in POLICIES:
        # Keep a visible marker in the registry table for debugging/introspection.
        # Actual reverse lookup is patched below because RLlib's default
        # get_policy_class() only supports modules under ray.rllib.algorithms.
        POLICIES[POLICY_REGISTRY_NAME] = "custom.counterfactual_ppo_policy"

    if rllib_registry is not None:
        original_get_name = getattr(rllib_registry, "_cf_original_get_policy_class_name", None)
        if original_get_name is None:
            original_get_name = rllib_registry.get_policy_class_name
            rllib_registry._cf_original_get_policy_class_name = original_get_name

            def _patched_get_policy_class_name(policy_class):
                if policy_class is CounterfactualPPOTorchPolicy:
                    return POLICY_REGISTRY_NAME
                return original_get_name(policy_class)

            rllib_registry.get_policy_class_name = _patched_get_policy_class_name

        original_get_class = getattr(rllib_registry, "_cf_original_get_policy_class", None)
        if original_get_class is None:
            original_get_class = rllib_registry.get_policy_class
            rllib_registry._cf_original_get_policy_class = original_get_class

            def _patched_get_policy_class(name: str):
                if name == POLICY_REGISTRY_NAME:
                    return CounterfactualPPOTorchPolicy
                return original_get_class(name)

            rllib_registry.get_policy_class = _patched_get_policy_class

    return POLICY_REGISTRY_NAME


register_counterfactual_policy()
