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
    """PPO with leave-one-out counterfactual credit applied as reward shaping.

    Theory (single-permutation Monte Carlo Shapley estimator):
        For each agent i, the marginal contribution to team value is approximated as:
            phi_i(s) = V(s) - V(s \\ i)
        where V(s\\i) is the team value with agent i masked out.

        Instead of adding this as a constant offset to advantages (which has zero
        gradient effect under importance sampling), we shape the per-agent reward:
            r_i_shaped = r_i + alpha * normalize(phi_i)
        and let standard GAE propagate it through advantage estimation.

    This places counterfactual credit assignment in the reward channel, where it
    actually affects policy gradients, while keeping the formulation as a
    1-permutation Monte Carlo estimator of Shapley value.
    """

    @override(PPOTorchPolicy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        """Inject counterfactual credit into rewards before standard GAE."""
        cf_coef = float(self.config.get("counterfactual_advantage_coef", 0.0))
        cf_clip = float(self.config.get("counterfactual_credit_clip", 0.0))

        if (
            cf_coef > 0.0
            and self.model is not None
            and hasattr(self.model, "compute_counterfactual_values")
            and SampleBatch.AGENT_INDEX in sample_batch
            and SampleBatch.CUR_OBS in sample_batch
            and len(sample_batch[SampleBatch.REWARDS]) > 0
        ):
            try:
                with torch.no_grad():
                    obs_t = torch.as_tensor(
                        sample_batch[SampleBatch.CUR_OBS],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    agent_idx_t = torch.as_tensor(
                        sample_batch[SampleBatch.AGENT_INDEX],
                        dtype=torch.long,
                        device=self.device,
                    )
                    # Forward pass to populate value_function
                    _ = self.model({"obs_flat": obs_t, "obs": obs_t}, [], None)
                    v_full = self.model.value_function().detach()
                    v_loo = self.model.compute_counterfactual_values(obs_t, agent_idx_t).detach()
                    cf_credit = (v_full - v_loo).cpu().numpy()

                # Normalize per-batch (z-score) to keep magnitude stable
                if cf_credit.size > 0:
                    cf_mean = float(cf_credit.mean())
                    cf_std = float(cf_credit.std() + 1e-6)
                    cf_norm = (cf_credit - cf_mean) / cf_std
                    if cf_clip > 0.0:
                        import numpy as _np
                        cf_norm = _np.clip(cf_norm, -cf_clip, cf_clip)
                    # Shape rewards: r_shaped = r + alpha * normalized_credit
                    sample_batch[SampleBatch.REWARDS] = (
                        sample_batch[SampleBatch.REWARDS] + cf_coef * cf_norm
                    ).astype(sample_batch[SampleBatch.REWARDS].dtype)
            except Exception as e:
                # If anything fails, fall through to standard GAE without shaping
                pass

        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)

    @override(PPOTorchPolicy)
    def loss(
        self,
        model: ModelV2,
        dist_class: type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> TensorType:
        # Standard PPO loss — counterfactual credit is now in rewards (via postprocess)
        # so we just call the parent loss without modification.
        loss = super().loss(model, dist_class, train_batch)

        # Track CF stats for logging
        try:
            with torch.no_grad():
                if (
                    hasattr(model, "compute_counterfactual_values")
                    and SampleBatch.AGENT_INDEX in train_batch
                ):
                    v_full = model.value_function().detach()
                    v_loo = model.compute_counterfactual_values(
                        train_batch[SampleBatch.CUR_OBS],
                        train_batch[SampleBatch.AGENT_INDEX],
                    ).detach()
                    cf_credit = v_full - v_loo
                    model.tower_stats["counterfactual_credit_mean"] = cf_credit.mean()
                    model.tower_stats["counterfactual_credit_std"] = cf_credit.std()
                    model.tower_stats["counterfactual_credit_abs_mean"] = cf_credit.abs().mean()
        except Exception:
            pass

        return loss

    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch):
        stats = super().stats_fn(train_batch)
        extra_keys = [
            "counterfactual_credit_mean",
            "counterfactual_credit_std",
            "counterfactual_credit_abs_mean",
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
