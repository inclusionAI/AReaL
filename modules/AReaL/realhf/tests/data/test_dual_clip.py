import pytest
import torch

from realhf.impl.model.utils.ppo_functional import actor_loss_fn


# Copied from https://github.com/opendilab/PPOxFamily/blob/main/chapter7_tricks/dual_clip.py
def ppo_dual_clip(
    logp_new: torch.FloatTensor,
    logp_old: torch.FloatTensor,
    adv: torch.FloatTensor,
    clip_ratio: float,
    dual_clip: float,
) -> torch.FloatTensor:
    """
    **Overview**:
        This function implements the Proximal Policy Optimization (PPO) policy loss with dual-clip
        mechanism, which is a variant of PPO that provides more reliable and stable training by
        limiting the updates to the policy, preventing it from deviating too much from its previous versions.
    Arguments:
        - logp_new (:obj:`torch.FloatTensor`): The log probability calculated by the new policy.
        - logp_old (:obj:`torch.FloatTensor`): The log probability calculated by the old policy.
        - adv (:obj:`torch.FloatTensor`): The advantage value, which measures how much better an
            action is compared to the average action at that state.
        - clip_ratio (:obj:`float`): The clipping ratio used to limit the change of policy during an update.
        - dual_clip (:obj:`float`): The dual clipping ratio used to further limit the change of policy during an update.
    Returns:
        - policy_loss (:obj:`torch.FloatTensor`): The calculated policy loss, which is the objective we
            want to minimize for improving the policy.
    """
    assert (
        dual_clip is None or dual_clip > 1.0
    ), "Dual_clip value must be greater than 1.0, but get value: {}".format(dual_clip)
    # This is the ratio of the new policy probability to the old policy probability.
    # $$r(\theta) = \frac{\pi_{new}(a|s)}{\pi_{old}(a|s)}$$
    ratio = torch.exp(logp_new - logp_old)
    # The first clipping operation is performed here, we limit the update to be within a certain range.
    # $$clip_1 = min(r(\theta)*A(s,a), clip(r(\theta), 1-clip\_ratio, 1+clip\_ratio)*A(s,a))$$
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    clip1 = torch.min(surr1, surr2)
    # The second clipping operation is performed here, we further limit the update to be within a stricter range.
    # $$clip_2 = max(clip_1, dual\_clip * A(s,a))$$
    if dual_clip is not None:
        clip2 = torch.max(clip1, dual_clip * adv)
        # We only apply the dual-clip when the advantage is negative, i.e., when the action is worse than the average.
        policy_loss = -(torch.where(adv < 0, clip2, clip1)).mean()
    else:
        policy_loss = -clip1.mean()
    return policy_loss


@pytest.mark.parametrize("eps_clip", [0.01, 0.05, 0.1, 0.2, 0.5])
@pytest.mark.parametrize("c_clip", [None, 1.5, 2.0, 5.0])
@pytest.mark.parametrize("size", [(1,), (10,), (100,)])
def test_dual_clip_acc(size, eps_clip, c_clip):
    old_logp = -torch.randn(size, dtype=torch.float32).abs()
    new_logp = -torch.randn(size, dtype=torch.float32).abs()
    adv = torch.randn(size, dtype=torch.float32)

    loss1 = ppo_dual_clip(new_logp, old_logp, adv, eps_clip, c_clip)
    loss2, _ = actor_loss_fn(new_logp, old_logp, adv, eps_clip, c_clip=c_clip)
    assert torch.allclose(loss1, loss2)
