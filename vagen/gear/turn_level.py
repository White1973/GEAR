"""Utilities for placing process rewards on their originating turns."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


def redistribute_turn_process_rewards(
    reward_tensor: torch.Tensor,
    reward_extra_infos: Mapping[str, Sequence[Any]],
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Move process reward mass from the trajectory terminal to each turn end.

    The batch reward manager initially places the combined scalar score at the
    final response token.  GEAR returns the exact scaled process components and
    their exclusive response offsets; this function preserves the total return
    while relocating only those components.
    """
    positions_batch = reward_extra_infos.get("turn_process_reward_positions")
    rewards_batch = reward_extra_infos.get("turn_process_rewards")
    if positions_batch is None or rewards_batch is None:
        return reward_tensor

    adjusted = reward_tensor.clone()
    batch_size, response_length = adjusted.shape
    for row in range(min(batch_size, len(positions_batch), len(rewards_batch))):
        positions = list(positions_batch[row] or [])
        rewards = [float(value) for value in (rewards_batch[row] or [])]
        valid_pairs = [
            (int(end) - 1, value)
            for end, value in zip(positions, rewards)
            if int(end) > 0 and int(end) <= response_length
        ]
        if not valid_pairs:
            continue

        generated_indices = torch.nonzero(response_mask[row].bool(), as_tuple=False).flatten()
        if generated_indices.numel() == 0:
            continue
        terminal = int(generated_indices[-1].item())
        process_total = sum(value for _, value in valid_pairs)
        adjusted[row, terminal] -= process_total
        for token_idx, value in valid_pairs:
            # A recorded turn end must point to a generated token, never an
            # environment-observation suffix. Ignore malformed metadata safely.
            if bool(response_mask[row, token_idx].item()):
                adjusted[row, token_idx] += value
            else:
                adjusted[row, terminal] += value
    return adjusted
