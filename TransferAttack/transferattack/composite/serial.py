from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Sequence

import torch

from .. import load_attack_class


class SerialAttack:
    """Wrapper that chains multiple registered attacks sequentially."""

    def __init__(
        self,
        model_name,
        epsilon=16 / 255,
        alpha=1.6 / 255,
        epoch=10,
        decay=1.0,
        targeted=False,
        random_start=False,
        norm=None,
        loss=None,
        sequence: Optional[Sequence[str]] = None,
        attack: str = 'Serial',
        device=None,
        **kwargs,
    ) -> None:
        if not sequence or len(sequence) < 2:
            raise ValueError('SerialAttack expects at least two sub-attacks')
        self.sequence: List[str] = [token.strip() for token in sequence if token.strip()]
        if len(self.sequence) < 2:
            raise ValueError('SerialAttack expects at least two valid sub-attacks')
        self.attack = attack
        self.model_name = model_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.targeted = targeted
        self.random_start = random_start
        self.norm = norm
        self.loss = loss
        self.device = device
        self.model = None
        self.extra_kwargs = deepcopy(kwargs)
        self._base_kwargs: Dict = dict(
            model_name=model_name,
            epsilon=epsilon,
            alpha=alpha,
            epoch=epoch,
            decay=decay,
            targeted=targeted,
            random_start=random_start,
        )
        if norm is not None:
            self._base_kwargs['norm'] = norm
        if loss is not None:
            self._base_kwargs['loss'] = loss
        self._base_kwargs.update(self.extra_kwargs)
        self.sub_attacks: List = []

    def propagate_model_to_sub_attacks(self):
        if self.model is None or self.device is None:
            raise RuntimeError('SerialAttack requires an injected surrogate before execution')
        self.sub_attacks = []
        for name in self.sequence:
            attack_cls = load_attack_class(name)
            init_kwargs = dict(self._base_kwargs)
            init_kwargs['model_name'] = self.model_name
            init_kwargs['device'] = self.device
            sub_attack = attack_cls.__new__(attack_cls)
            sub_attack._preloaded_model = self.model
            attack_cls.__init__(sub_attack, **init_kwargs)
            sub_attack.device = self.device
            self.sub_attacks.append(sub_attack)

    def _ensure_ready(self):
        if not self.sub_attacks:
            self.propagate_model_to_sub_attacks()

    def __call__(self, images: torch.Tensor, labels):
        self._ensure_ready()
        running = images.clone()
        for sub_attack in self.sub_attacks:
            delta = sub_attack(running, labels)
            running = torch.clamp(running + delta.detach().cpu(), 0.0, 1.0)
        return running - images
