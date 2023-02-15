from typing import Callable, Dict

import torch

from tabsplanation.explanations.latent_shift import grad, LatentShift
from tabsplanation.explanations.losses import AwayLoss, BinaryStretchLoss, ValidityLoss
from tabsplanation.types import B, D, H, Tensor


class LazyRevise(LatentShift):
    """Revise but with fewer gradient computations."""

    def __init__(
        self,
        classifier,
        autoencoder,
        hparams: Dict,
        validity_loss: ValidityLoss = BinaryStretchLoss,
    ):
        """
        gradient_frequency: int (>=1)
            Says how many steps are taken before the gradient is computed again.
            Setting to 1 is equivalent to Revise.
            Setting to inf is equivalent to Latent Shift.
        """
        super(LazyRevise, self).__init__(
            classifier, autoencoder, hparams, validity_loss
        )

        self._gradient_frequency = hparams["gradient_frequency"]

    def get_cf_latents(
        self,
        input: Tensor[B, D],
        target_class: Tensor[B, 1],
    ) -> Tensor["nb_shifts", B, H]:

        ae = self.autoencoder
        clf = self.classifier

        if isinstance(target_class, int) and isinstance(input, torch.Tensor):
            target_class = torch.full((len(input),), target_class).to(input.device)

        # Switch AE to evaluation mode (influences model behaviour in some
        # cases, e.g. `BatchNorm` doesn't compute the mean and standard deviation
        # which means the model can be applied to just one point)
        ae.eval()

        # This is the function of which we take the gradient.
        # The gradient is the direction of steepest ascent.
        # We expect that when we go in the direction of the gradient,
        # the probability of the current class decreases.
        if target_class is None:
            validity_loss_fn = AwayLoss()
        else:
            validity_loss_fn = self.validity_loss()

        def clf_decode(z: Tensor[B, H], target_class):
            logits = clf(ae.decode(z))
            source_class = logits.argmax(dim=-1)
            return validity_loss_fn(logits, source_class, target_class)

        latents: Tensor[1, B, H] = (
            ae.encode(input).reshape(1, input.shape[0], -1).detach()
        )

        if self._gradient_frequency is None:
            z_perturbed: Tensor[self._max_iter - 1, B, H] = self.shift_latents(
                clf_decode, latents[-1, :, :], target_class
            )

            latents: Tensor[self._max_iter, B, H] = torch.concat([latents, z_perturbed])

            return latents

        while len(latents) + self._gradient_frequency < self._max_iter:

            z_perturbed: Tensor[self._gradient_frequency, B, H] = self.shift_latents(
                clf_decode, latents[-1, :, :], target_class
            )

            latents: Tensor["", B, H] = torch.concat([latents, z_perturbed])

        return latents

    def shift_latents(
        self,
        clf_decode: Callable[[Tensor[B, H], Tensor[B, 1]], Tensor[B, H]],
        latents: Tensor[B, H],
        target_class: Tensor[B, 1],
    ) -> Tensor["gradient_frequency", B, H]:

        # source_class = classifier.predict(autoencoder.decode(latents)).detach()

        latents.requires_grad = True
        with torch.enable_grad():
            gradient: Tensor[B, H] = grad(clf_decode(latents, target_class), latents)

        # 2) Apply latent shift

        nb_shifts = (
            self._gradient_frequency
            if self._gradient_frequency is not None
            else self._max_iter
        )
        shifts: Tensor["gradient_frequency", 1, 1] = (
            torch.tensor([i * self._shift_step for i in range(1, 1 + nb_shifts)])
            .reshape(-1, 1, 1)
            .to(latents.device)
        )

        # Multiply gradient by -1 to minimize the loss
        z_perturbed: Tensor["gradient_frequency", B, H] = (
            (latents - shifts * gradient).to(torch.float).detach()
        )

        return z_perturbed

    def validity_rate(
        self,
        input: Tensor[B, D],
        target_class: Tensor[B, 1],
    ) -> Tensor["nb_shifts", B, H]:
        latents: Tensor["nb_shifts", B, H] = self.get_cf_latents(input, target_class)
        nb_shifts, batch, latent_dim = latents.shape
        cf_paths = self.autoencoder.decode(latents.reshape(-1, latent_dim)).reshape(
            nb_shifts, batch, input.shape[-1]
        )
        preds = self.classifier.predict(cf_paths)
        validity_rate = torch.any(preds == target_class, dim=0).mean(dtype=torch.float)
        return validity_rate
