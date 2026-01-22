"""
Certified Unlearning for Neural Networks
Based on Koloskova et al., ICML 2025

This code extracts the core unlearning logic from:
https://github.com/stair-lab/certified-unlearning-neural-networks-icml-2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Optional, Tuple
from Models.mlp import MLP



def compute_noise_variance(
        epsilon: float,
        delta: float,
        num_iterations: int,
        learning_rate: float,
        clip_norm_0: float,
        clip_norm_1: float
) -> float:
    """
    Compute the required noise variance for (ε, δ)-unlearning guarantee.

    Based on Theorem 1 in the paper for gradient clipping.

    Args:
        epsilon: Privacy parameter ε
        delta: Privacy parameter δ
        num_iterations: Number of fine-tuning iterations t
        learning_rate: Learning rate γ
        clip_norm_0: Clipping threshold C₀
        clip_norm_1: Clipping threshold C₁

    Returns:
        Required noise variance σ²
    """
    t = num_iterations
    gamma = learning_rate

    # Formula from paper: σ² = (9 log(1/δ))/(ε²t) * (C₀ + C₁γT)²
    numerator = 9 * np.log(1 / delta)
    denominator = (epsilon ** 2) * t
    clip_term = (clip_norm_0 + clip_norm_1 * gamma * t) ** 2

    sigma_squared = (numerator / denominator) * clip_term
    return sigma_squared


class CertifiedUnlearning:
    """
    Implements certified unlearning via Privacy Amplification by Iteration (PABI).

    This corresponds to the "Gradient Clipping" method in the paper.
    """

    def __init__(
            self,
            model_state_dict: dict,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize certified unlearning.

        Args:
            model_state_dict: State dict of pre-trained PyTorch model
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = MLP(input_dim=784, output_dim=10).to(device)
        self.model.load_state_dict(model_state_dict)

    @staticmethod
    def clip_gradient(
            model: nn.Module,
            clip_norm: float
    ) -> None:
        """
        Clip gradients by their L2 norm.

        Args:
            model: Model whose gradients to clip
            clip_norm: Maximum allowed gradient norm
        """
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

    @staticmethod
    def add_gradient_noise(
            model: nn.Module,
            noise_variance: float
    ) -> None:
        """
        Add Gaussian noise to gradients for privacy.

        Args:
            model: Model whose gradients to perturb
            noise_variance: Variance of Gaussian noise
        """
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * np.sqrt(noise_variance)
                param.grad += noise

    def unlearn(
            self,
            retain_loader: DataLoader,
            epsilon: float,
            delta: float,
            num_iterations: int,
            learning_rate: float = 0.001,
            clip_norm_0: float = 1.0,
            clip_norm_1: float = 1.0,
            loss_fn: Optional[nn.Module] = None,
            verbose: bool = True
    ) -> nn.Module:
        """
        Perform certified unlearning via noisy fine-tuning on retain set.

        Args:
            retain_loader: DataLoader for data to retain (D_r)
            epsilon: Privacy parameter ε
            delta: Privacy parameter δ
            num_iterations: Number of fine-tuning iterations T
            learning_rate: Learning rate γ
            clip_norm_0: Clipping threshold C₀
            clip_norm_1: Clipping threshold C₁
            loss_fn: Loss function (default: CrossEntropyLoss)
            verbose: Print progress

        Returns:
            Unlearned model with (ε, δ)-unlearning guarantee
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        # Compute required noise variance
        sigma_squared = compute_noise_variance(
            epsilon, delta, num_iterations, learning_rate,
            clip_norm_0, clip_norm_1
        )

        if verbose:
            print(f"Starting certified unlearning (noisy fine-tuning phase):")
            print(f"  Privacy: (ε={epsilon}, δ={delta})")
            print(f"  Iterations: {num_iterations}")
            print(f"  Noise variance: {sigma_squared:.6f}")

        # Setup optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        self.model.train()
        iteration = 0

        # Phase 1: Noisy fine-tuning on retain set with gradient clipping and noise
        while iteration < num_iterations:
            for batch_idx, (data, target) in enumerate(retain_loader):
                data = data.view(data.size(0), -1)
                data = data.to(self.device)  # Move batch to GPU
                target = target.to(self.device)  # Move labels to GPU
                if iteration >= num_iterations:
                    break


                optimizer.zero_grad()

                # Forward pass
                output = self.model(data)
                loss = loss_fn(output, target)

                # Backward pass
                loss.backward()

                # Clip gradients
                self.clip_gradient(self.model, clip_norm_1)

                # Add noise for privacy
                self.add_gradient_noise(self.model, sigma_squared)

                # Update parameters
                optimizer.step()

                iteration += 1

                if verbose and iteration % 100 == 0:
                    print(f"  Iteration {iteration}/{num_iterations}, Loss: {loss.item():.4f}")

        if verbose:
            print("Noisy fine-tuning complete! Starting post-unlearning fine-tuning...")

        return self.model

    def post_unlearn_finetune(
            self,
            retain_loader: DataLoader,
            num_epochs: int,
            learning_rate: float = 0.001,
            loss_fn: Optional[nn.Module] = None,
            weight_decay: float = 0.0,
            verbose: bool = True
    ) -> nn.Module:
        """
        Post-unlearning fine-tuning phase WITHOUT noise or clipping.

        This phase recovers model accuracy after the noisy unlearning phase.
        Standard SGD training on retain set.

        Args:
            retain_loader: DataLoader for data to retain (D_r)
            num_epochs: Number of fine-tuning epochs
            learning_rate: Learning rate
            loss_fn: Loss function (default: CrossEntropyLoss)
            weight_decay: L2 regularization weight decay
            verbose: Print progress

        Returns:
            Fine-tuned model
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        if verbose:
            print(f"Starting post-unlearning fine-tuning phase:")
            print(f"  Epochs: {num_epochs}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Weight decay: {weight_decay}")

        # Setup optimizer for post-unlearning phase (fresh optimizer without noise)
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.model.train()

        # Phase 2: Standard fine-tuning without noise or clipping
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(retain_loader):
                data = data.view(data.size(0), -1)
                data = data.to(self.device)  # Move batch to GPU
                target = target.to(self.device)  # Move labels to GPU


                optimizer.zero_grad()

                # Forward pass
                output = self.model(data)
                loss = loss_fn(output, target)

                # Backward pass
                loss.backward()

                # Standard optimizer step (NO clipping, NO noise)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            if verbose:
                print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if verbose:
            print("Post-unlearning fine-tuning complete!")

        return self.model