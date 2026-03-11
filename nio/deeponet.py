from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import torch
from torch import Tensor, nn


ActivationFactory: TypeAlias = Callable[[], nn.Module]


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
    activation_factory: ActivationFactory,
) -> nn.Sequential:
    """
    Build a standard MLP of the form:

        Linear(input_dim -> hidden_dim)
        + [Activation + Linear(hidden_dim -> hidden_dim)] * (num_hidden_layers - 1)
        + Activation
        + Linear(hidden_dim -> output_dim)

    When num_hidden_layers == 0, the network degenerates to a single linear layer:

        Linear(input_dim -> output_dim)

    Parameters
    ----------
    input_dim:
        Input feature dimension.
    output_dim:
        Output feature dimension.
    hidden_dim:
        Width of hidden layers.
    num_hidden_layers:
        Number of hidden linear layers.
    activation_factory:
        A callable that returns a fresh activation module instance each time,
        e.g. ``lambda: nn.ReLU()`` or ``nn.GELU``.

    Returns
    -------
    nn.Sequential
        The constructed MLP module.
    """
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, but got {input_dim}.")
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive, but got {output_dim}.")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, but got {hidden_dim}.")
    if num_hidden_layers < 0:
        raise ValueError(
            f"num_hidden_layers must be non-negative, but got {num_hidden_layers}."
        )

    layers: list[nn.Module] = []

    if num_hidden_layers == 0:
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    layers.append(nn.Linear(input_dim, hidden_dim))
    for _ in range(num_hidden_layers - 1):
        layers.append(activation_factory())
        layers.append(nn.Linear(hidden_dim, hidden_dim))

    layers.append(activation_factory())
    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)


class BranchNet(nn.Module):
    """
    Branch network in DeepONet.

    The branch network maps a discretized input function to its coefficient vector:

        u  ->  c(u)

    Shape convention
    ----------------
    Input:
        function_values: [batch_size, function_input_dim]

    Output:
        branch_coefficients: [batch_size, num_basis]
    """

    def __init__(
        self,
        function_input_dim: int,
        num_basis: int,
        branch_hidden_dim: int,
        branch_num_hidden_layers: int,
        branch_activation: ActivationFactory = nn.GELU,
    ) -> None:
        super().__init__()
        self.function_input_dim = function_input_dim
        self.num_basis = num_basis
        self.branch_hidden_dim = branch_hidden_dim
        self.branch_num_hidden_layers = branch_num_hidden_layers

        self.network = _build_mlp(
            input_dim=function_input_dim,
            output_dim=num_basis,
            hidden_dim=branch_hidden_dim,
            num_hidden_layers=branch_num_hidden_layers,
            activation_factory=branch_activation,
        )

    def forward(self, function_values: Tensor) -> Tensor:
        """
        Parameters
        ----------
        function_values:
            Tensor with shape [batch_size, function_input_dim].

        Returns
        -------
        Tensor
            Branch coefficients with shape [batch_size, num_basis].
        """
        if function_values.ndim != 2:
            raise ValueError(
                "function_values must be a 2D tensor of shape "
                "[batch_size, function_input_dim], "
                f"but got shape {tuple(function_values.shape)}."
            )

        if function_values.shape[-1] != self.function_input_dim:
            raise ValueError(
                "The last dimension of function_values must match function_input_dim. "
                f"Expected {self.function_input_dim}, "
                f"but got {function_values.shape[-1]}."
            )

        branch_coefficients = self.network(function_values)
        return branch_coefficients


class TrunkNet(nn.Module):
    """
    Trunk network in DeepONet.

    The trunk network maps evaluation coordinates to basis values:

        y  ->  [phi_1(y), phi_2(y), ..., phi_p(y)]

    Shape convention
    ----------------
    Input:
        eval_points: [num_points, geometry_dim]

    Output:
        trunk_basis_values: [num_points, num_basis]
    """

    def __init__(
        self,
        geometry_dim: int,
        num_basis: int,
        trunk_hidden_dim: int,
        trunk_num_hidden_layers: int,
        trunk_activation: ActivationFactory = nn.GELU,
    ) -> None:
        super().__init__()
        self.geometry_dim = geometry_dim
        self.num_basis = num_basis
        self.trunk_hidden_dim = trunk_hidden_dim
        self.trunk_num_hidden_layers = trunk_num_hidden_layers

        self.network = _build_mlp(
            input_dim=geometry_dim,
            output_dim=num_basis,
            hidden_dim=trunk_hidden_dim,
            num_hidden_layers=trunk_num_hidden_layers,
            activation_factory=trunk_activation,
        )

    def forward(self, eval_points: Tensor) -> Tensor:
        """
        Parameters
        ----------
        eval_points:
            Tensor with shape [num_points, geometry_dim].

        Returns
        -------
        Tensor
            Trunk basis values with shape [num_points, num_basis].
        """
        if eval_points.ndim != 2:
            raise ValueError(
                "eval_points must be a 2D tensor of shape "
                "[num_points, geometry_dim], "
                f"but got shape {tuple(eval_points.shape)}."
            )

        if eval_points.shape[-1] != self.geometry_dim:
            raise ValueError(
                "The last dimension of eval_points must match geometry_dim. "
                f"Expected {self.geometry_dim}, "
                f"but got {eval_points.shape[-1]}."
            )

        trunk_basis_values = self.network(eval_points)
        return trunk_basis_values


class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet).

    DeepONet approximates an operator of the form

        u -> G(u)(y)

    by decomposing the output into a linear combination of learned basis functions:

        G(u)(y) ~= sum_{k=1}^{num_basis} c_k(u) * phi_k(y)

    where:
    - the branch network produces the coefficients c_k(u),
    - the trunk network produces the basis values phi_k(y).

    Shape convention
    ----------------
    Inputs
    ------
    function_values:
        [batch_size, function_input_dim]

    eval_points:
        [num_points, geometry_dim]

    Outputs
    -------
    operator_values:
        [batch_size, num_points]
    """

    def __init__(
        self,
        function_input_dim: int,
        geometry_dim: int,
        num_basis: int,
        branch_hidden_dim: int,
        branch_num_hidden_layers: int,
        trunk_hidden_dim: int,
        trunk_num_hidden_layers: int,
        branch_activation: ActivationFactory = nn.GELU,
        trunk_activation: ActivationFactory = nn.GELU,
        use_output_bias: bool = True,
    ) -> None:
        super().__init__()
        self.function_input_dim = function_input_dim
        self.geometry_dim = geometry_dim
        self.num_basis = num_basis
        self.use_output_bias = use_output_bias

        self.branch_net = BranchNet(
            function_input_dim=function_input_dim,
            num_basis=num_basis,
            branch_hidden_dim=branch_hidden_dim,
            branch_num_hidden_layers=branch_num_hidden_layers,
            branch_activation=branch_activation,
        )

        self.trunk_net = TrunkNet(
            geometry_dim=geometry_dim,
            num_basis=num_basis,
            trunk_hidden_dim=trunk_hidden_dim,
            trunk_num_hidden_layers=trunk_num_hidden_layers,
            trunk_activation=trunk_activation,
        )

        if use_output_bias:
            self.output_bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("output_bias", None)

    def forward(self, function_values: Tensor, eval_points: Tensor) -> Tensor:
        """
        Evaluate the learned operator on a batch of input functions and query points.

        Parameters
        ----------
        function_values:
            Tensor with shape [batch_size, function_input_dim].
        eval_points:
            Tensor with shape [num_points, geometry_dim].

        Returns
        -------
        Tensor
            Operator values with shape [batch_size, num_points].
        """
        branch_coefficients = self.branch_net(function_values)
        trunk_basis_values = self.trunk_net(eval_points)

        operator_values = torch.einsum(
            "bk,pk->bp",
            branch_coefficients,
            trunk_basis_values,
        )

        if self.output_bias is not None:
            operator_values = operator_values + self.output_bias

        return operator_values

    def forward_branch(self, function_values: Tensor) -> Tensor:
        """
        Compute branch coefficients only.

        Parameters
        ----------
        function_values:
            Tensor with shape [batch_size, function_input_dim].

        Returns
        -------
        Tensor
            Branch coefficients with shape [batch_size, num_basis].
        """
        return self.branch_net(function_values)

    def forward_trunk(self, eval_points: Tensor) -> Tensor:
        """
        Compute trunk basis values only.

        Parameters
        ----------
        eval_points:
            Tensor with shape [num_points, geometry_dim].

        Returns
        -------
        Tensor
            Trunk basis values with shape [num_points, num_basis].
        """
        return self.trunk_net(eval_points)

    def extra_repr(self) -> str:
        return (
            f"function_input_dim={self.function_input_dim}, "
            f"geometry_dim={self.geometry_dim}, "
            f"num_basis={self.num_basis}, "
            f"use_output_bias={self.use_output_bias}"
        )
