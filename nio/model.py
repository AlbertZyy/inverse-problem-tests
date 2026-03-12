from __future__ import annotations
import sys
sys.path.append('nio')

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from deeponet import DeepONet
from fno import FNO2d


class NIO2d(nn.Module):
    """
    Neural Inverse Operator for 2D inverse problems.

    Data flow:
        measurements: [B, L, function_input_dim]
            -> DeepONet lifting
        lifted scalar fields: [B, L, N]
            -> permutation-invariant R aggregation
        aggregated features: [B, d_v, N]
            -> reshape to [B, d_v, H, W]
            -> FNO2d
        prediction: [B, 1, H, W]

    Notes
    -----
    1. This module reuses an existing DeepONet and an existing FNO2d.
    2. The wrapped DeepONet must expose:
           - deeponet.branch_net
           - deeponet.trunk_net
    3. The wrapped FNO2d should already be configured with:
           - in_channels == d_v
           - out_channels == 1
    """

    def __init__(
        self,
        deeponet: DeepONet,
        fno: FNO2d,
        d_v: int,
        geometry_dim: int = 2,
    ) -> None:
        super().__init__()

        if d_v <= 0:
            raise ValueError(f"`d_v` must be positive, got {d_v}.")
        if geometry_dim <= 0:
            raise ValueError(
                f"`geometry_dim` must be positive, got {geometry_dim}."
            )

        self.deeponet = deeponet
        self.fno = fno
        self.d_v = d_v
        self.geometry_dim = geometry_dim

        if not hasattr(self.deeponet, "branch_net"):
            raise AttributeError(
                "The provided DeepONet must expose `branch_net`."
            )
        if not hasattr(self.deeponet, "trunk_net"):
            raise AttributeError(
                "The provided DeepONet must expose `trunk_net`."
            )

        # R aggregation layer:
        #   h(z) = D * mean_l f_l(z) + E z
        #
        # Here:
        #   - D is implemented as a learnable channel-wise scaling vector
        #   - E z is implemented as a linear projection from coordinates to d_v
        self.measurement_scale = nn.Parameter(torch.ones(d_v))
        self.coord_proj = nn.Linear(geometry_dim, d_v, bias=True)

    def _flatten_coords(
        self,
        coords: Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Tuple[int, int]]:
        """
        Normalize coordinates to flattened shape [N, geometry_dim].

        Supported input formats:
            - [N, geometry_dim], requires `grid_size`
            - [H, W, geometry_dim], infers `grid_size=(H, W)`

        Returns
        -------
        coords_flat:
            Tensor of shape [N, geometry_dim]
        grid_size:
            Tuple (H, W)
        """
        if coords.ndim == 2:
            if coords.shape[-1] != self.geometry_dim:
                raise ValueError(
                    f"Expected coords shape [N, {self.geometry_dim}], "
                    f"got {tuple(coords.shape)}."
                )
            if grid_size is None:
                raise ValueError(
                    "`grid_size` must be provided when `coords` has shape "
                    "[N, geometry_dim]."
                )

            height, width = grid_size
            num_points = height * width
            if coords.shape[0] != num_points:
                raise ValueError(
                    "Mismatch between flattened coordinate count and grid_size: "
                    f"coords.shape[0]={coords.shape[0]}, "
                    f"but height * width = {num_points}."
                )

            coords_flat = coords.contiguous()
            return coords_flat, (height, width)

        if coords.ndim == 3:
            height, width, coord_dim = coords.shape
            if coord_dim != self.geometry_dim:
                raise ValueError(
                    f"Expected coords shape [H, W, {self.geometry_dim}], "
                    f"got {tuple(coords.shape)}."
                )

            coords_flat = coords.reshape(height * width, coord_dim).contiguous()

            if grid_size is not None and grid_size != (height, width):
                raise ValueError(
                    f"Inconsistent `grid_size`: got {grid_size}, "
                    f"but coords imply {(height, width)}."
                )

            return coords_flat, (height, width)

        raise ValueError(
            "`coords` must have shape [N, geometry_dim] or [H, W, geometry_dim], "
            f"got {tuple(coords.shape)}."
        )

    def _deeponet_lifting(
        self,
        measurements: Tensor,
        coords_flat: Tensor,
    ) -> Tensor:
        """
        Apply DeepONet lifting from boundary measurements to interior scalar fields.

        Parameters
        ----------
        measurements:
            Tensor of shape [B, L, function_input_dim]
        coords_flat:
            Tensor of shape [N, geometry_dim]

        Returns
        -------
        lifted_fields:
            Tensor of shape [B, L, N]
        """
        if measurements.ndim != 3:
            raise ValueError(
                "`measurements` must have shape [B, L, function_input_dim], "
                f"got {tuple(measurements.shape)}."
            )

        batch_size, num_measurements, function_input_dim = measurements.shape
        del function_input_dim  # shape-only check; not used later

        num_points = coords_flat.shape[0]

        # Branch net over all measurements in the mini-batch:
        #   [B, L, function_input_dim] -> [B * L, function_input_dim]
        #   -> [B * L, num_basis] -> [B, L, num_basis]
        branch_in = measurements.reshape(batch_size * num_measurements, -1)
        branch_out = self.deeponet.branch_net(branch_in)

        if branch_out.ndim != 2:
            raise ValueError(
                "Expected `deeponet.branch_net(...)` to return shape "
                "[B * L, num_basis], "
                f"got {tuple(branch_out.shape)}."
            )

        num_basis = branch_out.shape[-1]
        branch_out = branch_out.reshape(batch_size, num_measurements, num_basis)

        # Trunk net over spatial query points:
        #   [N, geometry_dim] -> [N, num_basis]
        trunk_out = self.deeponet.trunk_net(coords_flat)

        if trunk_out.ndim != 2:
            raise ValueError(
                "Expected `deeponet.trunk_net(...)` to return shape "
                "[N, num_basis], "
                f"got {tuple(trunk_out.shape)}."
            )
        if trunk_out.shape[0] != num_points:
            raise ValueError(
                "Unexpected number of points returned by trunk_net: "
                f"expected {num_points}, got {trunk_out.shape[0]}."
            )
        if trunk_out.shape[-1] != num_basis:
            raise ValueError(
                "Branch/trunk output dimensions must match: "
                f"branch num_basis={num_basis}, "
                f"trunk num_basis={trunk_out.shape[-1]}."
            )

        # DeepONet scalar output at each spatial point:
        #   [B, L, P] x [N, P] -> [B, L, N]
        lifted_fields = torch.einsum("blp,np->bln", branch_out, trunk_out)
        return lifted_fields

    def _aggregate_measurements(
        self,
        lifted_fields: Tensor,
        coords_flat: Tensor,
    ) -> Tensor:
        """
        R aggregation:
            [B, L, N] -> [B, d_v, N]

        The measurement axis is reduced by mean pooling, then expanded into
        the FNO channel dimension with a learnable channel-wise scaling and
        coordinate injection.
        """
        if lifted_fields.ndim != 3:
            raise ValueError(
                "`lifted_fields` must have shape [B, L, N], "
                f"got {tuple(lifted_fields.shape)}."
            )

        # Permutation-invariant reduction over the measurement axis.
        # [B, L, N] -> [B, N]
        scalar_field = lifted_fields.mean(dim=1)

        # Channel lift using learnable D.
        # [B, N] -> [B, d_v, N]
        lifted = scalar_field.unsqueeze(1) * self.measurement_scale.view(
            1, self.d_v, 1
        )

        # Coordinate injection E z.
        # [N, geometry_dim] -> [N, d_v] -> [1, d_v, N]
        coord_features = self.coord_proj(coords_flat).transpose(0, 1).unsqueeze(0)

        aggregated = lifted + coord_features
        return aggregated

    def forward(
        self,
        measurements: Tensor,
        coords: Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        Forward pass of NIO2d.

        Parameters
        ----------
        measurements:
            Boundary observations after randomized batching.
            Shape: [B, L, function_input_dim]

        coords:
            Spatial query coordinates.
            Supported shapes:
                - [N, geometry_dim], together with `grid_size=(H, W)`
                - [H, W, geometry_dim]

        grid_size:
            Required when `coords` is flattened as [N, geometry_dim].

        Returns
        -------
        pred:
            Reconstructed coefficient field with shape [B, 1, H, W]
        """
        coords_flat, (height, width) = self._flatten_coords(coords, grid_size)

        lifted_fields = self._deeponet_lifting(
            measurements=measurements,
            coords_flat=coords_flat,
        )  # [B, L, N]

        aggregated = self._aggregate_measurements(
            lifted_fields=lifted_fields,
            coords_flat=coords_flat,
        )  # [B, d_v, N]

        batch_size = measurements.shape[0]
        num_points = height * width
        if aggregated.shape[-1] != num_points:
            raise RuntimeError(
                "Aggregated feature size does not match grid size: "
                f"expected last dim {num_points}, got {aggregated.shape[-1]}."
            )

        fno_in = aggregated.reshape(batch_size, self.d_v, height, width)
        pred = self.fno(fno_in)

        if pred.ndim != 4:
            raise ValueError(
                "Expected `fno(...)` to return shape [B, C, H, W], "
                f"got {tuple(pred.shape)}."
            )
        if pred.shape[0] != batch_size:
            raise ValueError(
                f"Unexpected batch size in FNO output: expected {batch_size}, "
                f"got {pred.shape[0]}."
            )
        if pred.shape[2:] != (height, width):
            raise ValueError(
                "Unexpected spatial size in FNO output: "
                f"expected {(height, width)}, got {tuple(pred.shape[2:])}."
            )

        return pred


if __name__ == "__main__":
    deeponet = DeepONet(252, 2, 20, 20, 4, 20, 4)   # 已有实现
    fno = FNO2d(32, 1, 8, 8, 64, 64)           # 已有实现，要求 in_channels=d_v, out_channels=1

    model = NIO2d(
        deeponet=deeponet,
        fno=fno,
        d_v=32,
        geometry_dim=2,
    )

    measurements = torch.randn(8, 12, 252)      # [B, L, function_input_dim]
    coords = torch.randn(64, 64, 2)            # [H, W, 2]

    pred = model(measurements, coords)         # [8, 1, 64, 64]
    print(pred.shape)  # should be [8, 1, 64, 64]
