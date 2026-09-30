import torch


def _layout_masks(layout, *, device=None):
    """Build compact masks from an encoder-provided latent layout."""
    context, target = layout.masks(device=device)
    return context, target


class CausalPatchMask:
    def __init__(
        self,
        image_size=None,
        patch_size=None,
        tubelet_size=None,
        video_frames=None,
        context_tubelets=1,
        num_views=1,
        layout=None,
    ):
        if layout is not None:
            if any(value is not None for value in (image_size, patch_size, tubelet_size, video_frames)):
                raise ValueError("layout cannot be combined with image/temporal geometry arguments")
            self.ctx_idx, self.tgt_idx = _layout_masks(layout)
            self.n_ctx = self.ctx_idx.numel()
            self.grid_depth = layout.grid_depth
            self.grid_height = layout.grid_height
            self.grid_width = layout.grid_width
            self.num_views = layout.num_views
            self.n_target = self.tgt_idx.numel()
            return
        if image_size is None or patch_size is None or tubelet_size is None or video_frames is None:
            raise ValueError("image_size, patch_size, tubelet_size, and video_frames are required")
        height, width = image_size
        if patch_size <= 0 or height % patch_size or width % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if tubelet_size <= 0 or video_frames % tubelet_size:
            raise ValueError("video_frames must be divisible by tubelet_size")
        grid_depth = video_frames // tubelet_size
        if num_views <= 0:
            raise ValueError(f"num_views must be positive, got {num_views}")
        grid_height = height // patch_size
        grid_width = width // patch_size
        n_ctx = context_tubelets * num_views * grid_height * grid_width
        n_tot = grid_depth * num_views * grid_height * grid_width
        if not 0 < n_ctx < n_tot:
            raise ValueError(
                f"context_tubelets={context_tubelets} invalid for grid_depth={grid_depth}"
            )
        token_ids = torch.arange(n_tot, dtype=torch.long).reshape(
            grid_depth, num_views, grid_height, grid_width
        )
        self.ctx_idx = token_ids[:context_tubelets].flatten()
        self.tgt_idx = token_ids[context_tubelets:].flatten()
        self.n_ctx = self.ctx_idx.numel()
        self.grid_depth = grid_depth
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_views = num_views
        self.n_target = self.tgt_idx.numel()

    @classmethod
    def from_layout(cls, layout):
        return cls(layout=layout)
