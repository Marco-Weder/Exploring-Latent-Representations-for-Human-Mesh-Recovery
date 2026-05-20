import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

# SMPL-H kinematic tree for 21 body joints (root excluded). Index 0 = L_Hip, ..., 20 = R_Wrist.
# -1 means parent is the root pelvis. Used only when USE_KINEMATIC_PE=True.
_SMPLH_PARENTS_21 = [-1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 11, 12, 13, 15, 16, 17, 18]


def _build_kinematic_laplacian_pe(num_joints: int = 21) -> torch.Tensor:
    """Laplacian-eigenvector positional encoding for the SMPL kinematic tree.

    Returns (num_joints, num_joints): each row is the PE of one joint.
    Uses the full eigenvector basis (no truncation, no trivial-eigvec drop).
    """
    parents = _SMPLH_PARENTS_21[:num_joints]
    A = torch.zeros(num_joints, num_joints)
    for i, p in enumerate(parents):
        if p >= 0:
            A[i, p] = 1.0
            A[p, i] = 1.0
    deg = A.sum(dim=1).clamp(min=1.0)
    d_inv_sqrt = torch.diag(deg.pow(-0.5))
    L = torch.eye(num_joints) - d_inv_sqrt @ A @ d_inv_sqrt
    _, eigvecs = torch.linalg.eigh(L)
    return eigvecs


from .quantize_cnn import QuantizeEMAReset
from .fsq import FSQQuantizer
from .rotation_utils import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle

# Import `pose_transformer` via a path that bypasses `tokenhmr/lib/models/__init__.py`.
# Going through that __init__ triggers a circular import when this module is loaded
# from inside TokenHMR initialization (lib/models → heads → token_classifier → here →
# back to lib/models, which is still partially initialised). Inserting
# `tokenhmr/lib/models/` directly lets us import `components.pose_transformer` as a
# top-level package, skipping the parent __init__ entirely. The `components/__init__.py`
# is empty and `pose_transformer.py`'s only relative import (`.t_cond_mlp`) stays valid
# because it's a sibling inside the `components` package.
_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../tokenhmr/lib/models'))
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)
from components.pose_transformer import (
    TransformerEncoder,
    TransformerCrossAttn,
    CrossAttention,
)

# --- SMPL Instantiation ---
from smplx import SMPLHLayer
smpl_type = 'smplh'
current_dir = os.path.dirname(os.path.realpath(__file__))
body_model_path = os.path.join(current_dir, '..', '..', 'data/body_models', smpl_type)
body_model = eval(f'{smpl_type.upper()}Layer')(body_model_path, num_betas=10, ext='pkl')
body_model = body_model.cuda() if torch.cuda.is_available() else body_model


def step_multiplier_mapping():
    return {0: 1e-2, 1: 5e-2, 2: 1e-1, 3: 1e-1, 4: 5e-1, 5: 5e-1}


def _make_cross_attn(dim, context_dim, heads, depth, mlp_dim, dropout):
    """Down/up cross-attention module. depth>=2 → stacked Perceiver-style block;
    depth==1 → bare CrossAttention (matches the previous architecture exactly)."""
    if depth <= 1:
        return CrossAttention(
            dim=dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim // heads,
            dropout=dropout,
        )
    return TransformerCrossAttn(
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim // heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        context_dim=context_dim,
    )


class TransformerTokenizer(nn.Module):
    def __init__(self, arch_params=None, input_joint_dim=6, output_joint_dim=6, mesh_inference=True, add_noise=False):
        super().__init__()
        self.num_joints = arch_params.NB_JOINTS if hasattr(arch_params, 'NB_JOINTS') else 21
        self.width = arch_params.WIDTH
        self.depth = arch_params.DEPTH
        self.quant = arch_params.QUANTIZER
        self.rot_type = arch_params.ROT_TYPE

        # FSQ overrides code_dim / num_code from FSQ_LEVELS; the EMA path keeps the
        # configured CODE_DIM / NB_CODE. Doing this before any module construction so
        # `self.to_code` and the decoder pick up the right code_dim.
        if self.quant == 'fsq':
            self._fsq_levels = list(arch_params.FSQ_LEVELS)
            self.code_dim = len(self._fsq_levels)
            self.num_code = int(np.prod(self._fsq_levels))
        else:
            self.code_dim = arch_params.CODE_DIM[0] if isinstance(arch_params.CODE_DIM, list) else arch_params.CODE_DIM
            self.num_code = arch_params.NB_CODE[0] if isinstance(arch_params.NB_CODE, list) else arch_params.NB_CODE
        self.input_joint_dim = input_joint_dim
        self.output_joint_dim = output_joint_dim

        self.mesh_inference = mesh_inference
        self.add_noise = add_noise
        self.step_multiplier_mapping = step_multiplier_mapping()
        if self.add_noise:
            from utils.skeleton import get_smplx_body_parts
            self.smplx_body_parts = get_smplx_body_parts()

        self.num_tokens = getattr(arch_params, 'NUM_TOKENS', 160)

        # FSQ uses small code_dim = len(levels) (typ. 3-7) and can't satisfy the %8 rule;
        # heads always operate on `width` because the decoder embeds token_dim -> width
        # internally, so the constraint only applies to the EMA path.
        if self.quant != 'fsq':
            assert self.code_dim % 8 == 0, f'CODE_DIM={self.code_dim} must be divisible by 8'
        assert self.width % 8 == 0, f'WIDTH={self.width} must be divisible by 8'

        # Tier-1 capacity knobs (default to old single-block / 1× behaviour for compat).
        ffn_mult       = int(getattr(arch_params, 'FFN_MULT', 1))
        n_down_blocks  = int(getattr(arch_params, 'N_DOWN_BLOCKS', 1))
        n_up_blocks    = int(getattr(arch_params, 'N_UP_BLOCKS', 1))

        self.use_kinematic_pe = bool(getattr(arch_params, 'USE_KINEMATIC_PE', False))
        if self.use_kinematic_pe:
            lap_pe = _build_kinematic_laplacian_pe(self.num_joints)  # (J, J)
            self.register_buffer('lap_pe', lap_pe)
            # Encoder runs at `width` now → kinematic PE projects to width.
            self.kinematic_pe_proj = nn.Linear(self.num_joints, self.width, bias=False)

        _dropout          = float(getattr(arch_params, 'DROPOUT', 0.0))
        _emb_dropout      = float(getattr(arch_params, 'EMB_DROPOUT', 0.0))
        _emb_dropout_type = str(getattr(arch_params, 'EMB_DROPOUT_TYPE', 'drop'))
        _cross_dropout    = float(getattr(arch_params, 'CROSS_ATTN_DROPOUT', 0.0))

        # 1. ENCODER — runs at `width` (matches CNN baseline's full-width processing).
        self.encoder = TransformerEncoder(
            num_tokens=self.num_joints,
            token_dim=self.input_joint_dim,
            dim=self.width,
            depth=self.depth,
            heads=8,
            mlp_dim=ffn_mult * self.width,
            dropout=_dropout,
            emb_dropout=_emb_dropout,
            emb_dropout_type=_emb_dropout_type,
        )

        # 2. DOWNSAMPLE — stacked cross-attn at `width`, queries also at `width`.
        self.latent_queries = nn.Parameter(torch.randn(1, self.num_tokens, self.width))
        self.cross_attn_down = _make_cross_attn(
            dim=self.width,
            context_dim=self.width,
            heads=8,
            depth=n_down_blocks,
            mlp_dim=ffn_mult * self.width,
            dropout=_cross_dropout,
        )

        # 2b. Project to code_dim only at the bottleneck (just before the quantizer).
        # FSQ-only: normalize before bottleneck to prevent encoder saturation into tanh bounds
        # (which caused perplexity collapse 1000 -> 90 in the [8,8,6,5] run). EMA path stays
        # byte-for-byte identical because nn.Identity is a no-op.
        self.code_norm = nn.LayerNorm(self.width) if self.quant == 'fsq' else nn.Identity()
        self.to_code = nn.Linear(self.width, self.code_dim)

        # 3. QUANTIZER (still at code_dim).
        if self.quant == 'fsq':
            self.quantizer = FSQQuantizer(levels=self._fsq_levels)
        else:
            self.quantizer = QuantizeEMAReset(
                self.num_code, self.code_dim,
                dist_metric=getattr(arch_params, 'DIST_METRIC', 'l2'),
            )

        # 4. DECODER — runs at width, embeds quantized code_dim → width via to_token_embedding.
        self.decoder = TransformerEncoder(
            num_tokens=self.num_tokens,
            token_dim=self.code_dim,
            dim=self.width,
            depth=self.depth,
            heads=8,
            mlp_dim=ffn_mult * self.width,
            dropout=_dropout,
            emb_dropout=_emb_dropout,
        )

        # 5. UPSAMPLE — stacked cross-attn at width.
        self.joint_queries = nn.Parameter(torch.randn(1, self.num_joints, self.width))
        self.cross_attn_up = _make_cross_attn(
            dim=self.width,
            context_dim=self.width,
            heads=8,
            depth=n_up_blocks,
            mlp_dim=ffn_mult * self.width,
            dropout=_cross_dropout,
        )

        self.decoder_projection = nn.Linear(self.width, self.output_joint_dim)

    def encode(self, x):
        batch_size = x.shape[0]
        if x.dim() == 2:
            x = x.view(batch_size, self.num_joints, -1)

        if x.shape[-1] == 3 and self.input_joint_dim == 6:
            x = matrix_to_rotation_6d(x)

        x_encoder = self.encoder(x)
        if self.use_kinematic_pe:
            kin_pe = self.kinematic_pe_proj(self.lap_pe).unsqueeze(0)  # (1, J, width)
            x_encoder = x_encoder + kin_pe

        queries = self.latent_queries.expand(batch_size, -1, -1)
        x_encoder = self.cross_attn_down(queries, context=x_encoder)
        x_encoder = self.to_code(self.code_norm(x_encoder))  # (B, num_tokens, code_dim)

        x_encoder = x_encoder.permute(0, 2, 1).contiguous()
        x_encoder = self.quantizer.preprocess(x_encoder)
        code_idx = self.quantizer.quantize(x_encoder)
        return code_idx.view(batch_size, -1)

    def decode_logits(self, logits):
        batch_size = logits.shape[0]
        decode_feat = self.quantizer.dequantize_logits(logits)  # (B, num_tokens, code_dim)

        x_decoder = self.decoder(decode_feat)
        j_queries = self.joint_queries.expand(batch_size, -1, -1)
        x_decoder = self.cross_attn_up(j_queries, context=x_decoder)
        return self.decoder_projection(x_decoder)

    def forward(self, x, global_step=None):
        batch_size = x.shape[0]
        if x.dim() == 2:
            x = x.view(batch_size, self.num_joints, -1)

        if x.shape[-1] == 3 and self.input_joint_dim == 6:
            x = matrix_to_rotation_6d(x)

        if self.training and self.add_noise and global_step is not None:
            step = global_step // 5000
            noise_multiplier = float(self.step_multiplier_mapping[step]) if step <= 5 else 0.5
            noised_samples = np.random.randint(low=0, high=batch_size - 1, size=batch_size // 2)
            mask_part = np.random.randint(len(self.smplx_body_parts.keys()))
            masked_joints = self.smplx_body_parts[mask_part]
            noise = torch.cuda.FloatTensor(1).uniform_() * noise_multiplier
            x = x.clone()
            for s_idx in noised_samples:
                x[s_idx, masked_joints] += noise

        # 1. Encode (full width)
        x_encoder = self.encoder(x)
        if self.use_kinematic_pe:
            kin_pe = self.kinematic_pe_proj(self.lap_pe).unsqueeze(0)  # (1, J, width)
            x_encoder = x_encoder + kin_pe
        queries = self.latent_queries.expand(batch_size, -1, -1)
        x_encoder = self.cross_attn_down(queries, context=x_encoder)  # (B, num_tokens, width)
        x_encoder = self.to_code(self.code_norm(x_encoder))            # (B, num_tokens, code_dim)

        # 2. Quantize
        x_encoder_chan = x_encoder.permute(0, 2, 1)
        x_quantized_chan, loss, perplexity = self.quantizer(x_encoder_chan)
        x_quantized = x_quantized_chan.permute(0, 2, 1)

        # 3. Decode + Upsample (decoder embeds code_dim → width internally)
        x_decoder = self.decoder(x_quantized)
        j_queries = self.joint_queries.expand(batch_size, -1, -1)
        x_decoder = self.cross_attn_up(j_queries, context=x_decoder)
        pred_pose_6d = self.decoder_projection(x_decoder)

        output = {}
        if self.rot_type == 'rot6d':
            pred_pose_rotmat = rotation_6d_to_matrix(pred_pose_6d.reshape(-1, 6)).view(batch_size, self.num_joints, 3, 3)
        output.update({'pred_pose_body_6d': pred_pose_6d, 'pred_pose_body_rotmat': pred_pose_rotmat})

        if self.mesh_inference:
            pred_pose_aa = matrix_to_axis_angle(pred_pose_rotmat.view(-1, 3, 3)).view(batch_size, 3 * self.num_joints)
            pred_body_mesh = body_model(body_pose=pred_pose_rotmat)
            output.update({'pred_pose_body_aa': pred_pose_aa, 'pred_body_mesh': pred_body_mesh,
                           'pred_body_vertices': pred_body_mesh.vertices, 'pred_body_joints': pred_body_mesh.joints})
        return output, loss, perplexity


class TransformerDecodeTokens(nn.Module):
    """Decoder-only wrapper for TransformerTokenizer — used by TokenHMR's TokenClassfier
    at inference (mirror of VanillaDecodeTokens for the CNN baseline).

    Reads architecture hyperparameters from the checkpoint's `hparams.ARCH` and reconstructs
    only the decoder-side modules (quantizer, decoder, joint_queries, cross_attn_up,
    decoder_projection). Compatible with both:
      - "simple transformer" checkpoints (FFN_MULT=1, N_UP_BLOCKS=1 — bare CrossAttention)
      - Tier-1 checkpoints                (FFN_MULT>=2, N_UP_BLOCKS>=2 — stacked TransformerCrossAttn)

    `forward(logits)` accepts soft codebook logits `(B, num_tokens, num_codes)` from the
    TokenClassfier and returns `(B, num_joints, 6)` 6D pose, matching VanillaDecodeTokens.
    """

    def __init__(self, ckpt_path: str = '', mesh_inference: bool = False):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        arch = ckpt['hparams'].ARCH

        self.num_joints = getattr(arch, 'NB_JOINTS', 21)
        self.width      = arch.WIDTH
        self.depth      = arch.DEPTH
        self.num_tokens = getattr(arch, 'NUM_TOKENS', 160)

        # Mirror the TransformerTokenizer quantizer branch — FSQ overrides
        # code_dim / num_code from FSQ_LEVELS so the decoder embedding shape matches the ckpt.
        self.quant = getattr(arch, 'QUANTIZER', 'ema_reset')
        if self.quant == 'fsq':
            self._fsq_levels = list(arch.FSQ_LEVELS)
            self.code_dim = len(self._fsq_levels)
            self.num_code = int(np.prod(self._fsq_levels))
        else:
            self.code_dim = arch.CODE_DIM[0] if isinstance(arch.CODE_DIM, list) else arch.CODE_DIM
            self.num_code = arch.NB_CODE[0]  if isinstance(arch.NB_CODE,  list) else arch.NB_CODE

        # Same defaults as TransformerTokenizer (1 = old simple-transformer behaviour).
        ffn_mult       = int(getattr(arch, 'FFN_MULT', 1))
        n_up_blocks    = int(getattr(arch, 'N_UP_BLOCKS', 1))
        _dropout       = float(getattr(arch, 'DROPOUT', 0.0))
        _emb_dropout   = float(getattr(arch, 'EMB_DROPOUT', 0.0))
        _cross_dropout = float(getattr(arch, 'CROSS_ATTN_DROPOUT', 0.0))

        if self.quant == 'fsq':
            self.quantizer = FSQQuantizer(levels=self._fsq_levels)
        else:
            self.quantizer = QuantizeEMAReset(self.num_code, self.code_dim)

        self.decoder = TransformerEncoder(
            num_tokens=self.num_tokens,
            token_dim=self.code_dim,
            dim=self.width,
            depth=self.depth,
            heads=8,
            mlp_dim=ffn_mult * self.width,
            dropout=_dropout,
            emb_dropout=_emb_dropout,
        )

        self.joint_queries = nn.Parameter(torch.randn(1, self.num_joints, self.width))
        self.cross_attn_up = _make_cross_attn(
            dim=self.width,
            context_dim=self.width,
            heads=8,
            depth=n_up_blocks,
            mlp_dim=ffn_mult * self.width,
            dropout=_cross_dropout,
        )
        self.decoder_projection = nn.Linear(self.width, 6)

        self._load_weights(ckpt['net'])

    def _load_weights(self, state_dict: dict):
        """Filter the full tokenizer state_dict down to decoder-side keys and load them."""
        decoder_prefixes = (
            'quantizer.',
            'decoder.',
            'joint_queries',
            'cross_attn_up.',
            'decoder_projection.',
        )
        filtered = {k: v for k, v in state_dict.items()
                    if any(k.startswith(p) for p in decoder_prefixes)}
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        decoder_missing = [k for k in missing if any(k.startswith(p) for p in decoder_prefixes)]
        if decoder_missing:
            print(f'[TransformerDecodeTokens] WARNING: missing decoder keys: {decoder_missing}')
        if unexpected:
            print(f'[TransformerDecodeTokens] WARNING: unexpected keys: {unexpected}')

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Soft-decode token logits to 6D pose.

        Args:
            logits: (B, num_tokens, num_codes) softmax weights from TokenClassfier.
        Returns:
            (B, num_joints, 6) 6D rotation predictions for the body joints.
        """
        batch_size = logits.shape[0]
        decode_feat = self.quantizer.dequantize_logits(logits)  # (B, num_tokens, code_dim)
        x_decoder = self.decoder(decode_feat)
        j_queries = self.joint_queries.expand(batch_size, -1, -1)
        x_decoder = self.cross_attn_up(j_queries, context=x_decoder)
        return self.decoder_projection(x_decoder)
