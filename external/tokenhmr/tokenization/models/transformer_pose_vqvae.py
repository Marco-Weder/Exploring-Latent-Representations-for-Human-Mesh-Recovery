import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
from collections import OrderedDict

from .quantize_cnn import QuantizeEMAReset
from .rotation_utils import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# UPDATED: Import CrossAttention here
from external.tokenhmr.tokenhmr.lib.models.components.pose_transformer import TransformerEncoder, CrossAttention

# --- SMPL Instantiation ---
from smplx import SMPLHLayer
smpl_type = 'smplh'
current_dir = os.path.dirname(os.path.realpath(__file__))
body_model_path = os.path.join(current_dir, '..', '..', 'data/body_models', smpl_type)
body_model = eval(f'{smpl_type.upper()}Layer')(body_model_path, num_betas=10, ext='pkl')
body_model = body_model.cuda() if torch.cuda.is_available() else body_model

def step_multiplier_mapping():
    return {0: 1e-2, 1: 5e-2, 2: 1e-1, 3: 1e-1, 4: 5e-1, 5: 5e-1}

class TransformerTokenizer(nn.Module):
    def __init__(self, arch_params=None, input_joint_dim=6, output_joint_dim=6, mesh_inference=True, add_noise=False):
        super().__init__()
        self.num_joints = arch_params.NB_JOINTS if hasattr(arch_params, 'NB_JOINTS') else 21
        self.code_dim = arch_params.CODE_DIM[0] if isinstance(arch_params.CODE_DIM, list) else arch_params.CODE_DIM
        self.num_code = arch_params.NB_CODE[0] if isinstance(arch_params.NB_CODE, list) else arch_params.NB_CODE
        self.width = arch_params.WIDTH
        self.depth = arch_params.DEPTH
        self.quant = arch_params.QUANTIZER
        self.rot_type = arch_params.ROT_TYPE
        self.input_joint_dim = input_joint_dim
        self.output_joint_dim = output_joint_dim
        self.mesh_inference = mesh_inference
        self.add_noise = add_noise
        self.step_multiplier_mapping = step_multiplier_mapping()
        if self.add_noise:
            from utils.skeleton import get_smplx_body_parts
            self.smplx_body_parts = get_smplx_body_parts()

        self.num_tokens = getattr(arch_params, 'NUM_TOKENS', 160)
        self.n_heads = getattr(arch_params, 'N_HEADS', 8)
        self.dim_head = getattr(arch_params, 'DIM_HEAD', 64)

        # 1. ENCODER (Self-Attention)
        self.encoder = TransformerEncoder(
            num_tokens=self.num_joints,
            token_dim=self.input_joint_dim,
            dim=self.code_dim,
            depth=self.depth,
            heads=self.n_heads,
            mlp_dim=self.width,
        )

        # 2. DOWNSAMPLE (Cross-Attention)
        self.latent_queries = nn.Parameter(torch.randn(1, self.num_tokens, self.code_dim) * 0.02)
        self.cross_attn_down = CrossAttention(dim=self.code_dim, context_dim=self.code_dim, heads=self.n_heads, dim_head=self.dim_head)

        # 3. QUANTIZER
        self.quantizer = QuantizeEMAReset(self.num_code, self.code_dim)

        # 4. DECODER (Self-Attention)
        self.decoder = TransformerEncoder(
            num_tokens=self.num_tokens,
            token_dim=self.code_dim,
            dim=self.width,
            depth=self.depth,
            heads=self.n_heads,
            mlp_dim=self.width,
        )

        # 5. UPSAMPLE (Cross-Attention)
        self.joint_queries = nn.Parameter(torch.randn(1, self.num_joints, self.width) * 0.02)
        self.cross_attn_up = CrossAttention(dim=self.width, context_dim=self.width, heads=self.n_heads, dim_head=self.dim_head)
        
        self.decoder_projection = nn.Linear(self.width, self.output_joint_dim)

    def encode(self, x):
        batch_size = x.shape[0]
        if x.dim() == 2:
            x = x.view(batch_size, self.num_joints, -1)
            
        if x.shape[-1] == 3 and self.input_joint_dim == 6:
            x = matrix_to_rotation_6d(x)
            
        x_encoder = self.encoder(x)
        
        # Cross-Attention Downsample
        queries = self.latent_queries.expand(batch_size, -1, -1) 
        x_encoder = self.cross_attn_down(queries, context=x_encoder)
        
        x_encoder = x_encoder.contiguous().view(-1, self.code_dim)
        code_idx = self.quantizer.quantize(x_encoder)
        
        return code_idx.view(batch_size, -1)

    def decode_logits(self, logits):
        batch_size = logits.shape[0]
        decode_feat = self.quantizer.dequantize_logits(logits) 
        
        if decode_feat.shape[1] == self.code_dim:
            decode_feat = decode_feat.permute(0, 2, 1)
        
        x_decoder = self.decoder(decode_feat)
        
        # Cross-Attention Upsample
        j_queries = self.joint_queries.expand(batch_size, -1, -1)
        x_decoder = self.cross_attn_up(j_queries, context=x_decoder)
        
        return self.decoder_projection(x_decoder)

    def forward(self, x, global_step=None):
        batch_size = x.shape[0]
        # ADDED: Reshape safety in forward
        if x.dim() == 2:
            x = x.view(batch_size, self.num_joints, -1)

        if x.shape[-1] == 3 and self.input_joint_dim == 6:
            x = matrix_to_rotation_6d(x)
            
        if self.training and self.add_noise and global_step is not None:
            step = global_step // 5000
            noise_multiplier = float(self.step_multiplier_mapping[step]) if step <=5 else 0.5
            noised_samples = np.random.randint(low=0, high=batch_size-1, size=batch_size//2)
            mask_part = np.random.randint(len(self.smplx_body_parts.keys()))
            masked_joints = self.smplx_body_parts[mask_part]
            
            noise = torch.cuda.FloatTensor(1).uniform_() * noise_multiplier
            x = x.clone()
            x[noised_samples[:, None], masked_joints] += noise
            
        # 1. Encode + Cross-Attention Downsample
        x_encoder = self.encoder(x)
        queries = self.latent_queries.expand(batch_size, -1, -1) 
        x_encoder = self.cross_attn_down(queries, context=x_encoder)

        # 2. Quantize (Requires permute to channel-first)
        x_encoder_chan = x_encoder.permute(0, 2, 1) # (B, code_dim, 160)

        x_quantized, loss, perplexity = self.quantizer(x_encoder_chan)

        # 3. Decode + Cross-Attention Upsample
        x_quantized_seq = x_quantized.permute(0, 2, 1) 
        x_decoder = self.decoder(x_quantized_seq)
        
        j_queries = self.joint_queries.expand(batch_size, -1, -1)
        x_decoder = self.cross_attn_up(j_queries, context=x_decoder)
        
        pred_pose_6d = self.decoder_projection(x_decoder)
        
        # ... keep same dictionary and mesh logic below ...
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