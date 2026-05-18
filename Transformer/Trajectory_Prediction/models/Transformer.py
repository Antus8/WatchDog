###############################################################################
# (WATCHDOG): Elderly Assistance via Active Attention-based Fall Detection 
# and Motion Prediction
#
# Original work:
# Copyright (c) 2022 MARS Lab at Simon Fraser University
# Written by 
# Mohammad Mahdavian <mmahdavi@sfu.ca>
#
# Modifications:
# Copyright (c) 2025 SST Lab at Politecnico di Bari
# Written by 
# Antonello Longo <a.longo70@phd.poliba.it>
#
# This file is part of WATCHDOG: Elderly Assistance via Active 
# Attention-based Fall Detection and Motion Prediction
#
# WATCHDOG is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# WATCHDOG is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with WATCHDOG. If not, see <http://www.gnu.org/licenses/>.
###############################################################################


import numpy as np
import os
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import utils.utils as utils
import models.TransformerEncoder as Encoder
import models.TransformerDecoder as Decoder

class Transformer(nn.Module):
  def __init__(self,
              num_encoder_layers=6,
              num_decoder_layers=6,
              model_dim=256,  
              model_dim_traj=128,
              num_heads=8,
              dim_ffn=2048,
              dropout=0.1,
              init_fn=utils.normal_init_,
              use_query_embedding=False,
              pre_normalization=False,
              query_selection=False,
              end_attention=True,
              target_seq_len=25):
    """Implements the Transformer model for sequence-to-sequence modeling."""
    super(Transformer, self).__init__()
    self._model_dim = model_dim
    self._model_dim_traj = model_dim_traj
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout
    self._use_query_embedding = use_query_embedding
    self._query_selection = query_selection
    self._tgt_seq_len = target_seq_len
    self._end_attention = end_attention

    # TRAJECTORY TRANSFORMER
    self._encoder_traj = Encoder.TransformerEncoder(
        num_layers=num_encoder_layers,
        model_dim=model_dim_traj,
        num_heads=num_heads,
        dim_ffn=dim_ffn,
        dropout=dropout,
        init_fn=init_fn,
        pre_normalization=pre_normalization
    )
    
    self._decoder_traj = Decoder.TransformerDecoder(
        num_layers=num_decoder_layers,
        model_dim=model_dim_traj,
        num_heads=num_heads,
        dim_ffn=dim_ffn,
        dropout=dropout,
        init_fn=init_fn,
        use_query_embedding=use_query_embedding,
        pre_normalization=pre_normalization
    )
    
    if self._query_selection:
      self._position_predictor = nn.Linear(self._model_dim, self._tgt_seq_len)

    self._self_attn_traj = nn.MultiheadAttention(model_dim_traj, num_heads, dropout) # Cross attention module

    # End attention modules
    self._self_attn_end_traj = nn.MultiheadAttention(model_dim_traj, num_heads, dropout)

    # This is used to project the encodings of the pose in the dimension space of the traj
    self._traj_pos_linear = nn.Linear(32, self._model_dim_traj)  

    # These are used to compute qkv for the final end attention module both in pose and trajectory
    self.qkv_traj_end = nn.Linear(self._model_dim_traj, 3*self._model_dim_traj)

    # These are used to compute the q, k and v for the cross attention
    self.q_traj = nn.Linear(self._model_dim_traj, self._model_dim_traj) 
    self.k_traj = nn.Linear(self._model_dim_traj, self._model_dim_traj)
    self.v_traj = nn.Linear(self._model_dim_traj, self._model_dim_traj)


  def forward(self,
              pose_embeddings,
              source_seq_traj,
              target_seq_traj,
              encoder_trajectory_encodings=None,
              decoder_trajectory_encodings=None,
              query_embedding_traj=None,
              mask_target_padding=None,
              mask_look_ahead=None,
              get_attn_weights=False,
              query_selection_fn=None):
      
    if self._use_query_embedding:
      bs = source_seq_traj.size()[1]
      query_embedding_traj = query_embedding_traj.unsqueeze(1).repeat(1, bs, 1)
      decoder_trajectory_encodings = encoder_trajectory_encodings
    
    # ENCODERS for pose and trajectory
    memory_traj, enc_weights_traj = self._encoder_traj(source_seq_traj, encoder_trajectory_encodings)

    # Cross-attention where q is the traj emb, while k and v are computed from the pose
    pose_embeddings_memory = pose_embeddings.permute(1, 0, 2)
    memory_copy = self._traj_pos_linear(pose_embeddings_memory)

    # Compute cross attention between traj and my transformer
    query = self.q_traj(memory_traj)
    key = self.k_traj(memory_copy)
    value = self.v_traj(memory_copy)
    
    attn_output_traj_pose, attn_weights_traj_pose = self._self_attn_traj(
        query, 
        key, 
        value, 
        need_weights=True
    )

    memory_traj = memory_traj.clone() + attn_output_traj_pose
    
    out_attn_traj, out_weights_traj = self._decoder_traj(
        target_seq_traj,
        memory_traj,
        decoder_trajectory_encodings,
        query_embedding=query_embedding_traj,
        mask_target_padding=mask_target_padding,
        mask_look_ahead=mask_look_ahead,
        get_attn_weights=get_attn_weights
    )
    
    output_attn = []
    output_attn_traj=[]
    
    # END attention module
    for i in range(len(out_attn_traj)):
        enc_dec_tot_traj = torch.concat((memory_traj,out_attn_traj[i]),dim=0) 
        qkv_end_traj = self.qkv_traj_end(enc_dec_tot_traj)
        q_end_traj,k_end_traj,v_end_traj = qkv_end_traj.chunk(3,dim=-1)
        end_attn_traj, end_attn_traj_weights = self._self_attn_end_traj(q_end_traj,k_end_traj,v_end_traj,need_weights=True)
        output_attn_traj.append(end_attn_traj[-self._tgt_seq_len:])
        

    out_weights_ = None
    enc_weights_ = None
    prob_matrix_ = None

    return  output_attn_traj,memory_traj, out_weights_traj, enc_weights_traj