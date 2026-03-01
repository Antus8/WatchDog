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
import random

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import utils.utils as utils
import utils.PositionEncodings as PositionEncodings
import models.TransformerEncoder as Encoder
import models.TransformerDecoder as Decoder
from models.Transformer import Transformer

from fall_transformer.model import BodyClassificationNetwork


_SOURCE_LENGTH = 110
_TARGET_LENGTH = 55
_POSE_DIM = 54
_PAD_LENGTH = _SOURCE_LENGTH
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WatchPoseTransformer(nn.Module):
  """Implements the sequence-to-sequence Transformer .model for pose prediction."""
  def __init__(self,
               model_dim=32,
               model_dim_traj=128,
               num_encoder_layers=6,
               num_decoder_layers=6,
               num_heads=8,
               dim_ffn=2048,
               dropout=0.1,
               target_seq_length=_TARGET_LENGTH,
               source_seq_length=_SOURCE_LENGTH,
               input_dim=None,
               init_fn=utils.xavier_init_,
               non_autoregressive=False,
               use_query_embedding=False,
               pre_normalization=False,
               predict_activity=False,
               use_memory=False,
               pose_embedding=None,
               pose_decoder=None,
               traj_embedding=None,
               traj_decoder=None,
               copy_method='uniform_scan',
               query_selection=False,
               include_last_obs=False,
               pos_encoding_params=(10000, 1),
               train_dataset = None, 
               eval_dataset = None,
               fall_options = None):
    """Initialization of pose transformers."""
    super(WatchPoseTransformer, self).__init__()
    self._target_seq_length = target_seq_length
    self._source_seq_length = source_seq_length
    self._model_dim = model_dim
    self._model_dim_traj = model_dim_traj
    self._use_query_embedding = use_query_embedding
    self._predict_activity = predict_activity
    self._num_decoder_layers = num_decoder_layers
    self._mlp_dim = model_dim 
    self._non_autoregressive = non_autoregressive
    self._pose_embedding = pose_embedding
    self._pose_decoder = pose_decoder
    self._traj_embedding = traj_embedding
    self._traject_decoder = traj_decoder
    self._query_selection = query_selection
    thisname = self.__class__.__name__
    self._copy_method = copy_method
    self._pos_encoding_params = pos_encoding_params
    self._include_last_obs = include_last_obs

    self._train_dataset_fn = train_dataset
    self._eval_dataset_fn = eval_dataset

    self._fall_options = fall_options
    
    self._transformer = Transformer(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        model_dim=model_dim,
        model_dim_traj=model_dim_traj,
        num_heads=num_heads,
        dim_ffn=dim_ffn,
        dropout=dropout,
        init_fn=init_fn,
        use_query_embedding=use_query_embedding,
        pre_normalization=pre_normalization,
        query_selection=query_selection,
        target_seq_len=target_seq_length
    )
    print("Trajectory Transformer LOADED...")

    t_params = filter(lambda p: p.requires_grad, self._transformer.parameters())
    nparams = sum([np.prod(p.size()) for p in t_params])
    # print('[INFO] ({}) Transformer has {} parameters!'.format(thisname, nparams))


    # Compute positional encoding for the input sequences to be fed to encoder and decoder of the transformers
    self._traj_encoder = PositionEncodings.PositionEncodings1D(
        num_pos_feats=self._model_dim_traj,
        temperature=self._pos_encoding_params[0],
        alpha=self._pos_encoding_params[1]
    )
    self._traj_decoder = PositionEncodings.PositionEncodings1D(
        num_pos_feats=self._model_dim_traj,
        temperature=self._pos_encoding_params[0],
        alpha=self._pos_encoding_params[1]
    )
    # self.init_pose_encoder_decoders(init_fn)
    self._use_class_token = False
    self.init_position_encodings()
    self.init_query_embedding()

    if self._use_class_token:
      self.init_class_token()

    self.load_fall_transformer()



  def load_fall_transformer(self):
    '''
    Load the fall transformer with pretrained weights to compute fall predictions and pose embeddings
    '''
    self.fall_transformer = BodyClassificationNetwork(self._fall_options['num_transformer_blocks'], 
                                                      self._fall_options['d_input'], 
                                                      self._fall_options['d_model'], 
                                                      self._fall_options['num_heads'], 
                                                      self._fall_options['num_layers'], 
                                                      self._fall_options['d_ff'], 
                                                      self._fall_options['max_seq_length'], 
                                                      self._fall_options['dropout'], 
                                                      self._fall_options['inner_df'])
    
    weights_path = "../../Perception/weights/best_model.pth"

    self.fall_transformer.load_state_dict(torch.load(weights_path))
    self.fall_transformer.to(_DEVICE)

    # Freeze layers
    for param in self.fall_transformer.parameters():
      param.requires_grad = False
    print("Fall Transformer LOADED...")


  def init_query_embedding(self):
    """Initialization of query sequence embedding."""
    self._query_embed_traj = nn.Embedding(self._target_seq_length, self._model_dim_traj)

    print('[INFO] ({}) Init query embeddings!'.format(self.__class__.__name__))
    nn.init.xavier_uniform_(self._query_embed_traj.weight.data)
    
    # self._query_embed.weight.data.normal_(0.0, 0.004)

  def init_class_token(self):
    token = torch.FloatTensor(1, self._model_dim)
    print('[INFO] ({}) Init class token!'.format(self.__class__.__name__))
    self._class_token = nn.Parameter(token, requires_grad=True)
    nn.init.xavier_uniform_(self._class_token.data)

  def init_position_encodings(self):
    src_len = self._source_seq_length-1
    if self._include_last_obs:
        src_len += 1
    # when using a token we need an extra element in the sequence
    if self._use_class_token:
      src_len = src_len + 1
    
    encoder_traj_encodings = self._traj_encoder(src_len).view(
            src_len, 1, self._model_dim_traj)
    decoder_traj_encodings = self._traj_decoder(self._target_seq_length).view(
            self._target_seq_length, 1, self._model_dim_traj)
    
    mask_look_ahead = torch.from_numpy(
        utils.create_look_ahead_mask(
            self._target_seq_length, self._non_autoregressive))
    
    self._encoder_traj_encodings = nn.Parameter(
        encoder_traj_encodings, requires_grad=False)
    self._decoder_traj_encodings = nn.Parameter(
        decoder_traj_encodings, requires_grad=False)
    self._mask_look_ahead = nn.Parameter(
        mask_look_ahead, requires_grad=False)
    

  def forward(self, 
              input_pose_seq,
              target_pose_seq=None,
              input_traj_seq=None,
              target_traj_seq=None,
              mask_target_padding=None,
              get_attn_weights=False):
    """Performs the forward pass of the pose transformers.

    Args:
      input_pose_seq: Shape [batch_size, src_sequence_length, dim_pose].
      target_pose_seq: Shape [batch_size, tgt_sequence_length, dim_pose].
      input_traj_seq: Shape [batch_size, src_sequence_length, 3].
      target_traj_seq: Shape [batch_size, tgt_sequence_length, 3].

    Returns:
      A tensor of the predicted sequence with shape [batch_size, 
      tgt_sequence_length, dim_pose].
    """
    if self.training:
      return self.forward_training(
          input_pose_seq, target_pose_seq, input_traj_seq, target_traj_seq, mask_target_padding, get_attn_weights)

    # eval forward for non auto regressive type of model
    if self._non_autoregressive:
      return self.forward_training(
          input_pose_seq, target_pose_seq, input_traj_seq, target_traj_seq, mask_target_padding, get_attn_weights)

    return self.forward_autoregressive(
        input_pose_seq, target_pose_seq, input_traj_seq, target_traj_seq, mask_target_padding, get_attn_weights)



  def plot_joints(self, joints, traj):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    parents, offset, rot_ind, exp_map_ind = utils.load_constants("../data/h3.6m")

    ins = np.squeeze(joints.cpu().numpy())
    ins = self._train_dataset_fn.dataset.unnormalize(ins)

    ins_traj = np.squeeze(traj.cpu().numpy())
    ins_traj = self._train_dataset_fn.dataset.unnormalize_traj(ins_traj)

    joints_left=[3, 4, 5, 10, 11, 12]
    joints_right=[0, 1, 2, 13, 14, 15]

    elev = -125
    azim = -45

    frames3 = [ins.shape[0] -1] 

    k_num=0
    fig = plt.figure(figsize=(30,30))

    for kk in frames3:
      batch=kk
      ax = fig.add_subplot(1,2,1, projection='3d')
      ax.view_init(elev=elev, azim=azim)
      # ax.set_xlim3d([-3,3])
      # ax.set_ylim3d([-3,3])
      # ax.set_zlim3d([-1,1])
      
      ax.set_zticks([])
      ax.set_xticks([])
      ax.set_yticks([])

      for i in range(89):
        if i!=0 and i!=88:
            continue # Fraw only first and last pose
        pos = ins[batch][i].reshape(16,3)
        pos = np.concatenate((np.array([[0,0,0]]),pos),axis=0) + ins_traj[batch][i] - ins_traj[batch][0]
        alpha = min((i+0.1) * 0.15 + 0.25, 1.0)
        for j, j_parent in enumerate(parents):    
            if j_parent == -1:
                continue                
            ax.plot([pos[j, 0], pos[j_parent, 0]],
                    [pos[j, 2], pos[j_parent, 2]],
                    [pos[j, 1], pos[j_parent, 1]], zdir='z', c='blue',alpha=alpha)
    
    in_gt_traj = ins_traj[batch] - ins_traj[batch][0]
    ax.plot(in_gt_traj[:,0],in_gt_traj[:,2],in_gt_traj[:,1],'b--')

    ax.view_init(elev=elev, azim=azim)

    plt.show()


  def plot_single_pose(self, joints, traj):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9,  8, 11, 12, 8,  14, 15]

    wanted_joints =   [1, 2, 3, 4, 5, 6, 10, 11, 13, 14, 16]

    ins = np.squeeze(joints.cpu().numpy())
    ins = self._train_dataset_fn.dataset.unnormalize(ins)

    ins_traj = np.squeeze(traj.cpu().numpy())
    ins_traj = self._train_dataset_fn.dataset.unnormalize_traj(ins_traj)

    elev = -125
    azim = -45

    k_num=0
    fig = plt.figure(figsize=(30,30))

    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    x_lim = (-3, 3)
    y_lim = (-3, 3)
    z_lim = (-1, 1)

    ax.set_xticks(np.arange(x_lim[0], x_lim[1] + 1, 1))
    ax.set_yticks(np.arange(y_lim[0], y_lim[1] + 1, 1))
    ax.set_zticks(np.arange(z_lim[0], z_lim[1] + 1, 0.5))

    pos = ins[0][0].reshape(16,3)
    pos = np.concatenate((np.array([[0,0,0]]),pos),axis=0) + ins_traj[0][0] - ins_traj[0][0]
    alpha = 0.5

    # set depth to 0
    for p in pos:
      p[0] = p[0] + 0.5 + pos[0][0]
      p[1] = p[1] + 0.5 + pos[0][1]
      p[2] = 0
    
    pos = self.transform_coords(pos)
    for p in pos:
      p[2] = 0

    pos[0] = [0.5, 0.5, 0]

    for j, j_parent in enumerate(parents):    
        if j_parent == -1:# or j not in wanted_joints:
            continue                
        ax.plot([pos[j, 0], pos[j_parent, 0]],
                [pos[j, 1], pos[j_parent, 1]],
                [pos[j, 2], pos[j_parent, 2]], zdir='z', c='blue',alpha=alpha)
    
    out_points = []
    # Add points (joints) and their index labels
    for j in range(pos.shape[0]):
        # Plot a point for each joint
        if j in wanted_joints:
          ax.scatter(pos[j, 0], pos[j, 1], pos[j, 2], c='red', s=50, marker='o')  # Red points

          # Add index labels close to each point
          ax.text(pos[j, 0], pos[j, 1], pos[j, 2], str(j), color='black', fontsize=12)
          out_points.append([pos[j, 0], pos[j, 1]])

    image = self.plot_2d_points(out_points)

    ax_img = fig.add_subplot(1, 2, 2)
    ax_img.axis('off')  # Hide axes for the image
    ax_img.imshow(image) 


    # ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

  
  def transform_coords(self, joints):
    min_coords = np.min(joints, axis=0)  # (min_x, min_y, min_z)
    max_coords = np.max(joints, axis=0)  # (max_x, max_y, max_z)

    # Normalize to [0, 1]
    normalized_joints = (joints - min_coords) / (max_coords - min_coords)

    return normalized_joints


  def forward_training(self,
                       input_pose_seq_,
                       target_pose_seq_,
                       input_traj_seq_,
                       target_traj_seq_,
                       mask_target_padding,
                       get_attn_weights=False):
    """Compute forward pass for training and non recursive inference.
    Args:
       input_pose_seq_: Source sequence [batch_size, src_len, skeleton_dim].
       target_pose_seq_: Query target sequence [batch_size, tgt_len, skeleton_dim].
       mask_target_padding: Mask for target masking with ones where elements 
          belong to the padding elements of shape [batch_size, tgt_len, skeleton_dim].
       get_attn_weights: Boolean to indicate if attention weights should be returned.
    Returns:
    """

    # MEDIAPIPE TENSOR having shape [batch_size, seq_len, 23]
    input_pose_seq = input_pose_seq_ # To be fed into the fall transformer
    target_pose_seq = target_pose_seq_ 


    input_pose_seq = input_pose_seq.to(_DEVICE)
    # GO THROUGH first transformer
    with torch.no_grad():
      fall_pred = self.fall_transformer(input_pose_seq)
      pose_embeddings = self.fall_transformer.get_embeddings()

    input_traj_seq = input_traj_seq_
    target_traj_seq = target_traj_seq_
    
    if self._traj_embedding is not None:
      input_traj_seq = self._traj_embedding(input_traj_seq)
      target_traj_seq = self._traj_embedding(target_traj_seq)

    # compute the look-ahead mask and the positional encodings

    input_traj_seq = torch.transpose(input_traj_seq, 0, 1)
    target_traj_seq = torch.transpose(target_traj_seq, 0, 1)


    # 3) compute the attention weights using the transformer
    
    attn_output_traj, memory_traj, out_weights_traj, enc_weights_traj = self._transformer(
        pose_embeddings,
        input_traj_seq,
        target_traj_seq,
        query_embedding_traj = self._query_embed_traj.weight,
        encoder_trajectory_encodings=self._encoder_traj_encodings,
        decoder_trajectory_encodings=self._decoder_traj_encodings,
        mask_look_ahead=self._mask_look_ahead,
        mask_target_padding=mask_target_padding,
        get_attn_weights=get_attn_weights)
    
    out_sequence_traj = []
    target_traj_seq_ = torch.transpose(target_traj_seq_, 0, 1) # bring back to original dimension

    # 4) decode sequence with pose decoder. The decoding process is time
    # independent. It means non-autoregressive or parallel decoding.
      
    for l in range(self._num_decoder_layers):
      out_sequence_traj_ = self._traject_decoder(
          attn_output_traj[l].view(-1, self._model_dim_traj))
      
      out_sequence_traj_ = out_sequence_traj_.view(
          self._target_seq_length, -1, 3)
      
      out_sequence_traj_ = out_sequence_traj_ + target_traj_seq_[:, :, 0:3]
      
      out_sequence_traj_ = torch.transpose(out_sequence_traj_,0,1)
      out_sequence_traj.append(out_sequence_traj_)

    return out_sequence_traj, fall_pred



  def forward_autoregressive(self,
                      input_pose_seq,
                      target_pose_seq=None,
                      mask_target_padding=None,
                      get_attn_weights=False):
    """Compute forward pass for auto-regressive inferece in test time."""
    thisdevice = self._encoder_pos_encodings.device
    # the first query pose is the first in the target
    prev_target = target_pose_seq[:, 0, :]
    # 1) Enconde using the pose embeding
    if self._pose_embedding is not None:
      input_pose_seq = self._pose_embedding(input_pose_seq)
      target_pose_seq = self._pose_embedding(target_pose_seq)
    # [batch_size, 1, model_dim]
    target_seq = target_pose_seq[:, 0:1, :]

    # 2) compute the look-ahead mask and the positional encodings
    # [sequence_length, batch_size, model_dim]
    input_pose_seq = torch.transpose(input_pose_seq, 0, 1)
    target_seq = torch.transpose(target_seq, 0, 1)

    # 3) use auto recursion to compute the predicted set of tokens    
    memory, enc_attn_weights = self._transformer._encoder(
        input_pose_seq, self._encoder_pos_encodings)

    # get only the first In teory it should only be one target pose at testing
    batch_size = memory.size()[1]
    out_pred_seq = torch.FloatTensor(
        batch_size, self._target_seq_length, self._pose_dim).to(thisdevice)

    for t in range(self._target_seq_length):
      position_encodings = self._pos_decoder(t+1).view(
          t+1, 1, self._model_dim).to(thisdevice)
      mask_look_ahead = torch.from_numpy(
          utils.create_look_ahead_mask(t+1)).to(thisdevice)

      # a list of length n_decoder_layers with elements of 
      # shape [t, batch_size, model_dim]
      out_attn, out_weights = self._transformer._decoder(
          target_seq,
          memory,
          position_encodings,
          mask_look_ahead=mask_look_ahead
      )
      # get only the last predicted token decode it to get the pose and 
      # then encode the pose. shape [1*batch_size, pose_dim]
      # for 8 seeds of evaluation (batch_size)
      pred_pose = self._pose_decoder(
          out_attn[-1][t:(t+1), :, :].view(-1, self._model_dim))
      # apply residual between last target pose and recently generated pose
      if self._pose_dim == self._input_dim:
        pred_pose = pred_pose + prev_target
      else:
        prev_target[:, 0:self._pose_dim] = pred_pose + prev_target[:,0:self._pose_dim]
        pred_pose = prev_target
      prev_target = pred_pose
      out_pred_seq[:, t, :] = pred_pose.view(-1, self._input_dim)[:, 0:self._pose_dim]
      if self._pose_embedding is not None:
        pose_code = self._pose_embedding(pred_pose)
        # [1, batch_size, model_dim]
        pose_code = pose_code.view(-1, batch_size, self._model_dim)
      # [t+1, batch_size, model_dim]
      target_seq = torch.cat([target_seq, pose_code], axis=0)

    # 1) the last attention output contains all the necessary sequence; or
    # 2) Use all the memory to predict

    return [out_pred_seq]


def model_factory(params, fall_options, train_dataset_fn, eval_dataset_fn, traj_embedding_fn, traj_decoder_fn):
  init_fn = utils.normal_init_ \
      if params['init_fn'] == 'normal_init' else utils.xavier_init_
  return WatchPoseTransformer(
      model_dim=params['model_dim'],
      model_dim_traj=params['model_dim_traj'],
      num_encoder_layers=params['num_encoder_layers'],
      num_decoder_layers=params['num_decoder_layers'],
      num_heads=params['num_heads'],
      dim_ffn=params['dim_ffn'],
      dropout=params['dropout'],
      target_seq_length=params['target_seq_len'],
      source_seq_length=params['source_seq_len'],
      init_fn=init_fn,
      non_autoregressive=params['non_autoregressive'],
      use_query_embedding=params['use_query_embedding'],
      pre_normalization=params['pre_normalization'],
      predict_activity=params['predict_activity'],
      use_memory=params['use_memory'],
      traj_embedding=traj_embedding_fn(params),
      traj_decoder=traj_decoder_fn(params),
      query_selection=params['query_selection'],
      include_last_obs=params['include_last_obs'],
      pos_encoding_params=(params['pos_enc_beta'], params['pos_enc_alpha']),
      train_dataset = train_dataset_fn, 
      eval_dataset = eval_dataset_fn,
      fall_options = fall_options
  )


if __name__ == '__main__':
  transformer = WatchPoseTransformer(model_dim=_POSE_DIM, num_heads=8)
  transformer.eval()
  transformer._non_autoregressive = True
  batch_size = 8
  model_dim = 256
  tgt_seq = torch.FloatTensor(batch_size, _TARGET_LENGTH, 128).fill_(1)
  src_seq = torch.FloatTensor(batch_size, _SOURCE_LENGTH-1, 128).fill_(1)

  outputs = transformer(src_seq, tgt_seq)
  print(outputs[-1].size())