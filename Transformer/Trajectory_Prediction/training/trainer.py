import numpy as np
import os
import sys
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import training.seq2seq_model_fn as seq2seq_model_fn
import models.WatchPoseTransformer as WatchPoseTransformer
import models.PoseEncoderDecoder as PoseEncoderDecoder
import data.H36MDataset_v3 as H36MDataset_v3
import utils.utils as utils

from data.h36m_dataset import Human36mDataset


_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_NSEEDS = 8


class WatchModelFn(seq2seq_model_fn.ModelFn):
    def __init__(self,
                params,
                fall_options,
                train_dataset_fn,
                eval_dataset_fn,
                eval_dataset_fn_total,
                traj_encoder_fn,
                traj_decoder_fn):
        super(WatchModelFn, self).__init__(
            params, fall_options, train_dataset_fn, eval_dataset_fn,eval_dataset_fn_total, traj_encoder_fn, traj_decoder_fn)
        self._loss_fn = self.my_layerwise_loss_fn
        self._traj_loss_fn = self.my_layerwise_loss_fn
        self._train_dataset_fn = train_dataset_fn
        self._eval_dataset_fn = eval_dataset_fn

    def smooth_l1(self, decoder_pred, decoder_gt):
        l1loss = nn.SmoothL1Loss(reduction='mean')
        return l1loss(decoder_pred, decoder_gt)

    def loss_l1(self, decoder_pred, decoder_gt):
        return nn.L1Loss(reduction='mean')(decoder_pred, decoder_gt)

    def loss_activity(self, logits, class_gt):                                     
        """Computes entropy loss from logits between predictions and class."""
        return nn.functional.cross_entropy(logits, class_gt, reduction='mean')

    def compute_class_loss(self, class_logits, class_gt):
        """Computes the class loss for each of the decoder layers predictions or memory."""
        class_loss = 0.0
        for l in range(len(class_logits)):
            class_loss += self.loss_activity(class_logits[l], class_gt)

        return class_loss/len(class_logits)

    def select_loss_fn(self):
        if self._params['loss_fn'] == 'mse':
            return self.loss_mse
        elif self._params['loss_fn'] == 'smoothl1':
            return self.smooth_l1
        elif self._params['loss_fn'] == 'l1':
            return self.loss_l1
        else:
            raise ValueError('Unknown loss name {}.'.format(self._params['loss_fn']))

    def layerwise_loss_fn(self, decoder_pred, decoder_gt, decoder_pred_traj, decoder_gt_traj, class_logits=None, class_gt=None):
        """Computes layerwise loss between predictions and ground truth."""
        pose_loss = 0.0
        loss_fn = self.select_loss_fn()

        ### pose loss
        for l in range(len(decoder_pred)):
            pose_loss += loss_fn(decoder_pred[l], decoder_gt)

        pose_loss = pose_loss/len(decoder_pred)

        ### trajectory loss
        trajectory_loss = 0
        loss_fn_traj = self.select_loss_fn()
        for l in range(len(decoder_pred_traj)):
            trajectory_loss += loss_fn_traj(decoder_pred_traj[l], decoder_gt_traj)
            
        trajectory_loss = trajectory_loss/len(decoder_pred_traj)

        ### velocity loss
        loss_fn_velocity = self.select_loss_fn()
        velocity_loss = 0
        b,f,j = decoder_pred[-1].shape
        for l in range(len(decoder_pred)):
            velocity_loss += loss_fn_velocity((decoder_pred[0][:,1:f,:]-decoder_pred[0][:,0:f-1,:]) , (decoder_gt[:,1:f,:]-decoder_gt[:,0:f-1,:]))
            
        velocity_loss = velocity_loss/len(decoder_pred)

        if class_logits is not None:
            return pose_loss, self.compute_class_loss(class_logits, class_gt), trajectory_loss , velocity_loss

        return pose_loss, None , trajectory_loss , velocity_loss
    
    def my_layerwise_loss_fn(self, decoder_pred_traj, decoder_gt_traj):
        """Computes layerwise loss between predictions and ground truth."""
        
        ### trajectory loss
        trajectory_loss = 0
        loss_fn_traj = self.select_loss_fn()
        for l in range(len(decoder_pred_traj)):
            trajectory_loss += loss_fn_traj(decoder_pred_traj[l], decoder_gt_traj)
        
        trajectory_loss = trajectory_loss/len(decoder_pred_traj)        

        return trajectory_loss

    def init_model(self, train_dataset_fn, eval_dataset_fn, traj_encoder_fn, traj_decoder_fn):
        self._model = WatchPoseTransformer.model_factory(
            self._params,
            self._fall_options,
            train_dataset_fn,
            eval_dataset_fn,
            traj_encoder_fn, 
            traj_decoder_fn  
        )
    
    def select_optimizer(self):
        optimizer = optim.AdamW(
            self._model.parameters(), lr=self._params['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=self._params['weight_decay']
        )
        return optimizer



def dataset_factory_total(params):
  if params['dataset'] == 'h36m_v3':
    return H36MDataset_v3.dataset_factory_total(params)
  else:
    raise ValueError('Unknown dataset {}'.format(params['dataset']))

def dataset_factory(params):
  if params['dataset'] == 'h36m_v3':
    return H36MDataset_v3.dataset_factory(params)
  else:
    raise ValueError('Unknown dataset {}'.format(params['dataset']))


def load_params():

  params = {}

  params['model_prefix'] = 'trained_model'
  params['batch_size'] = 32
  params['data_path'] ='../data/h3.6m'
  params['learning_rate'] = 0.0001
  params['max_epochs'] = 300
  params['steps_per_epoch'] = 200
  params['action'] =' all' 
  params['use_one_hot'] = False
  params['init_fn'] ='xavier_init'
  params['include_last_obs'] = True
  params['model_dim'] =512 
  params['model_dim_traj'] = 64
  params['num_encoder_layers'] = 4
  params['num_decoder_layers'] =4
  params['num_heads'] = 8
  params['dim_ffn'] = 2048
  params['dropout']= 0.3
  params['source_seq_len'] = 5                  
  params['target_seq_len'] = 20
  params['frame_rate'] = 10
  params['max_gradient_norm'] = 0.1
  params['lr_step_size'] = 200
  params['gamma'] = 0.1 
  params['learning_rate_fn'] = 'step'
  params['warmup_epochs'] = 50
  params['pose_format'] = 'expmap'
  params['remove_low_std'] = False
  params['remove_global_trans'] = False
  params['loss_fn'] = 'mse'
  params['pad_decoder_inputs'] = True
  params['pad_decoder_inputs_mean'] = False
  params['use_wao_amass_joints'] = False
  params['non_autoregressive'] = True
  params['pre_normalization'] = True
  params['use_query_embedding'] = False
  params['predict_activity'] = False
  params['use_memory'] = False
  params['query_selection'] = False
  params['activity_weight'] =1.0
  params['pose_embedding_type'] = 'gcn_enc'
  params['traj_embedding_type'] = 'gcn_enc'
  params['encoder_ckpt'] = None
  params['dataset'] = 'h36m_v3'
  params['skip_rate'] = 5
  params['eval_num_seeds'] = _NSEEDS
  params['copy_method'] = None 
  params['finetuning_ckpt'] = None
  params['pos_enc_alpha'] = 10
  params['pos_enc_beta'] = 500
  params['gt_ratio'] = 0
  params['weight_decay'] = 0.00001
  params['std'] = 0.0005
  params['std_traj'] = 0.0005
  params['noisy'] = False 
  params['heading'] = False

  params['input_dim_traj'] = 3
  params['pose_dim_traj'] = 3

  return params



def main():
    
    params = load_params()

    fall_options = {'num_transformer_blocks':2,
               'd_input':23,
               'd_model':32,
               'num_heads':4,
               'num_layers':2,
               'd_ff':1024,
               'max_seq_length':30,
               'dropout':0.1,
               'inner_df':64,
                }

    # LOAD H36M DATASET FOR TRAINING AND EVALUATION
    train_dataset_fn, eval_dataset_fn = dataset_factory(params)
    _, eval_dataset_fn_total = dataset_factory_total(params)

    params['input_dim'] = train_dataset_fn.dataset._data_dim
    params['pose_dim'] = train_dataset_fn.dataset._pose_dim

    traj_encoder_fn, traj_decoder_fn = \
        PoseEncoderDecoder.select_traj_encoder_decoder_fn(params)

    config_path = os.path.join(params['model_prefix'], 'config', 'config.json')        
    with open(config_path, 'w') as file_:
        json.dump(params, file_, indent=4)

    model_fn = WatchModelFn(
        params, 
        fall_options,
        train_dataset_fn, 
        eval_dataset_fn, 
        eval_dataset_fn_total, 
        traj_encoder_fn, 
        traj_decoder_fn,
    )
    print("MODEL and DATASET loaded...")

    print("Starting TRAINING...")

    model_fn._model._non_autoregressive = True
    model_fn.train()



if __name__ == "__main__":
    main()