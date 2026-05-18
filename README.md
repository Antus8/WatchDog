# Robotic-Dog for Elderly Assistance via Transformer-based Fall Detection and Motion Prediction
AI-powered legged robotic platform for real-time fall detection and motion prediction in cluttered environments. 

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://antus8.github.io/WatchDog/) [![Paper](https://img.shields.io/badge/Paper-PDF-red)](#)


![Description](images/concept.png)

## Code flow

Fall transformer --> BodyClassificationNetwork (model.py) , trainer.py starts training and saves weights, inference.py performs inference
WatchDog transformer:
	- transformer_model_fn.py starts the full training pipeline
	- ModelFn handles the full training cycle (load sample, go through model, get loss, update weights)
	- The model is created in WatchPoseTransformer.model_factory
		- The actual model is a WatchPoseTransformer (from WatchPoseTransformer.py)
			- self.fall_transformer = BodyClassificationNetwork 
			- Inside there is a self._transformer = Transformer (from Transformer.py -actual TRAJECTORY-ONLY transformer architecture)
			- forward_training (goes through the whole transformer and gets predictions)

### To train the model
- Use transformer_model_fn.py script --> create a WatchModelFn to handle the cycle

/ '''python
    /model_fn = WatchModelFn(
        /params, train_dataset_fn, 
        /eval_dataset_fn, 
        /eval_dataset_fn_total,
        /traj_encoder_fn, 
        /traj_decoder_fn,
    /)

    /model_fn.train()'''
    
- WatchModelFn creates the full architecture that is a WatchPoseTransformer from WatchPoseTransformer.py. It is composed of 2 branches:
  - BodyClassificationNetwork that is the fall detection transformer, stored inside self.fall_transformer (from model.py)
  - Transformer that is the trajectory prediction transformer, stored inside self._transformer (from Transformer.py)


## Installation
Coming soon...

## Data
Coming soon...

## ROS
Coming soon...