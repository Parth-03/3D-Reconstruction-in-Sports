# 3D-Reconstruction-in-Sports

Multi Human 3D Reconstruction from single RGB images for various scenarios in Sports

## Main

### Contributors

## Multi-HMR

### Contributors

Scott Fortune wrote train_panda.ipynb and train_ipynb, except for the functions in cell "Functions from demo.py" which are from the Multi-HMR github.

### To run train_panda.ipynb

-   Follow instructions in MulitHMR/README.md to set up environment for the model
-   Follow instructions in Panda/Readme.txt to install Panda dataset for training
-   Run all cells in notebook

### To run train.ipynb

-   Follow instructions in MulitHMR/README.md to set up environment for the model
-   Download "Training Images 1280x720 (1)" and "Camera with SMPL-X joints and verts" from https://agora.is.tue.mpg.de/download.php
-   Place downloaded dataset and ground truths in "./AGORA" and "./AGORA/SMPLX" respectively
-   Run all cells in notebook

### To use fine-tuned model

-   load "fine*tuned_multiHMR_986_L.pkl" using pickle.load and replace " model = load_model(...)" \_note: don't use load_model() because we couldn't match the fine-tuned model to the same format*
    -   "fine_tuned_multiHMR_986_L.pkl" is currently in https://drive.google.com/drive/folders/1Q1aPaijgNfIXs7jdN-q_kdp2Bnh0VMb- since it was too large to add to github

## Group Rec

### Contributors

Eugene Mak wrote train.py, added new training function to process.py, added training data class to demo_data.py. Helper methods for aligning humans and SMPL loss based off Multi-HMR version by Scott Fortune.

### To run train.py

-   Follow instructions in GroupRec/README.md to set up environment for the model. You will need the pretrained model and other required data to put in the GroupRec/data folder, found here: https://drive.google.com/drive/folders/1eHi8WBiZsQDh2O5rE3rd_DvQjXJwp-8y?usp=drive_link
-   Follow instructions in Panda/Readme.txt to install Panda dataset for training. Ensure that file structure is as follows: data/Panda/images/Det/[subfolders inside images_train]
-   Recommended: run img_resizer to resize the giant Panda images to 1080p for faster training and processing. Ensure the output is in the location as above.
-   run: python train.py
-   Configure training settings in cfg_files/train.yaml
-   new model will be saved in output/training folder.

### To use fine-tuned model

-   replace 'relation_joint.pkl' with output from training folder and run as before. You may need to rename the new model to 'relation_joint.pkl' to work out of the box.

## Size Depth Disambiguation

### Contributors

Yuni Jeong wrote size_depth_disambiguation/run_sdd.ipynb and modified the optimization code from the Size Depth Disambiguation GitHub to make sdo_run.py, so that the model is compatible with the output of Multi-HMR.

### Using run_size_depth_opt

The function run_size_depth_opt() takes in an array of outputs produced by the Multi-HMR model. It's used in run.py to optimize the size and translation parameters.

### To run the original project demo

To run the demo of the original project demo, upload run_sdd.ipynb as Google Colab and follow the instructions on the notebook.
