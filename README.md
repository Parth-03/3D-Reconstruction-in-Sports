# 3D-Reconstruction-in-Sports
Multi Human 3D Reconstruction from single RGB images for various scenarios in Sports

## Main

### Contributors

## Multi-HMR

### Contributors

Scott Fortune wrote train_panda.ipynb and train_ipynb, except for the functions in cell "Functions from demo.py" which are from the Multi-HMR github.

### To run train_panda.ipynb

- Follow instructions in MulitHMR/README.md to set up environment for the model
- Follow instructions in Panda/Readme.txt to install Panda dataset for training
- Run all cells in notebook

### To run train.ipynb

- Follow instructions in MulitHMR/README.md to set up environment for the model
- Download "Training Images 1280x720 (1)" and "Camera with SMPL-X joints and verts" from https://agora.is.tue.mpg.de/download.php
- Place downloaded dataset and ground truths in "./AGORA" and "./AGORA/SMPLX" respectively
- Run all cells in notebook

### To use fine-tuned model

- load "fine_tuned_multiHMR_986_L.pkl" using pickle.load and replace " model = load_model(...)" *note: don't use load_model() because we couldn't match the fine-tuned model to the same format*
    - "fine_tuned_multiHMR_986_L.pkl" is currently in https://drive.google.com/drive/folders/1Q1aPaijgNfIXs7jdN-q_kdp2Bnh0VMb- since it was too large to add to github

## Group Rec

### Contributors

## Size Depth Disambiguation

### Contributors
