This repository is for exploring and implementing various lip reading papers.

1. original.py
   - This is the PyTorch version of this video(https://www.youtube.com/watch?v=uKyojQjbx4c&t=3834s).
   - The main changes are
     - 1) unlike in the video, where only part of the grid dataset was used, this version uses all of it.
     - 2) It additionally applies elements like gradient accumulation and wandb.
