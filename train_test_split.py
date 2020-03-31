import os, shutil, numpy as np
files = os.listdir("train_videos")
files = np.random.choice(files, 600, replace = False)
for file in files:
        shutil.move("train_videos/"+file, "test_videos")