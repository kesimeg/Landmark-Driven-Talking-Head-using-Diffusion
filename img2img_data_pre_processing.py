import torch
import os
from tqdm import tqdm
import argparse
import numpy as np


parser = argparse.ArgumentParser()


parser.add_argument(
    "--main-dir",
    help="Main dir to read preprocessed video data",
)
parser.add_argument(
    "--write-dir",
    help="The name of the direcory to write sampled dataset",
)

args = parser.parse_args()

main_dir = args.main_dir
write_dir = args.write_dir

test_id_set = set(["1015", "1020", "1021", "1030", "1033", "1052", "1062", "1081", "1082", "1089"])
validation_id_set = set(["1003", "1019", "1023", "1024", "1050", "1056", "1058", "1071","1073", "1074"])
train_id_set = set(np.arange(1001,1092).astype(str))

train_id_set -= test_id_set
train_id_set -= validation_id_set

file_list = os.listdir(main_dir)

def sample_video_frames(set_ids,start_frame,end_frame,sampling):
  if not os.path.exists(write_dir):
    os.mkdir(write_dir)

  for file in tqdm(file_list):
    folder_name = file.replace(".pt","")
    dir_name = os.path.join(write_dir,folder_name)

    person_id = folder_name.split("_")[0]

    if person_id not in set_ids:
      continue

    if not os.path.exists(dir_name):
      os.mkdir(dir_name)

    torch_file = torch.load(os.path.join(main_dir,file))
    video_len = torch_file["video"].shape[0]

    for i in range(start_frame,video_len - end_frame,sampling):
      frame = torch_file["video"][i]
      landmark = torch_file["landmark_figure"][i]
      save_path = os.path.join(dir_name,"frame_{}.pt".format(i))
      torch.save({"video":frame,"landmark":landmark},save_path)

    reference_vid_frame = torch_file["video"][0]
    reference_landmark_frame = torch_file["landmark_figure"][0]

    save_path = os.path.join(dir_name,"reference.pt")
    torch.save({"video":reference_vid_frame,"landmark":reference_landmark_frame},save_path)


#for validation we can decrease the number of frames especially initial and final frames since the movement is not much
sample_video_frames(validation_id_set,start_frame=10,end_frame=10,sampling=3)

#for train set we can use more frames compared to validation since train set determines the performance of the model
sample_video_frames(train_id_set,start_frame=5,end_frame=5,sampling=2)

#since this is done to decrease the amount of data to train and validate model we don't need to sample from test set


