import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
from torchvision import transforms
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel,  DDIMPipeline, DDIMScheduler,AutoencoderKL,UNet2DConditionModel
import torchvision
from PIL import Image
import numpy as np
from pipeline_conditional_diffusion import ConditionalPipeline
import cv2
import argparse



parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    help=(
        "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
        " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
        " or to a folder containing files that HF Datasets can understand."
    ),
)
parser.add_argument(
    "--load_video_name",
    type=str,
    help="To load a specific preprocessed video write the video's name (must be a .pt file in the dataset), ex: 1082_ITS_HAP_XX.pt",
    default=None,
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="The path of the checkpoint file",
)
parser.add_argument(
    "--write_real_video",
    action=argparse.BooleanOptionalAction,
    help="Whether to write real video along with generated",
)
parser.add_argument(
    "--write_landmark_video",
    action=argparse.BooleanOptionalAction,
    help="Whether to write landmark video along with generated",
)

class Tensor_dataset(Dataset):
    def __init__(self, root_dir,transform,identity_set,select_every=1):

        self.root_dir = root_dir
        self.folder_list = os.listdir(self.root_dir)
        self.identity_set = identity_set
        self.transform = transform
        self.select_every = select_every
        self.data_list = []
        self.reference_data = {}

        for folder in tqdm(self.folder_list):

          person_id = folder.split("_")[0]
          if person_id not in self.identity_set:
            continue

          self.data_list += [os.path.join(folder)]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_path = self.data_list[idx]
        folder = file_path.split("/")[0]

        data = torch.load(os.path.join(self.root_dir,file_path))


        target_frame_list = []
        target_landmark_list = []

        video_len = data["video"].shape[0]

        for i in range(0,video_len):# we also add first frame here
          target_frame_list.append(self.transform(data["video"][i]/255))
          target_landmark_list.append(self.transform(data["landmark_figure"][i]/255))

        reference_frame = self.transform(data["video"][0]/255)
        reference_landmark = self.transform(data["landmark_figure"][0]/255)

        reference_frame = reference_frame.unsqueeze(0).repeat(video_len,1,1,1)
        reference_landmark = reference_landmark.unsqueeze(0).repeat(video_len,1,1,1)

        target_frame_tens = torch.stack(target_frame_list)
        target_landmark_tens = torch.stack(target_landmark_list)

        condition = [reference_frame,target_landmark_tens,reference_landmark]
        condition = torch.cat(condition,axis=1)

        return {"input":condition,"target":target_frame_tens,"video_name":folder}

def write_video(frame_tens,file_name,size):
  result = cv2.VideoWriter('{}.mp4'.format(file_name),
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          25, size)
  for frame in frame_tens:
    result.write(frame)
  result.release()


transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
)

validation_id_set = set(["1003", "1019", "1023", "1024", "1050", "1056", "1058", "1071","1073", "1074"])
test_id_set = set(["1015", "1020", "1021", "1030", "1033", "1052", "1062", "1081", "1082", "1089"])
train_id_set = set(np.arange(1001,1092).astype(str))

train_id_set -= test_id_set
train_id_set -= validation_id_set



args = parser.parse_args()

dataset_name = args.dataset_name
load_video_name = args.load_video_name
checkpoint_dir = args.checkpoint_dir
write_real_video = args.write_real_video
write_landmark_video = args.write_landmark_video


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
) 



dataset = Tensor_dataset(dataset_name,transform=transform,identity_set = test_id_set)

test_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

if load_video_name:
    data = torch.load(os.path.join(dataset_name,load_video_name))

    target_frame_list = []
    target_landmark_list = []

    video_len = data["video"].shape[0]

    for i in range(0,video_len):# we also add first frame here
      target_frame_list.append(transform(data["video"][i]/255))
      target_landmark_list.append(transform(data["landmark_figure"][i]/255))

    reference_frame = transform(data["video"][0]/255)
    reference_landmark = transform(data["landmark_figure"][0]/255)

    reference_frame = reference_frame.unsqueeze(0).repeat(video_len,1,1,1)
    reference_landmark = reference_landmark.unsqueeze(0).repeat(video_len,1,1,1)

    target_frame_tens = torch.stack(target_frame_list)
    target_landmark_tens = torch.stack(target_landmark_list)

    condition = [reference_frame,target_landmark_tens,reference_landmark]
    condition = torch.cat(condition,axis=1).float()
    video_name = load_video_name.replace(".pt","")

    target = target_frame_tens
else:
    sample = next(iter(test_dataloader))
    condition = sample["input"].float()[0]
    video_name = sample["video_name"][0].replace(".pt","")
    target = sample["target"].float()[0]

unet = UNet2DModel.from_pretrained(checkpoint_dir, subfolder="unet")
unet.to(device)
noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
 )

input_cond = condition.to(device).float()

pipeline = ConditionalPipeline(
    unet=unet,
    scheduler=noise_scheduler,
)
pipeline.to(device)
generator = torch.Generator(device=device).manual_seed(1)

input_cond = condition.to(device)



frames = pipeline(
  generator=generator,
  input_cond=input_cond,
  batch_size=input_cond.size(0),
  num_inference_steps=50,
  output_type="tensor",
  same_noise_mode = True,
  disable_tqdm=False,
).images



permute = [2, 1, 0] # we need to change channels due to channel order opencv expects
size = (128, 128)

landmarks = condition.to(device).float()[:,3:6,:,:]

denormed_img = 255 * frames

denormed_img = denormed_img[:, permute]

denormed_img = denormed_img.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
write_video(denormed_img,video_name + "generated",(128,128))

if write_real_video:
    real_vid = 255 * (target[:, permute] + 1)/2
    real_vid = real_vid.permute(0,2,3,1).cpu().numpy().astype(np.uint8)

    write_video(real_vid,video_name +"real",size)

if write_landmark_video:
    real_landmark = 255 * (landmarks[:, permute] + 1)/2

    real_landmark = real_landmark.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
    write_video(real_landmark,video_name + "landmark",size)
