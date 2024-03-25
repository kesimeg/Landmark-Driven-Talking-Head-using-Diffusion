# Landmark-Driven-Talking-Head-using-Diffusion

This library implements an image to image diffusion model. The model takes a reference image, landmarks of the reference and landmarks of the target frame. Using this model for each frame a talking head video can be generated. A sample video can be found below. The generated videos using this model are the videos with captions "Random Noise" and "Same Noise". Using the same noise in each frame creates a better temporal continuity.


![](diffusion_gif.gif)

# Prerequisites

diffusers
mediapipe
Pytorch ignite

# Pre Processing

Pre processing has the following steps:
1) You need to reduce fps of videos to 25. (optional)
2) Download and put the reference frame to the same folder 
3) By running data_preprocessing.py you will extract landmarks using mediapipe. This code extracts landmarks and creates a .pt file for each video. You can run the script as the following:
```
python data_preprocessing.py -i "Input data" -o "pre_precessed_data" --input-video-type="mp4" -fps=25 --template-path="reference.jpeg"
```
4) By running img2img_data_pre_processing.py you will down sample the frames in the dataset since. Some frames in the videos look really similar which makes the data redundant and it increases the training time. Not only loading a video as a whole then sampling frames from it slows down the loader. The last two steps could have been done in a single step but due to hardware requirements these had to be seperate. You can run the code as the following:
```
python img2img_data_pre_processing.py --main-dir="pre_precessed_data" --write-dir="sampled_frame_data"
```
# Model training
The training code was adapted from an example code of diffusers library (https://github.com/huggingface/diffusers/tree/main/examples/unconditional_image_generation). Every n epochs it calculates SSIM between generated and real frames. You can track it using the tensorboard. The initial code of diffusers examples had lots of flexibity such as using weights&biases etc. Since I did not use most of them some of the options might be broken due since they are not tested after the code was changed. You can run the code in the following way (which is ensured to be run ;) ):
```
!accelerate launch diffusion_talking_head_train.py \
  --dataset_name="sampled_frame_data" \
  --eval_batch_size=8\
  --resolution=128\
  --output_dir="output" \
  --train_batch_size=64 \
  --num_epochs=11 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=0 \
  --lr_scheduler="constant"\
  --ddim_num_inference_steps=50 \
  --mixed_precision=no\
  --save_images_epochs=1\
  --checkpointing_steps=2572\
  --eval_model_epochs=5\
  --checkpoints_total_limit=3
```
# Video generation
To generate videos you can use the code : video_diffusion_talking_head.py . It can either sample a random pre processed video from the dataset (from pre_precessed_data) or you can select a pre processed video (again from pre_precessed_data) yourself and generate a video. This code only takes videos that had been preprocessed. You can run the code in the following way:
```
python "video_diffusion_talking_head.py" \
  --dataset_name="pre_precessed_data" \
  --checkpoint_dir="checkpoint dir" \
  --write_real_video \
  --write_landmark_video
```
# Acknowledgment

I want to thank to the resources below:
diffusers : https://github.com/huggingface/diffusers

simplified mediapipe landmarks: https://github.com/k-m-irfan/simplified_mediapipe_face_landmarks

The landmark extraction code was heavily inspired by : https://github.com/eeskimez/emotalkingface