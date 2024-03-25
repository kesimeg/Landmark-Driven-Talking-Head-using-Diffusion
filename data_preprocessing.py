import argparse
import os
from skimage import transform as tf
import numpy as np
import cv2
from tqdm import tqdm
import torch
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


parser = argparse.ArgumentParser()

parser.add_argument(
    "--template-path",
    type=str,
    help="Template face image path",
    default="reference.jpeg",
)
parser.add_argument(
    "-i",
    "--input-dir",
    type=str,
    help="Input directory path",
)
parser.add_argument(
    "-o",
    "--output-dir",
    type=str,
    help="Output directory path",
)
parser.add_argument(
    "--exception-file",
    type=str,
    help="Exception file name",
    default="exceptions.txt",
)
parser.add_argument(
    "-fps",
    type=int,
    help="frame rate of video",
    default=25,
)

parser.add_argument(
    "--input-video-type",
    type=str,
    help="input video format",
    default="mp4",
)

parser.add_argument(
    "--no-landmarks",
    type=bool,
    help="Exclude landmark extraction",
    default=False,
)
parser.add_argument(
    "--overwrite",
    type=bool,
    help="If true overwrites existing data, if true skips that file",
    default=False,
)
parser.add_argument(
    "--test-video-output",
    type=bool,
    help="If true also writes video",
    default=False,
)

def write_video(video_name,data):
    frame_len = data.shape[0]
    frame_width = int(data.shape[1])
    frame_height = int(data.shape[2])

    out_video = cv2.VideoWriter('{}_.avi'.format(video_name),
                                cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_height,frame_width))

    for i in range(frame_len):
        frame = data[i]
        out_video.write(frame)
    out_video.release()

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_list.append(frame)
        else:
            break
    cap.release()
    return frame_list


def detect_landmark(img,size):
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
  detection_result = detector.detect(mp_image).face_landmarks

  if not detection_result:
    return None
  else:
    detection_result = detection_result[0]

  landmark_array = np.zeros((len(all_landmark_coords),2))
  for i in range(0,len(detection_result)):

    if i in all_landmark_coords:
      x = int(detection_result[i].x * size[1])
      y = int(detection_result[i].y * size[0])

      index = landmark_coords_dict[i]
      landmark_array[index] = [x,y]

  return landmark_array


# finds mean points of the left eye,right eye, and nose
def get_parts(landmark_array):

    left_eye = landmark_array[20:36].mean(axis=0)
    right_eye = landmark_array[36:52].mean(axis=0)
    nose = landmark_array[128:135].mean(axis=0)

    points = np.stack([left_eye, right_eye, nose])

    return points


# draws a set of lines and connects dots between points in a list
def draw_line(image, color, thickness, coord_list, connect_start_end=False):
    for i in range(1, len(coord_list)):
        image = cv2.line(
            image, tuple(coord_list[i - 1]), tuple(coord_list[i]), color, thickness
        )

    # whether first and last points should be connected (used for eye,lips and teeth)
    if connect_start_end:
        image = cv2.line(
            image, tuple(coord_list[0]), tuple(coord_list[-1]), color, thickness
        )

    return image


# draws an entire face using landmarks and connecting them
def draw_face(preds, size, thickness=1,image=None):
    if image is None:
        image = np.zeros(size)
    image = draw_line(image, (0, 0, 255), thickness, preds[92:128, :],connect_start_end=True)  # face
    
    image = draw_line(image, (255, 255, 0), thickness, preds[0:10, :], connect_start_end=True)  # eye_brow1
    image = draw_line(image, (0, 255, 255), thickness, preds[10:19, :], connect_start_end=True)  # eye_brow2
    image = draw_line(
        image, (128, 0, 255), thickness, preds[20:36, :], connect_start_end=True
    )  # eye_1

    image = draw_line(
        image, (255, 128, 0), thickness, preds[36:52, :], connect_start_end=True
    )  # eye_2
    image = draw_line(
        image, (0, 255, 0), thickness, preds[72:92, :], connect_start_end=True
    )  # outer lips
    image = draw_line(
        image, (255, 255, 255), thickness, preds[52:72, :], connect_start_end=True
    )  # inner lips
    image = draw_line(image, (255, 0, 0), thickness, preds[128:135, :])  # nose

    image = image.astype("uint8")
    return image

def draw_points(preds,size,image=None):
    if image is None:
        image = np.zeros(size)
    for i in range(len(preds)):
        image = cv2.circle(image, (preds[i,0],preds[i,1]), radius=1, color=(0, 255, 0), thickness=-1)

    image = image.astype("uint8")
    return image

args = parser.parse_args()

template_path = args.template_path
input_dir = args.input_dir
output_dir = args.output_dir
exception_file = args.exception_file
fps = args.fps
input_video_type = args.input_video_type
no_landmarks = args.no_landmarks
overwrite = args.overwrite
test_video_output = args.test_video_output

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# list of corrupt files
corrupt_file_list = [
    "1076_MTI_NEU_XX.mp4",
    "1076_MTI_SAD_XX.mp4",
    "1064_TIE_SAD_XX.mp4",
    "1064_IEO_DIS_MD.mp4",
]

# Coordinates are thanks to :https://github.com/k-m-irfan/simplified_mediapipe_face_landmarks
Left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
Right_eyebrow = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
Left_eye = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
Right_eye = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
Inner_Lip = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
Outer_Lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
Face_Boundary = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
Nose = [168,6,197,195,5,4,1]

all_landmark_coords = Left_eyebrow + Right_eyebrow + Left_eye + Right_eye + Inner_Lip + Outer_Lip + Face_Boundary + Nose
landmark_coords_dict = dict(zip(all_landmark_coords,range(len(all_landmark_coords))))


# read the template image and detect landmarks
template_img = cv2.imread(template_path)
template_img = cv2.resize(template_img, (128, 128))
landmark_array = detect_landmark(template_img,(128, 128))

# scale the template image
min_val = landmark_array.min(axis=0)
max_val = landmark_array.max(axis=0)
norm_landmark_array = (landmark_array - min_val) / (max_val - min_val)
scaled_landmark_arr = norm_landmark_array * 80 + np.array([25, 25])
warp_points = get_parts(scaled_landmark_arr)


files = os.listdir(input_dir)

video_data_dict = {}

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# iterate overfiles and align faces
for file in tqdm(files):

    if file in corrupt_file_list:
        continue

    video_path = os.path.join(input_dir, file)
    file_name = os.path.join(output_dir, file.replace(input_video_type, "pt"))

    if not overwrite and os.path.exists(file_name):
        continue

    try:
        video_frames = load_video_frames(video_path)

        landmark_array = detect_landmark(video_frames[0],video_frames[0].shape[:2])
        warp_points_2 = get_parts(landmark_array)
        tform = tf.estimate_transform(
            "similarity", warp_points, warp_points_2
        )  # find the transformation matrix
        warped_frame_list = []
        landmark_list = []
        landmark_figure_list = []

        for frame in video_frames:
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # turn images into RGB
            warped_frame = (
                tf.warp(color_frame, tform, output_shape=(128, 128)) * 255
            )  # warp the frame according to transformation
            warped_frame = warped_frame.astype("uint8")  # use uint8 to reduce file size
            warped_frame_list.append(warped_frame)
   
            temp_landmark = detect_landmark(warped_frame,(128,128))

            if temp_landmark is not None:
                landmark = temp_landmark.copy()
            else:
                landmark = np.full((135,2),np.nan) # if face not detected fill with nan

            if not no_landmarks:
                landmark_list.append(landmark)
  
       
        landmark_array = np.array(landmark_list)
        for i in range(landmark_array.shape[-1]): 
            for j in range(0,landmark_array.shape[1]):
                
                ind = np.arange(landmark_array.shape[0])
                good = np.where(~np.isnan(landmark_array[:,j,i]))[0]
                f = interpolate.interp1d(ind[good],landmark_array[:,j,i][good]) # interpolate nan values
                landmark_array[:,j,i] = f(ind)
                landmark_array[:,j,i] = gaussian_filter1d(landmark_array[:,j,i], sigma=0.1)# filter each coordinate to reduce shaking


        for i in range(landmark_array.shape[0]): # filter each coordinate
            landmark_figure = draw_face(landmark_array[i].astype(int), size=(128, 128, 3))#, image = warped_frame_list[i]) # uncomment to draw landmarks on images
            landmark_figure = cv2.cvtColor(landmark_figure, cv2.COLOR_BGR2RGB)
            landmark_figure_list.append(landmark_figure)

        warped_frame_array = np.array(warped_frame_list)

        video_data_dict = {}
        video_data_dict["video"] = warped_frame_array

        if not no_landmarks:
            landmark_array = np.array(landmark_list)
            landmark_figure_array = np.array(landmark_figure_list)
            video_data_dict["landmark"] = landmark_array
            video_data_dict["landmark_figure"] = landmark_figure_array

        torch.save(
            video_data_dict,
            file_name,
        )

        if test_video_output:
            write_video(file_name,landmark_figure_array)

    except Exception as e:
        print(file)  # Print file name and exception as extra caution
        print(e)
        with open(exception_file, "a") as output_file:
            output_file.writelines(file + "\t" + str(e) + "\n")
