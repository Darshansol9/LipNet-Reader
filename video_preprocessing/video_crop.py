#from skvideo.io import vread
import cv2
import numpy as np
#from google.colab.patches import cv2_imshow
import dlib

face_detector_path = '/content/drive/My Drive/shape_predictor_68_face_landmarks.dat'

def readVideo(path):

  '''
  processing video frames and return the processed frames to the function call
  '''
    
  cap = cv2.VideoCapture(path)
  if (cap.isOpened()== False): 
    print("Error opening video file")
    
  frames = []
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
      frames.append(frame)
    else: 
      break
  cap.release()
  cv2.destroyAllWindows()

  frames = np.array(frames)
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(face_detector_path)
  mouth_video_frames = get_frames_mouth(detector, predictor, frames)
  #showMouthFrames(mouth_video_frames)
  
  return mouth_video_frames


def get_frames_mouth(detector, predictor, frames):
  '''
    Cropping images for mouth region and making the img frame size as (45 x 70 x 3)
  '''

  MAX_WIDTH = 45
  MAX_HEIGHT = 70
  CHANNEL = 3

  mouth_frames = []
  for frame in frames:
    dets = detector(frame,1)
    for i,d in enumerate(dets):
      #print("Detection Left: {} Top: {} Right: {} Bottom: {}".format(d.left(), d.top(), d.right(), d.bottom()))
      shape = predictor(frame, d)
      i += 1
      xmouthpoints = [shape.part(x).x for x in range(48,67)]
      ymouthpoints = [shape.part(x).y for x in range(48,67)]

      #Taking the mouth points min and max (x,y) and add pad 

      maxx = max(xmouthpoints)
      minx = min(xmouthpoints)
      maxy = max(ymouthpoints)
      miny = min(ymouthpoints) 
      pad = 12
      crop_image = frame[miny-pad:maxy+pad,minx-pad:maxx+pad]
      shape = np.shape(crop_image)
      cropped_padded = np.zeros((MAX_WIDTH, MAX_HEIGHT,CHANNEL))
      cropped_padded[:shape[0],:shape[1]] = crop_image

      mouth_frames.append(padded_array)
  
  return mouth_frames


def showMouthFrames(mouth_video):
    
    j = 35
    for i in range(5):
      img1 = cv2.hconcat(mouth_video[j:j+5])
      cv2.imshow("frames",img1)
      j = j + 5
