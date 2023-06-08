import cv2
import os
import numpy as np
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
directory_path = os.getcwd()

HISTO_ROWS = 5

def color_filter(img):
  kernel_size = 7
  blur = cv2.blur(img, (kernel_size, kernel_size), 0)

  cv2.imshow('blur', blur)

  b_low = np.array([0, 0, 0])
  b_up = np.array([110, 110, 110])

  mask = cv2.inRange(blur, b_low, b_up)

  return cv2.bitwise_not(mask)

def run_cv():
  
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
  video = cv2.VideoWriter('video.mp4', fourcc, 30, (640, 480))
  camera = PiCamera()
  camera.rotation = 180
  camera.resolution = (640, 480)
  camera.awb_mode = 'fluorescent'
  camera.awb_gains = 4
  camera.exposure_mode = 'off'
  rawCapture = PiRGBArray(camera, size=(640, 480))
  time.sleep(0.1)
  for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    test_image = frame.array
    height = test_image.shape[0]
    width = test_image.shape[1]
    height_thresh = height/HISTO_ROWS

    
    pt_a_x = 2.25*(test_image.shape[1]/7)
    pt_a_y = 7.5*(test_image.shape[0]/10)
    pt_b_x = 1.25*(test_image.shape[1]/7)
    pt_b_y = 10*(test_image.shape[0]/10)
    pt_c_x = 5.75*(test_image.shape[1]/7)
    pt_c_y = 10*(test_image.shape[0]/10)
    pt_d_x = 4.75*(test_image.shape[1]/7)
    pt_d_y = 7.5*(test_image.shape[0]/10)

    width_AD = np.sqrt(((pt_a_x - pt_d_x) ** 2) + ((pt_a_y - pt_d_y) ** 2))
    width_BC = np.sqrt(((pt_b_x - pt_c_x) ** 2) + ((pt_b_y - pt_c_y) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_a_x - pt_b_x) ** 2) + ((pt_a_y - pt_b_y) ** 2))
    height_CD = np.sqrt(((pt_c_x - pt_d_x) ** 2) + ((pt_c_y - pt_d_y) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([[pt_a_x, pt_a_y], [pt_b_x, pt_b_y], [pt_c_x, pt_c_y], [pt_d_x, pt_d_y]])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])

    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    roi_transform = cv2.warpPerspective(test_image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    test_image = cv2.resize(roi_transform, (640, 480))

    filtered = color_filter(test_image)

    cv2.imshow('filtered', filtered)

    average_locations = []

    # Define the desired height for rows
    x = 50

    for y in range(0, filtered.shape[0], x):
        row = filtered[y:y+x, :]
        black_pixels = np.where(row == 0)

        if len(black_pixels[0]) > 0:
            avg_location = (
                int(np.mean(black_pixels[1])),  # Average column
                int(y + np.mean(black_pixels[0]))  # Average row
            )
            average_locations.append(avg_location)

    for index in range(len(average_locations)):
      cords = average_locations[index]
      cv2.circle(test_image, (int(cords[0]), int(cords[1])), 5, (0, 255, 0), -1)
      if index > 0:
        cords_old = average_locations[index-1]
        cv2.line(test_image, (int(cords[0]), int(cords[1])), (int(cords_old[0]), int(cords_old[1])), (255, 0, 0), 3)

    cv2.imshow('image', test_image)
    rawCapture.truncate(0)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  
  # release the video capture object
  cap.release()
  # Closes all the windows currently opened.
  cv2.destroyAllWindows()
  #video.release()


if __name__ == "__main__":
  # pipeline = cv2.VideoCapture(directory_path + '\line.mp4')
  run_cv()