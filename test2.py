import cv2
import os
import numpy as np
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
directory_path = os.getcwd()

HISTO_ROWS = 5

def color_filter(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  kernel_size = 5
  blur = cv2.blur(hsv, (kernel_size, kernel_size), 0)

  black_low = np.array([0, 0, 0])
  black_up = np.array([360, 255, 70])

  mask = cv2.inRange(blur, black_low, black_up)

  return cv2.bitwise_not(mask)

def canny_filter(img):
  blurred_img = cv2.blur(img,ksize=(5,5))
  med_val = np.median(blurred_img) 
  low_thres = int(max(0 ,0.5*med_val))
  high_thres = int(min(255,2.0*med_val))
  print("l: " + str(low_thres) + " h: " + str(high_thres) + " med: " + str(med_val))
  edges = cv2.Canny(img, low_thres, high_thres)

  return edges

def run_cv():
  
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
  video = cv2.VideoWriter('video.mp4', fourcc, 30, (640, 480))
  camera = PiCamera()
  rawCapture = PiRGBArray(camera)
  camera.resolution = (640, 480)
  camera.framerate = 30
  time.sleep(0.1)
  for test_image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # ret, test_image = cap.read()
    test_image = cv2.resize(test_image, (640, 480))
    
    height = test_image.shape[0]
    width = test_image.shape[1]
    height_thresh = height/HISTO_ROWS

    filtered = color_filter(test_image)


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

    # print(average_locations)


    # for row in range(360):
    #   for col in range(480):
    #     # if filtered[row][col] == 255:
    #       # num_elements = histo_arr[int(row/height_thresh)][1]
    #       # histo_arr[int(row/height_thresh)][0] = (histo_arr[int(row/height_thresh)][0]*num_elements + col)/(num_elements + 1)
    #       # histo_arr[int(row/height_thresh)][1] += 1


    for index in range(len(average_locations)):
      cords = average_locations[index]
      cv2.circle(test_image, (int(cords[0]), int(cords[1])), 5, (0, 255, 0), -1)
      if index > 0:
        cords_old = average_locations[index-1]
        cv2.line(test_image, (int(cords[0]), int(cords[1])), (int(cords_old[0]), int(cords_old[1])), (255, 0, 0), 3)

    # draw_lines_all(test_image, lines)
    cv2.imshow('image', test_image)
    video.write(test_image)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  
  # release the video capture object
  cap.release()
  # Closes all the windows currently opened.
  cv2.destroyAllWindows()
  video.release()


if __name__ == "__main__":
  # pipeline = cv2.VideoCapture(directory_path + '\line.mp4')
  run_cv()