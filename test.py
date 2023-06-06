import cv2
import os
import numpy as np
import math
directory_path = os.getcwd()

HISTO_ROWS = 5

def draw_lines_all(img, lines, color=[0, 0, 255], thickness=3):
  try:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
  except:
    print('empty')

def color_filter(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  kernel_size = 5
  blur = cv2.blur(hsv, (kernel_size, kernel_size), 0)

  black_low = np.array([0, 0, 0])
  black_up = np.array([360, 255, 60])

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

def run_cv(cap):
  
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
  video = cv2.VideoWriter('video.mp4', fourcc, 1, (480, 360))

  while (cap.isOpened()):

    ret, test_image = cap.read()
    test_image = cv2.resize(test_image, (480, 360))
    
    height = test_image.shape[0]
    height_thresh = height/HISTO_ROWS

    filtered = color_filter(test_image)

    cannyed = canny_filter(filtered)

    blur = cv2.blur(cannyed, (5, 5), 0)

    rho = 3
    theta = np.pi / 180
    threshold = 80
    min_line_len = 20
    max_line_gap = 10
    lines = cv2.HoughLinesP(blur, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    # make a two dimensional np array of zeroes where the first array is of size HISTO_ROWS, and the second of size 2
    histo_arr = np.zeros((HISTO_ROWS, 2), dtype=np.int32)

    for line in lines:
      for x1, y1, x2, y2 in line:
        y1 = math.floor(y1/height_thresh)
        y2 = math.floor(y2/height_thresh)
        histo_arr[y1][0] = (x1 + histo_arr[y1][0]*histo_arr[y1][1])/(histo_arr[y1][1] + 1)
        histo_arr[y1][1] += 1
        
        histo_arr[y2][0] = (x2 + histo_arr[y2][0]*histo_arr[y2][1])/(histo_arr[y2][1] + 1)
        histo_arr[y2][1] += 1

    print(histo_arr)

    for index in range(len(histo_arr)):
      row = histo_arr[index]
      cv2.circle(test_image, (int(row[0]), int((index+0.5)*height_thresh)), 5, (0, 255, 0), -1)
      if index > 0:
        cv2.line(test_image, (int(row[0]), int((index+0.5)*height_thresh)), (int(histo_arr[index-1][0]), int((index-0.5)*height_thresh)), (255, 0, 0), 3)

    draw_lines_all(test_image, lines)
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
  pipeline = cv2.VideoCapture(directory_path + '/line.mp4')
  run_cv(pipeline)