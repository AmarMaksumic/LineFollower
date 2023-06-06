import cv2
import os
import numpy as np
directory_path = os.getcwd()


def draw_lines_all(img, lines, color=[0, 0, 255], thickness=7):
  try:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
  except:
    print('empty')

def color_filter(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # cv2.imshow('hsv', hsv)

  kernel_size = 5
  blur = cv2.blur(hsv, (kernel_size, kernel_size), 0)

  # cv2.imshow('gaus. blur', blur)

  white_low = np.array([0, 0, 140])
  white_up = np.array([179, 185, 255])
  yellow_low = np.array([120, 0, 0])
  yellow_up = np.array([120, 0, 0])

  mask_1 = cv2.inRange(blur, white_low, white_up)
  mask_2 = cv2.inRange(blur, yellow_low, yellow_up)

  mask = mask_1 | mask_2

  return mask

def canny_filter(img):

  low_thres = 60
  high_thres = 175
  edges = cv2.Canny(img, low_thres, high_thres)

  return edges

def run_cv(cap):
  while (cap.isOpened()):
  
    # Capture frame-by-frame
    ret, frame = cap.read()

    # cv2.imshow('frame', frame)

    # transform to ROI
    pt_a_x = 2.25*(frame.shape[1]/7)
    pt_a_y = 7.5*(frame.shape[0]/10)
    pt_b_x = 1.25*(frame.shape[1]/7)
    pt_b_y = 10*(frame.shape[0]/10)
    pt_c_x = 5.75*(frame.shape[1]/7)
    pt_c_y = 10*(frame.shape[0]/10)
    pt_d_x = 4.75*(frame.shape[1]/7)
    pt_d_y = 7.5*(frame.shape[0]/10)

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
    roi_transform = cv2.warpPerspective(frame, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    roi_rescale = cv2.resize(roi_transform, (400, 500))

    filtered = color_filter(roi_rescale)

    kernel_size = 5
    dilated = cv2.dilate(filtered, (kernel_size, kernel_size), iterations=9)

    cannyed = canny_filter(roi_rescale)

    cv2.imshow('filtered', filtered)
    cv2.imshow('dilated', dilated)
    cv2.imshow('cannyed', cannyed)

    combined = cv2.bitwise_and(dilated, cannyed)
    
    cv2.imshow('combined', combined)


    rho = 3
    theta = np.pi / 180
    threshold = 80
    min_line_len = 40
    max_line_gap = 10
    lines = cv2.HoughLinesP(combined, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    blank_image = np.zeros((roi_rescale.shape[0], roi_rescale.shape[1], 4), np.uint8)

    draw_lines_all(blank_image, lines)
    cv2.imshow('image', blank_image)

    pts_src = np.array([[0, 0], [0, blank_image.shape[0]], [blank_image.shape[1], blank_image.shape[0]], [blank_image.shape[1], 0]])
    pts_dst = np.array([[pt_a_x, pt_a_y], [pt_b_x, pt_b_y], [pt_c_x, pt_c_y], [pt_d_x, pt_d_y]])

    h, status = cv2.findHomography(pts_src, pts_dst)

    im_out = cv2.warpPerspective(blank_image, h, (frame.shape[1], frame.shape[0]))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    im_out = cv2.addWeighted(frame, 1, im_out, 1, 0.0)

    # # Make a True/False mask of pixels whose BGR values sum to more than zero
    # alpha = np.sum(im_out, axis=-1) > 0

    # # Convert True/False to 0/255 and change type to "uint8" to match "na"
    # alpha = np.uint8(alpha * 255)

    # # Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
    # res = np.dstack((im_out, alpha))

    # frame[0:frame.shape[0], 0:frame.shape[1]] = im_out

    # alpha_s = im_out[:, :, 3] / 255.0
    # alpha_l = 1.0 - alpha_s

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    # print(im_out.shape[1])

    # for c in range(0, 3):
    #   frame[0:frame.shape[0], 0:frame.shape[1], c] = (alpha_s * im_out[:, :, c] +
    #                                                   alpha_l * frame[0:frame.shape[0], 0:frame.shape[1], c])

    cv2.imshow('im_outImage.jpg', im_out)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  
  # release the video capture object
  cap.release()
  # Closes all the windows currently opened.
  cv2.destroyAllWindows()

if __name__ == "__main__":
  pipeline = cv2.VideoCapture(0)
  run_cv(pipeline)