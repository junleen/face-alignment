# -*- coding: utf-8 -*-
import numpy as np
import cv2

# landmarks line mask
def generate_mask(image, landmarks):
    '''
    generate face mask according to landmarks
    Args:
        image: numpy.ndarray
        landmarks: 68x2 numpy.ndarray
    Return:
        a mask map with

    '''

    # layer1: line
    # layer2: region without expansion
    # layer3: wider mask
    linemask = generate_line_mask(image, landmarks)
    regionmask = generate_region_mask(image, landmarks)
    widermask = generate_wider_mask(image, landmarks)
    mask = np.stack([linemask, regionmask, widermask]).transpose(1, 2, 0)

    # return channel: BGR(linemask, regionmask, widermask)channel0:
    return mask

def generate_line_mask(image, landmarks):
    linemask = image.copy() # np.zeros_like(image).astype(np.uint8)
    # face
    linemask = connect_line(linemask, landmarks[0:17])

    # eyebow
    linemask = connect_line(linemask, landmarks[17:22])
    linemask = connect_line(linemask, landmarks[22:27])

    # nose
    linemask = connect_line(linemask, np.vstack([landmarks[27:31], landmarks[33]]))
    linemask = connect_line(linemask, landmarks[31:36])

    # eyes
    linemask = connect_line(linemask, np.vstack([landmarks[36:42], landmarks[36]]))
    linemask = connect_line(linemask, np.vstack([landmarks[42:48], landmarks[42]]))

    # mouth
    linemask = connect_line(linemask, np.vstack([landmarks[48:60], landmarks[48]]))
    linemask = connect_line(linemask, np.vstack([landmarks[60:68], landmarks[60]]))

    return linemask

def connect_line(input, landmarks):
    img = input.copy()
    size= len(landmarks)
    for i in range(0, size-1):
        img = cv2.line(img,
                         (landmarks[i, 0], landmarks[i, 1]),
                         (landmarks[i+1, 0], landmarks[i+1, 1]),
                         (255, 255, 255),
                         1,
                         cv2.LINE_AA)

    return img

# face landmarks origin
def generate_region_mask(image, landmarks):
    regionmask = np.zeros_like(image[:, :, 0])
    '''
        Use five landmarks
        w = (five_landmarks[0, 1] - five_landmarks[1, 1]) / (five_landmarks[0, 0] - five_landmarks[1, 0])
        b = five_landmarks[0, 1] - five_landmarks[0, 0] * w
    '''
    # ----- layer2: eye-1
    eyepoints = np.vstack([landmarks[17:22], landmarks[36:42]])
    hull = cv2.convexHull(eyepoints.astype(np.int32)).astype(np.int32)
    regionmask = cv2.drawContours(regionmask, [hull], 0, (255), -1)
    # ----- layer2: eye-2
    eyepoints = np.vstack([landmarks[22:27], landmarks[42:48]])
    hull = cv2.convexHull(eyepoints.astype(np.int32)).astype(np.int32)
    regionmask = cv2.drawContours(regionmask, [hull], 0, (255), -1)
    # ----- layer3: mouth
    mouthpoints = landmarks[48:68]
    hull = cv2.convexHull(mouthpoints.astype(np.int32)).astype(np.int32)
    regionmask = cv2.drawContours(regionmask, [hull], 0, (255), -1)

    return regionmask


#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0', flip_input=False)
def generate_wider_mask(image, landmarks):
    '''
    generate weight mask according to landmarks
    :param image: np.ndarray
    :param landmarks: 68x2
    :return: a weight mask with weight of 0, 64, 128, 192
    '''
    #----- get five landmarks
    #five_landmarks = convert_to_five_landmarks(landmarks)

    facemask = generate_facial_mask(image, landmarks)
    eyemask = generate_eye_mask(image, landmarks)
    mouthmask = generate_mouth_mask(image, landmarks)

    weightmask = np.zeros_like(image[:, :, 0]).astype(np.uint8)
    weightmask[facemask > 0] = 64
    weightmask[eyemask > 0] = 128
    weightmask[mouthmask > 0] = 192

    return weightmask

def generate_facial_mask(image, landmarks):
    '''
    generate weight mask according to landmarks
    :param image: np.ndarray
    :param landmarks: 68x2
    :return: a mask map
    '''
    facemask = np.zeros_like(image[:, :, 0])
    # ----- layer1: face region
    hull = cv2.convexHull(np.int32(landmarks)).astype(np.int32)
    facemask = cv2.drawContours(facemask, [hull], 0, (255), -1)

    return facemask

def generate_eye_mask(image, landmarks):
    '''
    generate weight mask according to landmarks
    :param image: np.ndarray
    :param landmarks: 68x2
    :return: a mask map
    '''
    eyemask = np.zeros_like(image[:, :, 0])
    '''
        Use five landmarks
        w = (five_landmarks[0, 1] - five_landmarks[1, 1]) / (five_landmarks[0, 0] - five_landmarks[1, 0])
        b = five_landmarks[0, 1] - five_landmarks[0, 0] * w
    '''
    # ----- layer2: eye-1
    # use for each
    w = (landmarks[36, 1] - landmarks[39, 1]) / (landmarks[36, 0] - landmarks[39, 0])
    b = landmarks[36, 1] - landmarks[36, 0] * w
    eyepoints = np.vstack([landmarks[17:22], landmarks[36:42]])
    eyesymmetricpoints = flip_to_get_symmetric_points(eyepoints, (w, b))
    eyepoints = np.vstack([eyepoints, eyesymmetricpoints]).astype(np.int32)
    hull = cv2.convexHull(eyepoints).astype(np.int32)
    eyemask = cv2.drawContours(eyemask, [hull], 0, (255), -1)

    # ----- layer2: eye-2
    w = (landmarks[42, 1] - landmarks[45, 1]) / (landmarks[42, 0] - landmarks[45, 0])
    b = landmarks[42, 1] - landmarks[42, 0] * w
    eyepoints = np.vstack([landmarks[22:27], landmarks[42:48]])
    eyesymmetricpoints = flip_to_get_symmetric_points(eyepoints, (w, b))
    eyepoints = np.vstack([eyepoints, eyesymmetricpoints]).astype(np.int32)
    hull = cv2.convexHull(eyepoints).astype(np.int32)
    eyemask = cv2.drawContours(eyemask, [hull], 0, (255), -1)

    return eyemask

def generate_mouth_mask(image, landmarks):
    '''
    generate weight mask according to landmarks
    :param image: np.ndarray
    :param landmarks: 68x2
    :return: a mask map
    '''
    # ----- layer3: mouth
    mouthmask = np.zeros_like(image[:, :, 0])

    w = (landmarks[64, 1] - landmarks[60, 1]) / (landmarks[64, 0] - landmarks[60, 0])
    b = landmarks[60, 1] - landmarks[60, 0] * w

    mouthpoints = landmarks[48:68]
    mouthsymmetricpoints = flip_to_get_symmetric_points(mouthpoints, (w, b))
    mouthpoints = np.vstack([mouthpoints, mouthsymmetricpoints]).astype(np.int32)

    hull = cv2.convexHull(mouthpoints).astype(np.int32)
    mouthmask = cv2.drawContours(mouthmask, [hull], 0, (255), -1)

    return mouthmask


# Tools
def flip_to_get_symmetric_points(points, line):
    '''enlarge mouth region by symmetric flip
    the line l: y = ax + b
    the point: p(x0, y0)
    the image point p'(x1, y1)
    we can get the point coordinates as:
        x1 = [(1 - a**2)*x0 + 2a*y0 - 2a*b ] / (1 + a**2)
        y1 = [2a*x0 + (a**2 - 1)*y0 + 2b] / (1 + a**2)

    Args:
        points: input points
        line: a, b of the lines parameters
    Return: symmetric points
    '''
    a, b = line
    new_x = ((1 - a**2) * points[:, 0] + 2 * a * points[:, 1] - 2 * a * b) / (1 + a**2)  # shape: (n, )
    new_y = (2 * a * points[:, 0] + (a**2 - 1) * points[:, 1] + 2 * b) / (1 + a**2)  # shape: (n, )
    new_points = np.vstack([new_x, new_y]).T

    return new_points

def convert_to_five_landmarks(landmarks):
    '''
    transform 68 landmarks to 5 landmarks
    Arg:
        landmarks: 68 x 2 array

    Return:
        five_landmarks: 5 x 2 array
    '''
    if landmarks.shape[0] == 2:
        landmarks = landmarks.T

    five_marks = np.zeros((5, 2))
    five_marks[0] = np.mean(landmarks[36:42], axis=0)
    five_marks[1] = np.mean(landmarks[42:48], axis=0)
    five_marks[2] = np.mean(landmarks[29:36], axis=0)
    five_marks[3] = np.mean(np.stack([landmarks[48],landmarks[60]], axis=0), axis=0)
    five_marks[4] = np.mean(np.stack([landmarks[54],landmarks[64]], axis=0), axis=0)

    return five_marks
