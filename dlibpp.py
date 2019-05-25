# -*- coding: utf-8 -*-
'''make dlib a class to align face'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import dlib
import numpy as np

class DLIBPP(object):

    def __init__(self, num_of_marks=68, plus=True):
        super(DLIBPP, self).__init__()
        self.current_path, _ = os.path.split(os.path.abspath(__file__))

        self.detector_path = os.path.join(self.current_path, "dlibmodels/mmod_human_face_detector.dat")

        if plus:
            self.detector = dlib.cnn_face_detection_model_v1(self.detector_path)
        else:
            self.detector = dlib.get_frontal_face_detector()
        self.img = None
        self.bounding_boxes = None
        self.points = None
        self.num_of_marks = 68 if num_of_marks==68 else 81
        self.plus = plus

        self.predictor_path = os.path.join(self.current_path, "dlibmodels/shape_predictor_%s_face_landmarks.dat" % str(num_of_marks))
        self.shape_predictor = dlib.shape_predictor(self.predictor_path)

    def get_five_landmarks(self, image, nrof_marks=68):
        '''
        get 2-D landmarks from image
        Args:
            image:
            nrof_marks: return with 68 landmarks or 5 landmarks

        Return:
            bounding_boxes: (n, 4)
            points: (n, 68, 2) or (n, 5, 2)
        '''
        _, landmarks = self.detect_faces(image)
        if nrof_marks == 5 and len(landmarks) > 0:
            _temp_marks = []
            for landmarks_item in landmarks:
                _temp_marks.append(self.convert_to_five_landmarks(landmarks_item))
            landmarks = np.squeeze(_temp_marks)

        return landmarks


    def detect_faces(self, image):
        '''
        input image, return bounding box and five landmarks of faces
        Args:
            image:
            reporttime: bool type. If print the usage time

        Return:
            bounding_boxes: (n, 4)
            points: (n, 68, 2)
        '''
        if image is None:
            print('Input image is None, return None\n')
            return None, None

        bounding_boxes, points = self.__convert_dlib_results__(image)
        self.img, self.bounding_boxes, self.points = image, bounding_boxes, points

        return bounding_boxes, points

    def generate_crop_box(self, image, image_info=None, scale=1.2):
        '''
        giving provided image_info and rescale the box to new size
        Args:
            image_info: the bounding box or the landmarks

        Return:
            a box with 4 values: [left, top, right, bottom] or a
            list contains several box, each has 4 landmarks
        '''
        if image_info is not None:
            if np.max(image_info.shape) > 4:  # key points to get bounding box
                kpt = image_info
                if kpt.shape[0] < 3:
                    kpt = kpt.T   # nof_marks x 2
                if kpt.shape[0] <= 5:  # 5 x 2
                    scale = scale*scale
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
            else:  # bounding box
                bbox = image_info
                left = bbox[0]
                right = bbox[2]
                top = bbox[1]
                bottom = bbox[3]

            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size * scale)
            box = [center[0] - size / 2, center[1] - size / 2,
                   center[0] + size / 2, center[1] + size / 2]

            return box
        else:
            box, landmarks = self.detect_faces(image)
            boxes = []
            for landmarks_item in landmarks:
                boxes.append(self.generate_crop_box(image, landmarks_item, scale))

            return boxes


    def __convert_dlib_results__(self, image):
        '''
        Convert dlib sub member type to numpy array
        Args:
            image: input image

        Return:
            boxes: (n, 4)
            points: (n, 68, 2)
        '''
        if self.plus:
            bounding_boxes = self.detector(image, 1)
        else:
            bounding_boxes = self.detector(image)

        return_boxes = []
        return_points = []
        for idx in range(len(bounding_boxes)):
            rect = bounding_boxes[idx].rect
            shape = self.shape_predictor(image, rect).parts()
            landmarks = np.array([[p.x, p.y] for p in shape])

            # return_boxes.append([rect.left(), rect.top(), rect.right(), rect.bottom()])
            return_boxes.append(self.generate_crop_box(image, image_info=landmarks, scale=1.0))
            return_points.append(landmarks)

        if len(bounding_boxes) > 0:
            return_boxes, return_points = np.int32(return_boxes), np.array(return_points)

        return return_boxes, return_points

    def convert_to_five_landmarks(self, landmarks):
        '''
        transform 68 landmarks to 5 landmarks
        Arg:
            landmarks: 68 x 2 array

        Return:
            5 x 2 array
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

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def __rotate__(self, image, angle, center=None, scale=1.0):

        (h, w) = image.shape[:2]  # 2
        if center is None:  # 3
            center = (w // 2, h // 2)  # 4
        M = cv2.getRotationMatrix2D(center, angle, scale)  # 5

        rotated = cv2.warpAffine(image, M, (w, h))  # 6
        return rotated  # 7

    def __align_rotate__(self, image, pts, scale=1.0):
        angle = np.arctan((pts[5] - pts[6]) / (pts[0] - pts[1])) / 3.14159 * 180
        newimg = self.__rotate__(image, angle, (pts[2], pts[7]), scale)

        return newimg
