import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

class CVPipeline(object):
    
    def __init__(self, imgpath, roi_center, roi_radius, huber_param):
                
        self.roi_center = roi_center
        self.roi_radius = roi_radius
        self.huber_param = huber_param
        
        self.img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        
    def crop_and_threshold(self):
        
        roi_mask = np.zeros_like(self.img, dtype = 'uint8')

        cv2.circle(roi_mask, self.roi_center, self.roi_radius, 255, -1)
        img_masked = cv2.bitwise_and(self.img, self.img, mask = roi_mask)
        
        self.img = cv2.GaussianBlur(self.img, (7, 7), 0)

        # Extract only ROI pixels for threshold calculation
        roi_pixels = self.img[roi_mask == 255]

        # Compute Otsu on ROI pixels only (no zeros from outside the mask)
        otsu_thresh, _ = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Otsu threshold: {otsu_thresh}")

        # Apply the threshold to the masked image
        _, img_thr = cv2.threshold(img_masked, otsu_thresh, 255, cv2.THRESH_TOZERO)
        self.img = img_thr
        
        print(f'Otsu threshold is {otsu_thresh}')
        
    def find_centroids(self):
        
        beam_points = []
        rows_valid = []

        for (i, row) in enumerate(self.img):
            total = row.sum()
            if total > 0:
                cols = np.arange(len(row))
                beam_points.append(int(np.rint(np.dot(cols, row) / total)))
                rows_valid.append(i)

        img_max = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

        for (x, y) in zip(beam_points, rows_valid):
            img_max = cv2.circle(img_max, (x, y), 2, color = (0, 0, 255), thickness = 1)
            
        self.img_max = img_max
    
    def run_pipeline(self):
        
        self.crop_and_threshold()
        self.find_centroids()
        