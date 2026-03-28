import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

class CVPipeline(object):

    def __init__(
        self,
        imgpath,
        roi_center,
        roi_radius,
        huber_param,
        spatcal_data,
        max_row_width=50
    ):

        self.roi_center = roi_center
        self.roi_radius = roi_radius
        self.huber_param = huber_param
        self.max_row_width = max_row_width
        (self.dev_x, self.dev_y) = spatcal_data
        
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
        _, img_thr = cv2.threshold(img_masked, otsu_thresh - 10, 255, cv2.THRESH_TOZERO)
        self.img = img_thr
                
    def find_centroids(self):

        beam_points = []
        rows_valid = []

        for (i, row) in enumerate(self.img):
            nonzero_cols = np.nonzero(row)[0]
            if len(nonzero_cols) == 0:
                continue
            if nonzero_cols[-1] - nonzero_cols[0] > self.max_row_width:
                continue
            total = row.sum()
            beam_points.append(int(np.rint(np.dot(nonzero_cols, row[nonzero_cols]) / total)))
            rows_valid.append(i)
                
        self.points = np.column_stack((np.array(beam_points), np.array(rows_valid)))
        
    def convert_to_device_coordinates(self):
        
        points_x = np.array([self.dev_x[i, j] for i, j in self.points])
        points_y = np.array([self.dev_y[i, j] for i, j in self.points])
        self.points_phys = np.column_stack((points_x, points_y))
        
    def fit_line(self):
        
        self.fit_params = (cv2.fitLine(
            self.points_phys,
            cv2.DIST_HUBER,
            self.huber_param,
            0.1, 0.1
        )).T[0] # v_x, v_y, x_0, y_0
        
    def visualize_centroids(self):
        
        img_centroids = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

        for (x, y) in zip(self.points[:, 0], self.points[:, 1]):
            img_centroids = cv2.circle(img_centroids, (x, y), 2, color = (0, 0, 255), thickness = 1)
            
        return img_centroids
        