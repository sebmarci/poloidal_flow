"""
Beam axis calibration from CMOS images of the ABES beam.

The poloidal separation `s` of the two deflection states is what converts a CCF
time lag into a velocity, so it has to be measured. A CMOS camera views the
beam; this module extracts the beam axis from such an image and fits a line
through it in device coordinates. Running it on the images of the two
deflection states gives their separation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

class CVPipeline(object):
    """
    Extract and fit the ABES beam axis from a CMOS image.

    The stages are meant to be called in order: ``crop_and_threshold``,
    ``find_centroids``, ``convert_to_device_coordinates``, ``fit_line``. Each
    stores its result on the instance for the next one.

    Parameters
    ----------
    imgpath : str
        Path to the CMOS image, read as 8-bit grayscale.
    roi_center : Tuple[int, int]
        Centre of the circular region of interest as (column, row) in pixels.
    roi_radius : int
        Radius of the region of interest in pixels.
    huber_param : float
        The C parameter of the Huber loss used by ``cv2.fitLine``: the residual
        beyond which a point is treated as an outlier, in device coordinate
        units (m).
    spatcal_data : Tuple[numpy.ndarray, numpy.ndarray]
        The 'Device x' and 'Device y' pixel maps of the camera, from
        ``flap_w7x_abes.ShotSpatCalCMOS``. Both are indexed as
        ``[column, row]``.
    max_row_width : int, default=175
        Reject an image row if its lit pixels span more than this many columns.
        Such a row is stray light or reflection rather than beam.
    min_row_pixels : int, default=5
        Reject an image row with fewer lit pixels than this.

    Attributes
    ----------
    img : numpy.ndarray
        The working image. Overwritten in place by ``crop_and_threshold``.
    points : numpy.ndarray
        Beam centroids as ``(column, row)`` pixel indices, shape ``(n, 2)``.
        Set by ``find_centroids``.
    points_phys : numpy.ndarray
        The same centroids as device ``(x, y)`` coordinates in m. Set by
        ``convert_to_device_coordinates``.
    fit_params : numpy.ndarray
        Fitted line as ``[v_x, v_y, x_0, y_0]``: a unit direction vector and a
        point on the line, in device coordinates. Set by ``fit_line``.

    Examples
    --------
    >>> spatcal = flap_w7x_abes.ShotSpatCalCMOS('20250312.022')
    >>> spatcal.read()
    >>> pl = CVPipeline('cmos/20250225.010.bmp', (639, 578), 220, 0.015,
    ...                 (spatcal.data['Device x'], spatcal.data['Device y']))
    >>> pl.crop_and_threshold()
    >>> pl.find_centroids()
    >>> pl.convert_to_device_coordinates()
    >>> pl.fit_line()
    """

    def __init__(
        self,
        imgpath,
        roi_center,
        roi_radius,
        huber_param,
        spatcal_data,
        max_row_width=175,
        min_row_pixels=5
    ):

        self.roi_center = roi_center
        self.roi_radius = roi_radius
        self.huber_param = huber_param
        self.max_row_width = max_row_width
        self.min_row_pixels = min_row_pixels
        (self.dev_x, self.dev_y) = spatcal_data
        
        self.img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        
    def crop_and_threshold(self):
        """
        Mask the image to the region of interest and threshold out the background.

        Replaces ``self.img`` with the thresholded image.

        Notes
        -----
        Otsu's method needs a bimodal histogram, so the threshold is computed
        from the pixels inside the circular ROI only; including the zeros
        outside the mask would drag it down. It is then applied with a 10 count
        margin and with ``THRESH_TOZERO``, which zeroes the background but keeps
        the grayscale values of the beam, since ``find_centroids`` weights the
        centroids by intensity.

        The median blur is applied to ``self.img`` after the masked copy is
        taken, so the threshold is computed on blurred pixels but applied to
        unblurred ones.
        """

        roi_mask = np.zeros_like(self.img, dtype = 'uint8')

        cv2.circle(roi_mask, self.roi_center, self.roi_radius, 255, -1)
        img_masked = cv2.bitwise_and(self.img, self.img, mask = roi_mask)
        
        # Apply median filter
        self.img = cv2.medianBlur(self.img, ksize=5)

        # Extract only ROI pixels for threshold calculation
        roi_pixels = self.img[roi_mask == 255]

        # Compute Otsu on ROI pixels only (no zeros from outside the mask)
        otsu_thresh, _ = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Otsu threshold: {otsu_thresh}")

        # Apply the threshold to the masked image
        _, img_thr = cv2.threshold(img_masked, otsu_thresh - 10, 255, cv2.THRESH_TOZERO)
        self.img = img_thr
                
    def find_centroids(self):
        """
        Find the beam centre in each image row.

        Sets ``self.points`` to the intensity-weighted centroid of every
        accepted row, as ``(column, row)`` pixel indices.

        Notes
        -----
        The beam runs roughly across the image rows, so one centroid per row
        traces the axis. Rows with fewer than `min_row_pixels` lit pixels or a
        lit span wider than `max_row_width` are skipped as noise or stray light.
        Callers should check that enough points survived before fitting.
        """

        beam_points = []
        rows_valid = []

        for (i, row) in enumerate(self.img):
            nonzero_cols = np.nonzero(row)[0]
            if len(nonzero_cols) < self.min_row_pixels:
                continue
            if nonzero_cols[-1] - nonzero_cols[0] > self.max_row_width:
                continue
            total = row.sum()
            beam_points.append(int(np.rint(np.dot(nonzero_cols, row[nonzero_cols]) / total)))
            rows_valid.append(i)
                
        self.points = np.column_stack((np.array(beam_points), np.array(rows_valid)))
        
    def convert_to_device_coordinates(self):
        """
        Map the pixel centroids to device coordinates.

        Sets ``self.points_phys`` to the device ``(x, y)`` position of each
        centroid in m, looked up in the CMOS spatial calibration maps.
        """

        points_x = np.array([self.dev_x[i, j] for i, j in self.points])
        points_y = np.array([self.dev_y[i, j] for i, j in self.points])
        self.points_phys = np.column_stack((points_x, points_y))
        
    def fit_line(self):
        """
        Fit the beam axis through the centroids in device coordinates.

        Sets ``self.fit_params`` to ``[v_x, v_y, x_0, y_0]``.

        Notes
        -----
        Uses ``cv2.fitLine`` with a Huber loss, so surviving outlier centroids
        (reflections, a stray bright row) pull on the fit far less than they
        would in a least squares fit.
        """

        self.fit_params = (cv2.fitLine(
            self.points_phys,
            cv2.DIST_HUBER,
            self.huber_param,
            0.1, 0.1
        )).T[0] # v_x, v_y, x_0, y_0
        
    def return_grayscale_img(self):
        """
        Return the current working image as a 3-channel BGR array.

        Convenience for overlaying coloured centroids or the fitted line on the
        thresholded image.
        """

        return cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        