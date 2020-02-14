import cv2
import numpy as np


def threshold_segmentation(snapshot):
    assert snapshot.ndim == 3

    snapshot_mask = []
    for i, img in enumerate(snapshot):
        snapshot_mask.append(aorta_mask_slice_cv(img))
    snapshot_mask = np.array(snapshot_mask)
    return snapshot_mask


def aorta_mask_slice_cv(image, min_area=60, threshold=275):
        """
        min_area: minimum area for contour to check if it is aorta, to filter out to small contours
        threshold: binary threshold, all that bigger substituted to 255, all that smaller to 0
        """
        ret, img_n = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        img_n = cv2.normalize(src=img_n, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cnt, _ = cv2.findContours(img_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # to manage nested contours
        for c in cnt:
            cv2.drawContours(img_n, [c], 0, 255, -1)
        cnt, _ = cv2.findContours(img_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # here assumption is made that aorta is a contour which is the closest to circle
        # not always true, but in most cases works
        cnt_areas = []
        for c in cnt:
            contour_area = cv2.contourArea(c)
            if contour_area > min_area:
                _, radius = cv2.minEnclosingCircle(c)
                radius = int(radius)
                circle_area = np.pi * radius ** 2
                # check how close contour is to circle
                a = round(circle_area / contour_area, 1)
                cnt_areas.append((c, contour_area, a))

        img_n = np.zeros(img_n.shape)
        if len(cnt_areas) != 0:
            cnt_areas = sorted(cnt_areas, key=lambda x: x[2])
            cv2.drawContours(img_n, [cnt_areas[0][0]], 0, 1, cv2.FILLED)
        return img_n