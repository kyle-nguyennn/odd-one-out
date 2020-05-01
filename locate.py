import cv2
import numpy as np
import logging


needle = cv2.imread('game_region.png', cv2.IMREAD_GRAYSCALE)
haystack = cv2.imread('desktop.png', cv2.IMREAD_GRAYSCALE)

res = cv2.matchTemplate(needle, haystack, cv2.TM_CCOEFF_NORMED)

print("pause")