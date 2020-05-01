import cv2
import numpy as np
import pyautogui

screen_size = pyautogui.size()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, screen_size)

game_img = cv2.imread('game_region.png')
cv2.imshow("test", game_img)
game_region = pyautogui.locateCenterOnScreen('game_region.png')

while True:
    img = pyautogui.screenshot()
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("Screenshot", frame)
    out.write(frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

# make sure everything is closed when exited
out.release()
cv2.destroyAllWindows()
