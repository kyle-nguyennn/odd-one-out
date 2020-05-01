import cv2
import pyautogui
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
screenshot = np.array(pyautogui.screenshot())
width = int(screenshot.shape[1])
height = int(screenshot.shape[0])
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))
print("Recording...")
#for i in range(20):
while True:
    screenshot = np.array(pyautogui.screenshot())
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
    #screenshot = screenshot[..., :3]
    out.write(screenshot)
    cv2.imshow("test", screenshot)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
print("Done")
out.release()