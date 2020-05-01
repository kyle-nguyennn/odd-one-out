import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyautogui
from PIL import Image
import imutils
import collections
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import time
import scipy
import keyboard
from multiprocessing import Process
import random

Box = collections.namedtuple('Box', 'left top width height')

# clf = LocalOutlierFactor(n_neighbors=3)
clf = OneClassSVM()
qIsPressed = False
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

def quitListener():
    global qIsPressed
    keyboard.wait('q')
    qIsPressed = True
    print(f"Quit game.")

def _calibrate():
    ### for some reason screen size and screenshot size are not same but scaled
    global scale
    global screenshot_size
    screen_size = pyautogui.size()
    print("Screen size: {}".format(screen_size))
    screenshot_size = np.array(pyautogui.screenshot()).shape[1::-1]
    print("Screenshot size: {}".format(screenshot_size))
    scale = screenshot_size[0] / screen_size[0]

def moveTo (x, y, interval = 0, physical=False):
    x_, y_ = x/scale, y/scale
    if physical:
        pyautogui.moveTo(x_, y_, interval)
    return x_, y_

def moveInGame(x,y, region, physical=False):
    x_base, y_base = moveTo(region.left, region.top, physical)
    x_, y_ = x/scale, y/scale
    if physical:
        pyautogui.moveRel(x_, y_)
    return x_base+x_, y_base+y_

def moveToCenter(box, duration = 0, physical=False):
    x, y, width, height = box
    if physical:
        moveTo(x+width/2, y+height/2, duration, physical)
    return x+width/2, y+height/2


def findInImage(needle, haystack, err=0.05):
    max_cor = 0
    correct_size = 1
    needle_corrected = needle.copy()
    max_res = None
    sample_size = int(1 / err)
    for s in np.linspace(0, 1.0, sample_size)[:0:-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(needle, width=int(needle.shape[1] * s))
        res = cv2.matchTemplate(resized, haystack, cv2.TM_CCOEFF_NORMED)
        cor = res.max()
        print(cor)
        if max_cor < cor:
            max_cor = cor
            max_res = res
            correct_size = s
            needle_corrected = resized

    coordinates = np.where(max_res == max_cor)
    x, y = coordinates[1][0], coordinates[0][0]
    height, width = needle_corrected.shape[0], needle_corrected.shape[1]

    return Box(x, y, width, height)

def locateOnScreen(img_path, err=0.05):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    screen = np.array(pyautogui.screenshot())
    screen = screen[..., :3]
    return findInImage(img, screen, err)

def observe(game_region, record=False, output=None):
    img = pyautogui.screenshot(region=game_region)
    if record:
        record_img = np.array(img)
        record_img = cv2.cvtColor(record_img, cv2.COLOR_BGR2RGB)
        #cv2.imshow("Gameplay", record_img)
        #output.write(record_img)
    return img

def drawCircles(img, x, y, r):
    record_img = np.array(img)
    x, y, r = int(x), int(y), int(r)
    record_img = cv2.cvtColor(record_img, cv2.COLOR_BGR2GRAY)
    output = cv2.circle(record_img, (x,y), r, (255, 255, 255), 5)
    resized = imutils.resize(output, width=int(output.shape[1] * 0.4))
    #out.write(output)
    cv2.imshow("Gameplay", resized)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        return

def oddOneOut(game_play, record=False):
    game_play_np = np.array(game_play)
    output = game_play_np.copy()
    gray = cv2.cvtColor(game_play_np, cv2.COLOR_BGR2GRAY)
    a = random.uniform(1.1, 1.5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, a, 1)

    # ensure at least 5 circles were found
    if circles is not None and circles.shape[1] >= 4:
        # convert the (x, y) coordinates and radius of the circles to integers
        np_circles = np.round(circles[0, :]).astype("int")
        # print(np_circles)
        # make sure all circles are the targeted circles
        # clustering the points based on coordinates
        # kmeans = KMeans(n_clusters=2, random_state=0).fit(np_circles[..., :2])
        # labels = kmeans.labels_
        # labels = clf.fit_predict(np_circles)
        # print(labels)
        # major = scipy.stats.mode(labels).mode
        # print(major)
        modes = scipy.stats.mode(np_circles).mode
        radius = modes[..., 2:].flatten().mean()
        cond = np.abs(np_circles[..., 2:] - modes[..., 2:]) <= 2
        targets = np_circles[cond.flatten()]
        # print(targets)

        colors = []
        for target in targets:
            x, y = target[0], target[1]
            colors.append(game_play_np[y, x, :])
        colors = np.array(colors)
        #print(colors)
        labels = clf.fit_predict(colors[..., :3])
        # print(labels)
        # identify the odd based on color
        # mean = colors.mean(axis=0)
        # std = colors.std(axis=0)
        mode = scipy.stats.mode(colors, axis=0).mode
        # print(colors != mode)
        outliers_idx = np.where(np.any(colors != mode, axis=1))
        #print(outliers_idx)
        from sklearn.metrics.pairwise import euclidean_distances
        # print(colors[outliers_idx])
        distances = euclidean_distances(colors[outliers_idx], mode)
        eliminated_id = outliers_idx[0][np.argmin(distances)]
        x_elim = targets[eliminated_id, 0]
        y_elim = targets[eliminated_id, 1]
        if record:
            drawCircles(game_play, x_elim, y_elim, radius)
        return (x_elim, y_elim)
    else: return None, None

if __name__ == "__main__":
    global out
    _calibrate()
    # set hyper parameter
    err = 0.2 # error tolerance for finding regions of the screen
    break_time = 0.02 # in seconds
    train_loop = 1
    game_loop  = 200
    # setup 'q' listener to quit playing
    #p = Process(target=quitListener)
    #p.start()
    # train loop
    for i in range(train_loop): # just once at a time for now. TODO: auto restart the game
        # set up screen recording

        # locate game_region. Assume this is fresh start
        # TODO: handle the case where it need restart the game
        print("Finding game region ...")
        game_region = locateOnScreen('game_region_windows.png', err)
        focus_region = Box(game_region.left, game_region.top + game_region.height/3, \
                game_region.width, 2*game_region.height/3)
        print(f"Found game region at ({game_region.left}, {game_region.top})")
        #out = cv2.VideoWriter('output.mp4', fourcc, 5.0, (game_region.width, game_region.height))

        print("Finding start button ...")
        start_button = locateOnScreen('start_button_windows.png', err)
        print(f"Found start button at ({start_button.left}, {start_button.top})")
        moveToCenter(start_button, physical=True)
        #pyautogui.leftClick()
        pyautogui.doubleClick()

        # wait to start
        print("Get ready ...")
        # game loop
        #for j in range(game_loop):
        while True:
            try:
                #time.sleep(break_time)
                game_play = observe(focus_region, record=False)
                x, y = oddOneOut(game_play, record=False)
                if x and y:
                    print(f"Odd at {x}, {y}")
                    x_, y_ = moveInGame(x, y, focus_region)
                    #pyautogui.leftClick(x_, y_)
                    pyautogui.doubleClick(x_, y_)
                if qIsPressed:
                    break
            except Exception as e:
                #TODO
                print(e)
                pass

        #out.release()
    #p.join()
