import pyautogui

pyautogui.PAUSE = 15
pyautogui.FAILSAFE = True


while True:
    pyautogui.moveTo(100, 200, 2)
    pyautogui.moveTo(200,100, 2)
