import time
import pyautogui

print("WGT SCREEN TEST")
print("Switch to the WGT game now.")
print("Taking screenshot in 5 seconds...")

time.sleep(5)

shot = pyautogui.screenshot()
shot.save("automation_screen_test.png")

print("DONE")
print("Saved: automation_screen_test.png")