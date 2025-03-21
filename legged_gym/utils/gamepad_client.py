# # '''
# # Inspired from:
# # Thomas FLAYOLS - LAAS CNRS
# # From https://github.com/thomasfla/solopython
# # '''

import inputs
import time
from multiprocessing import Process
from multiprocessing.sharedctypes import Value
from ctypes import c_double, c_bool
from pynput import keyboard

class GamepadClient():
    def __init__(self):
        self.running = Value(c_bool, True, lock=True)
      
        self.startButton = Value(c_bool, False, lock=True)
        self.selectButton = Value(c_bool, False, lock=True)
        self.leftJoystickX = Value(c_double, 0.0, lock=True)
        self.leftJoystickY = Value(c_double, 0.0, lock=True)
        self.rightJoystickX = Value(c_double, 0.0, lock=True)
        self.rightJoystickY = Value(c_double, 0.0, lock=True)
        self.yawControl = Value(c_double, 0.0, lock=True)  # Yaw control

        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_listener.start()

        args = (self.running, self.startButton, self.selectButton, self.leftJoystickX,
                self.leftJoystickY, self.rightJoystickX, self.rightJoystickY, self.yawControl)
        self.process = Process(target=self.run, args=args)
        self.process.start()
        time.sleep(0.2)

    def run(self, running, startButton, selectButton, leftJoystickX, leftJoystickY, rightJoystickX, rightJoystickY, yawControl):
        running.value = True
        while running.value:
            try:
                events = inputs.get_gamepad()
                for event in events:
                    self.handle_gamepad_input(event, startButton, selectButton, leftJoystickX, leftJoystickY, rightJoystickX, rightJoystickY, yawControl)
            except inputs.UnpluggedError:
                pass  # No gamepad found, ignore

    def handle_gamepad_input(self, event, startButton, selectButton, leftJoystickX, leftJoystickY, rightJoystickX, rightJoystickY, yawControl):
        if event.ev_type == 'Absolute':
            if event.code == 'ABS_X':
                leftJoystickX.value = event.state / 32768.0
            if event.code == 'ABS_Y':
                leftJoystickY.value = event.state / 32768.0
            if event.code == 'ABS_RX':
                rightJoystickX.value = event.state / 32768.0
            if event.code == 'ABS_RY':
                rightJoystickY.value = event.state / 32768.0
        if event.ev_type == 'Key':
            if event.code == 'BTN_START':
                startButton.value = event.state           
            elif event.code == 'BTN_SELECT':
                selectButton.value = event.state

    def on_press(self, key):
        try:
            if key == keyboard.Key.f2:
                self.yawControl.value = -1.0  # Yaw left
            elif key == keyboard.Key.f3:
                self.yawControl.value = 1.0  # Yaw right
            elif key == keyboard.Key.left:
                self.leftJoystickX.value = -1.0
            elif key == keyboard.Key.right:
                self.leftJoystickX.value = 1.0
            elif key == keyboard.Key.up:
                self.leftJoystickY.value = -1.0
            elif key == keyboard.Key.down:
                self.leftJoystickY.value = 1.0
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key in [keyboard.Key.f2, keyboard.Key.f3]:
                self.yawControl.value = 0.0  # Stop yaw movement
            elif key in [keyboard.Key.left, keyboard.Key.right]:
                self.leftJoystickX.value = 0.0
                self.yawControl.value = 0.0
            elif key in [keyboard.Key.up, keyboard.Key.down]:
                self.leftJoystickY.value = 0.0
        except AttributeError:
            pass

    def stop(self):
        self.running.value = False
        self.keyboard_listener.stop()
        self.process.terminate()
        self.process.join()

# Example usage
if __name__ == "__main__":
    gamepad_client = GamepadClient()
    try:
        while True:
            print("Left Joystick X:", gamepad_client.leftJoystickX.value)
            print("Left Joystick Y:", gamepad_client.leftJoystickY.value)
            print("Yaw Control:", gamepad_client.yawControl.value)
            time.sleep(0.1)
    except KeyboardInterrupt:
        gamepad_client.stop()
