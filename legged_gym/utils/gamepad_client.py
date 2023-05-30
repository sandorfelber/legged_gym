'''
Inspired from:
Thomas FLAYOLS - LAAS CNRS
From https://github.com/thomasfla/solopython
'''

import inputs
import time
from multiprocessing import Process
from multiprocessing.sharedctypes import Value
from ctypes import c_double, c_bool


class GamepadClient():
    def __init__(self):
        self.running = Value(c_bool, lock=True)
      
        self.startButton = Value(c_bool, lock=True)
        self.selectButton = Value(c_bool, lock=True)
        self.leftJoystickX = Value(c_double, lock=True)
        self.leftJoystickY = Value(c_double, lock=True)
        self.rightJoystickX = Value(c_double, lock=True)
        self.rightJoystickY = Value(c_double, lock=True)

        self.startButton.value = False
        self.selectButton.value = False
        self.leftJoystickX.value = 0.0
        self.leftJoystickY.value = 0.0
        self.rightJoystickX.value = 0.0
        self.rightJoystickY.value = 0.0

        args = (self.running, self.startButton, self.selectButton, self.leftJoystickX,
                self.leftJoystickY, self.rightJoystickX, self.rightJoystickY)
        self.process = Process(target=self.run, args=args)
        self.process.start()
        time.sleep(0.2)

    def run(self, running, startButton, selectButton, leftJoystickX, leftJoystickY, rightJoystickX, rightJoystickY):
        running.value = True
        while(running.value):
            events = inputs.get_gamepad()
            for event in events:
                if event.ev_type == 'Absolute':
                    if event.code == 'ABS_X':
                        leftJoystickX.value = event.state / 32768.0
                    if event.code == 'ABS_Y':
                        leftJoystickY.value = event.state / 32768.0
                    if event.code == 'ABS_RX':
                        rightJoystickX.value = event.state / 32768.0
                    if event.code == 'ABS_RY':
                        rightJoystickY.value = event.state / 32768.0
                if (event.ev_type == 'Key'):
                    if event.code == 'BTN_START':
                        startButton.value = event.state           
                    elif event.code == 'BTN_SELECT':
                        selectButton.value = event.state

    def stop(self):
        self.running.value = False
        self.process.terminate()
        self.process.join()