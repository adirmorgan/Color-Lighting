import numpy as np
import math


def light_on_linear(t):
    max_val = 70
    duration = 90    # frames

    if t > duration:
        return None

    return (t/duration) * max_val


def light_off_linear(t):
    max_val = 70
    duration = 90    # frames

    if t > duration:
        return None

    return (1 - t/duration) * max_val


def light_on_sigmoid(t):
    max_val = 50
    duration = 5    # frames

    if t > duration:
        return None

    return max_val/(1 + math.exp(-(t-duration/2)))


def light_off_sigmoid(t):
    max_val = 50
    duration = 5    # frames

    if t > duration:
        return None

    return max_val * (1 - 1/(1 + math.exp(-(t-duration/2))))

class LightShape:
    def __init__(self, object_mask):
        self.__object_mask = object_mask
        self.__extra_luma = 0
        self.__changing = False
        self.__current_change_frame = 0
        self.__lighting_function = None

    def update(self):
        # self.__update_object_mask()
        if self.__changing:
            cur_update = self.__lighting_function(self.__current_change_frame)
            if cur_update is None:
                self.__terminate_change()
            else:
                self.__extra_luma = cur_update
                self.__current_change_frame += 1

    def change_lighting(self, lighting_function):
        if self.__changing:
            return

        self.__changing = True
        self.__current_change_frame = 0
        self.__lighting_function = lighting_function

    def get_extra_luma(self):
        return self.__extra_luma

    def _set_extra_luma(self, luma):
        self.__extra_luma = luma

    def get_object_mask(self):
        return self.__object_mask

    def update_object_mask(self, mask, K=5):
        self.__object_mask = (mask - self.__object_mask) * (2/(K+1)) + self.__object_mask

    def __terminate_change(self):
        self.__changing = False
        self.__lighting_function = None