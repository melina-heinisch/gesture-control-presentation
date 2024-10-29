from enum import Enum

class Gesture(Enum):
    IDLE = 'idle'
    SWIPE_RIGHT = 'swipe_right'
    SWIPE_LEFT = 'swipe_left'
    ROTATE_RIGHT = 'rotate_right'
    ROTATE_LEFT = 'rotate_left'
    PINCH = 'pinch'
    SPREAD = 'spread'
    UP = 'up'
    DOWN = 'down'
    FLIP_TABLE = 'flip_table'
    SPIN = 'spin'