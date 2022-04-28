import enum


class EventTypes(enum.Enum):
    LOOP_START = 0
    MARKER = 1
    FUNC_CALL = 2
