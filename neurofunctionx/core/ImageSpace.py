from enum import Enum

class ImageSpace(Enum):
    CT = "CTPost"
    T1 = "T1Pre"
    T2 = "T2Pre"
    NoOrient = "NoOrient"
    Atlas = "Atlas"
    # todo add rest