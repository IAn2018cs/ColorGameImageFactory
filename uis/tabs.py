# coding=utf-8
import enum


class TabId(enum.Enum):
    BATCH_GENERATE = 0
    BATCH2LINE_ART = 1
    GENERATE_LINE_ART_CLUSTERING = 2
    TRY_COLOR_GAME = 3
    GAN_EXTRACT_LINE = 4
    GENERATE_LINE_ART_SVG = 5
    IMAGE2SVG = 6
