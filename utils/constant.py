#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

NEAR_INDEX = 0
FAR_INDEX = 1
HARD_SURFACE_OFFSET = 0.31326165795326233
PATCH_SIZE = 32
PATCH_SIZE_SQUARED = PATCH_SIZE**2
TRAIN_SET_LENGTH = 1000000
VALIDATION_SET_LENGTH = 10
CANONICAL_ZOOM_FACTOR = 1000/1280
CANONICAL_CAMERA_DIST_VAL = 1.6
CANONICAL_CAMERA_DIST_TRAIN = 2.0
#!HARDCODED May 25: prev = 1.9

DEFAULT_GEO_THRESH = 0.05
PERTURB_EPSILON = 0.01

#
NSR_BOUND = 1.6

GLOBAL_SEED = 42
# avartarclip style background augmentation
WHITE_BKG = 0
BLACK_BKG = 1
NOISE_BKG = 2
CHESSBOARD_BKG = 3

# # the center of head for beta=0 smpl Neutral model
# CAN_HEAD_OFFSET = 0.42
# CAN_HEAD_CAMERA_DIST = 0.6

# the center of head for beta=0 smpl Neutral model
CAN_HEAD_OFFSET = 0.47
CAN_HEAD_CAMERA_DIST = 0.5

# ration of (reconstruction) / (smpl mesh) 
SMPL_SCALE = 0.9
CANONICAL_CAMERA_DIST_TRAIN = CANONICAL_CAMERA_DIST_TRAIN * SMPL_SCALE
CANONICAL_CAMERA_DIST_VAL = CANONICAL_CAMERA_DIST_VAL * SMPL_SCALE
CAN_HEAD_CAMERA_DIST = CAN_HEAD_CAMERA_DIST * SMPL_SCALE
CAN_HEAD_OFFSET = CAN_HEAD_OFFSET * SMPL_SCALE