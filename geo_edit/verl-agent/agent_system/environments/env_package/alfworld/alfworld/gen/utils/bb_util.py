import numbers
import numpy as np

LIMIT = 99999999

def clip_bbox(bboxes, min_clip, max_x_clip, max_y_clip):
    '''
    # BBoxes are [x1, y1, x2, y2]
    '''
    bboxes_out = bboxes
    added_axis = False
    if len(bboxes_out.shape) == 1:
        added_axis = True
        bboxes_out = bboxes_out[:, np.newaxis]
    bboxes_out[[0, 2], ...] = np.clip(bboxes_out[[0, 2], ...], min_clip, max_x_clip)
    bboxes_out[[1, 3], ...] = np.clip(bboxes_out[[1, 3], ...], min_clip, max_y_clip)
    if added_axis:
        bboxes_out = bboxes_out[:, 0]
    return bboxes_out


def xyxy_to_xywh(bboxes, clip_min=-LIMIT, clip_width=LIMIT, clip_height=LIMIT, round=False):
    '''
    [x1 y1, x2, y2] to [xMid, yMid, width, height]
    '''
    added_axis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        added_axis = True
        bboxes = bboxes[:, np.newaxis]
    bboxes_out = np.zeros(bboxes.shape)
    x1 = bboxes[0, ...]
    y1 = bboxes[1, ...]
    x2 = bboxes[2, ...]
    y2 = bboxes[3, ...]
    bboxes_out[0, ...] = (x1 + x2) / 2.0
    bboxes_out[1, ...] = (y1 + y2) / 2.0
    bboxes_out[2, ...] = x2 - x1
    bboxes_out[3, ...] = y2 - y1
    if clip_min != -LIMIT or clip_width != LIMIT or clip_height != LIMIT:
        bboxes_out = clip_bbox(bboxes_out, clip_min, clip_width, clip_height)
    if bboxes_out.shape[0] > 4:
        bboxes_out[4:, ...] = bboxes[4:, ...]
    if added_axis:
        bboxes_out = bboxes_out[:, 0]
    if round:
        bboxes_out = np.round(bboxes_out).astype(int)
    return bboxes_out


def xywh_to_xyxy(bboxes, clip_min=-LIMIT, clip_width=LIMIT, clip_height=LIMIT, round=False):
    '''
    [xMid, yMid, width, height] to [x1 y1, x2, y2]
    '''
    added_axis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        added_axis = True
        bboxes = bboxes[:, np.newaxis]
    bboxes_out = np.zeros(bboxes.shape)
    x_mid = bboxes[0, ...]
    y_mid = bboxes[1, ...]
    width = bboxes[2, ...]
    height = bboxes[3, ...]
    bboxes_out[0, ...] = x_mid - width / 2.0
    bboxes_out[1, ...] = y_mid - height / 2.0
    bboxes_out[2, ...] = x_mid + width / 2.0
    bboxes_out[3, ...] = y_mid + height / 2.0
    if clip_min != -LIMIT or clip_width != LIMIT or clip_height != LIMIT:
        bboxes_out = clip_bbox(bboxes_out, clip_min, clip_width, clip_height)
    if bboxes_out.shape[0] > 4:
        bboxes_out[4:, ...] = bboxes[4:, ...]
    if added_axis:
        bboxes_out = bboxes_out[:, 0]
    if round:
        bboxes_out = np.round(bboxes_out).astype(int)
    return bboxes_out


def scale_bbox(bboxes, scalars, clip_min=-LIMIT, clip_width=LIMIT, clip_height=LIMIT, round=False, in_place=False):
    '''
    @bboxes {np.array} 4xn array of boxes to be scaled
    @scalars{number or arraylike} scalars for width and height of boxes
    @in_place{bool} If false, creates new bboxes.
    '''
    added_axis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes, dtype=np.float32)
    if len(bboxes.shape) == 1:
        added_axis = True
        bboxes = bboxes[:, np.newaxis]
    if isinstance(scalars, numbers.Number):
        scalars = np.full((2, bboxes.shape[1]), scalars, dtype=np.float32)
    if not isinstance(scalars, np.ndarray):
        scalars = np.array(scalars, dtype=np.float32)
    if len(scalars.shape) == 1:
        scalars = np.tile(scalars[:, np.newaxis], (1, bboxes.shape[1]))

    width = bboxes[2, ...] - bboxes[0, ...]
    height = bboxes[3, ...] - bboxes[1, ...]
    x_mid = (bboxes[0, ...] + bboxes[2, ...]) / 2.0
    y_mid = (bboxes[1, ...] + bboxes[3, ...]) / 2.0
    if not in_place:
        bboxes_out = bboxes.copy()
    else:
        bboxes_out = bboxes

    bboxes_out[0, ...] = x_mid - width * scalars[0, ...] / 2.0
    bboxes_out[1, ...] = y_mid - height * scalars[1, ...] / 2.0
    bboxes_out[2, ...] = x_mid + width * scalars[0, ...] / 2.0
    bboxes_out[3, ...] = y_mid + height * scalars[1, ...] / 2.0

    if clip_min != -LIMIT or clip_width != LIMIT or clip_height != LIMIT:
        bboxes_out = clip_bbox(bboxes_out, clip_min, clip_width, clip_height)
    if added_axis:
        bboxes_out = bboxes_out[:, 0]
    if round:
        bboxes_out = np.round(bboxes_out).astype(int)
    return bboxes_out


def make_square(bboxes, in_place=False):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        num_boxes = 1
        width = bboxes[2] - bboxes[0]
        height = bboxes[3] - bboxes[1]
    else:
        num_boxes = bboxes.shape[1]
        width = bboxes[2, ...] - bboxes[0, ...]
        height = bboxes[3, ...] - bboxes[1, ...]
    max_size = np.maximum(width, height)
    scalars = np.zeros((2, num_boxes))
    scalars[0, ...] = max_size * 1.0 / width
    scalars[1, ...] = max_size * 1.0 / height
    return scale_bbox(bboxes, scalars, in_place=in_place)
