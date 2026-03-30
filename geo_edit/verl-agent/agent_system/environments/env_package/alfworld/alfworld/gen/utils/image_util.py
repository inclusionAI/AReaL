import numpy as np

from alfworld.gen import constants


def bbox_to_mask(bbox):
    '''
    bbox to rectangle pixelwise mask
    '''
    x1, y1, x2, y2 = bbox
    mask = np.zeros((constants.DETECTION_SCREEN_HEIGHT, constants.DETECTION_SCREEN_WIDTH)).astype(int)
    mask[y1:y2, x1:x2] = 1
    return mask


def point_to_mask(point):
    '''
    single point to dense pixelwise mask
    '''
    x, y = point
    mask = np.zeros((constants.DETECTION_SCREEN_HEIGHT, constants.DETECTION_SCREEN_WIDTH)).astype(int)
    mask[y, x] = 1
    return mask


def decompress_mask(compressed_mask):
    '''
    decompress compressed mask array
    '''
    mask = np.zeros((constants.DETECTION_SCREEN_WIDTH, constants.DETECTION_SCREEN_HEIGHT))
    for start_idx, run_len in compressed_mask:
        for idx in range(start_idx, start_idx + run_len):
            mask[idx // constants.DETECTION_SCREEN_WIDTH, idx % constants.DETECTION_SCREEN_HEIGHT] = 1
    return mask


def compress_mask(seg_mask):
    '''
    compress mask array
    '''
    run_len_compressed = []  # list of lists of run lengths for 1s, which are assumed to be less frequent.
    idx = 0
    curr_run = False
    run_len = 0
    for x_idx in range(len(seg_mask)):
        for y_idx in range(len(seg_mask[x_idx])):
            if seg_mask[x_idx][y_idx] == 1 and not curr_run:
                curr_run = True
                run_len_compressed.append([idx, None])
            if seg_mask[x_idx][y_idx] == 0 and curr_run:
                curr_run = False
                run_len_compressed[-1][1] = run_len
                run_len = 0
            if curr_run:
                run_len += 1
            idx += 1
    if curr_run:
        run_len_compressed[-1][1] = run_len
    return run_len_compressed