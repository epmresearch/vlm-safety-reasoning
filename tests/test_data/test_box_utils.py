import pytest
from data.box_utils import normalize_bbox, denormalize_bbox

def test_normalize_bbox():
    # [xmin, ymin, xmax, ymax]
    box = [100, 200, 300, 400]
    img_size = (1000, 1000) # width, height
    norm_box = normalize_bbox(box, img_size)
    assert norm_box == [100, 200, 300, 400] # 100/1000 * 1000 = 100

def test_denormalize_bbox():
    norm_box = [100, 200, 300, 400]
    img_size = (1000, 1000)
    box = denormalize_bbox(norm_box, img_size)
    assert box == [100, 200, 300, 400]
