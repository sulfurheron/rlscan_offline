import numpy as np
import cv2
from PIL import Image

def scanner_frames_from_obs(obs):
    # TODO what happened to that Dali decoder? So much faster, helps not skip frames.
    jpg_frames = [
        obs.scanner_1_jpg,
        obs.scanner_2_jpg,
        obs.scanner_3_jpg,
        obs.scanner_4_jpg,
    ]
    if any(len(frame) == 0 for frame in jpg_frames):
        # for testing or if we haven't gotten image data yet
        scanner_frames = np.zeros((4, 256, 256, 1), np.uint8)
    else:
        images = [
            cv2.imdecode(np.frombuffer(jpg_frame, np.uint8), 0)
            for jpg_frame in jpg_frames]
        # images = [
        #     np.array(Image.open(jpg_frame, "jpeg"))
        #     for jpg_frame in jpg_frames]
        small_images = [
            # INTER_AREA interpolation preserves better fine features when downsizing images
            cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            for image in images]
        # create with shape (4, 256, 256, 1)
        #scanner_frames = np.asarray(small_images)[..., None]
        scanner_frames = small_images
        #assert scanner_frames.shape == (4, 256, 256, 1)
    return scanner_frames
    #return jpg_frames
