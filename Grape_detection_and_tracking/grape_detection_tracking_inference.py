from ultralytics import YOLO
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import os, string, random
from supervision.draw.color import ColorPalette
from log_settings import logger
from tqdm.notebook import tqdm
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from onemetric.cv.utils.iou import box_iou_batch
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker,STrack
# from yolox.tracker.byte_tracker import BYTETracker,STrack
from dataclasses import dataclass


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, 
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

class GrapeDetectTrack():
    def __init__(self,input_video_file_path):
        self.model = YOLO(model=f"{os.getcwd()}{os.sep}models{os.sep}grape_detection.pt")
        self.model.fuse()
        self.SOURCE_VIDEO_PATH = input_video_file_path
        logger.info("Video file received starting grape detection and trackings")
    def perform_tracking_detection(self):
        try:
            CLASS_NAMES_DICT=self.model.model.names
            CLASS_ID = [0]
            # settings
            LINE_START = Point(80, 10)
            LINE_END = Point(80, 1070)
            TARGET_VIDEO_PATH = f"{os.getcwd()}{os.sep}Detections{os.sep}{''.join(random.choices(string.ascii_letters, k=5))}_result.mp4"
            print(TARGET_VIDEO_PATH)
            VideoInfo.from_video_path(self.SOURCE_VIDEO_PATH)

            # create BYTETracker instance
            byte_tracker = BYTETracker(BYTETrackerArgs())
            # create VideoInfo instance
            video_info = VideoInfo.from_video_path(self.SOURCE_VIDEO_PATH)
            # create frame generator
            generator = get_video_frames_generator(self.SOURCE_VIDEO_PATH)
            # create LineCounter instance
            line_counter = LineCounter(start=LINE_START, end=LINE_END)
            # create instance of BoxAnnotator and LineCounterAnnotator
            box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
            line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)
            logger.info(f"{video_info.total_frames} has been identified in the input video file")
            # open target video file
            with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
                # loop over video frames
                for frame in tqdm(generator, total=video_info.total_frames):
                    # model prediction on single frame and conversion to supervision Detections
                    results = self.model(frame)
                    detections = Detections(
                        xyxy=results[0].boxes.xyxy.cpu().numpy(),
                        confidence=results[0].boxes.conf.cpu().numpy(),
                        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                    )
                    # filtering out detections with unwanted classes
                    mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
                    detections.filter(mask=mask, inplace=True)
                    # tracking detections
                    tracks = byte_tracker.update(
                        output_results=detections2boxes(detections=detections),
                        img_info=frame.shape,
                        img_size=frame.shape
                    )
                    tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
                    detections.tracker_id = np.array(tracker_id)
                    # filtering out detections without trackers
                    mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
                    detections.filter(mask=mask, inplace=True)
                    # format custom labels
                    labels = [
                        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                        for _, confidence, class_id, tracker_id
                        in detections
                    ]
                    # updating line counter
                    line_counter.update(detections=detections)
                    # annotate and display frame
                    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                    line_annotator.annotate(frame=frame, line_counter=line_counter)
                    sink.write_frame(frame)
            logger.info(f"Grape Detection Tracking completed file saved in {TARGET_VIDEO_PATH}")
            return TARGET_VIDEO_PATH
        except:
            logger.error("Error in Grape Detection Tracking")

# GrapeDetectTrack(input_video_file_path = "Grape-detection-and-tracking\\testing_data\\grape_video_pan_in_out.mp4").perform_tracking_detection()