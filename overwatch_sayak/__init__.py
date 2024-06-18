import importlib.metadata as importlib_metadata

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"

from overwatch_sayak.annotators.core import (
    BlurAnnotator,
    BoundingBoxAnnotator,
    BoxCornerAnnotator,
    CircleAnnotator,
    ColorAnnotator,
    CropAnnotator,
    DotAnnotator,
    EllipseAnnotator,
    HaloAnnotator,
    HeatMapAnnotator,
    LabelAnnotator,
    MaskAnnotator,
    OrientedBoxAnnotator,
    PercentageBarAnnotator,
    PixelateAnnotator,
    PolygonAnnotator,
    RichLabelAnnotator,
    RoundBoxAnnotator,
    TraceAnnotator,
    TriangleAnnotator,
)
from overwatch_sayak.annotators.utils import ColorLookup
from overwatch_sayak.classification.core import Classifications
from overwatch_sayak.dataset.core import (
    BaseDataset,
    ClassificationDataset,
    DetectionDataset,
)
from overwatch_sayak.dataset.utils import mask_to_rle, rle_to_mask
from overwatch_sayak.detection.annotate import BoxAnnotator
from overwatch_sayak.detection.core import Detections
from overwatch_sayak.detection.line_zone import LineZone, LineZoneAnnotator
from overwatch_sayak.detection.lmm import LMM
from overwatch_sayak.detection.overlap_filter import (
    OverlapFilter,
    box_non_max_merge,
    box_non_max_suppression,
    mask_non_max_suppression,
)
from overwatch_sayak.detection.tools.csv_sink import CSVSink
from overwatch_sayak.detection.tools.inferenceslicer import InferenceSlicer
from overwatch_sayak.detection.tools.json_sink import JSONSink
from overwatch_sayak.detection.tools.polygon_zone import PolygonZone, PolygonZoneAnnotator
from overwatch_sayak.detection.tools.smoother import DetectionsSmoother
from overwatch_sayak.detection.utils import (
    box_iou_batch,
    calculate_masks_centroids,
    clip_boxes,
    contains_holes,
    contains_multiple_segments,
    filter_polygons_by_area,
    mask_iou_batch,
    mask_to_polygons,
    mask_to_xyxy,
    move_boxes,
    move_masks,
    pad_boxes,
    polygon_to_mask,
    polygon_to_xyxy,
    scale_boxes,
)
from overwatch_sayak.draw.color import Color, ColorPalette
from overwatch_sayak.draw.utils import (
    calculate_optimal_line_thickness,
    calculate_optimal_text_scale,
    draw_filled_rectangle,
    draw_image,
    draw_line,
    draw_polygon,
    draw_rectangle,
    draw_text,
)
from overwatch_sayak.geometry.core import Point, Position, Rect
from overwatch_sayak.geometry.utils import get_polygon_center
from overwatch_sayak.keypoint.annotators import (
    EdgeAnnotator,
    VertexAnnotator,
    VertexLabelAnnotator,
)
from overwatch_sayak.keypoint.core import KeyPoints
from overwatch_sayak.metrics.detection import ConfusionMatrix, MeanAveragePrecision
from overwatch_sayak.tracker.track.core import ByteTrack
from overwatch_sayak.utils.conversion import cv2_to_pillow, pillow_to_cv2
from overwatch_sayak.utils.file import list_files_with_extensions
from overwatch_sayak.utils.image import (
    ImageSink,
    create_tiles,
    crop_image,
    letterbox_image,
    overlay_image,
    resize_image,
    scale_image,
)
from overwatch_sayak.utils.notebook import plot_image, plot_images_grid
from overwatch_sayak.utils.video import (
    FPSMonitor,
    VideoInfo,
    VideoSink,
    get_video_frames_generator,
    process_video,
)