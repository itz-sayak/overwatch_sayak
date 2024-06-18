from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from overwatch_sayak.config import CLASS_NAME_DATA_FIELD
from overwatch_sayak.detection.utils import get_data_item, is_data_equal
from overwatch_sayak.validators import validate_keypoints_fields


@dataclass
class KeyPoints:
    """
    The `ov.KeyPoints` class in the Supervision library standardizes results from
    various keypoint detection and pose estimation models into a consistent format. This
    class simplifies data manipulation and filtering, providing a uniform API for
    integration with Supervision [keypoints annotators](/keypoint/annotators).

    === "Ultralytics"

        Use [`ov.KeyPoints.from_ultralytics`](/keypoint/core/#overwatch.keypoint.core.KeyPoints.from_ultralytics)
        method, which accepts [YOLOv8](https://github.com/ultralytics/ultralytics)
        pose result.

        ```python
        import cv2
        import overwatch as ov
        from ultralytics import YOLO

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        model = YOLO('yolov8s-pose.pt')

        result = model(image)[0]
        key_points = ov.KeyPoints.from_ultralytics(result)
        ```

    === "Inference"

        Use [`ov.KeyPoints.from_inference`](/keypoint/core/#overwatch.keypoint.core.KeyPoints.from_inference)
        method, which accepts [Inference](https://inference.roboflow.com/) pose result.

        ```python
        import cv2
        import overwatch as ov
        from inference import get_model

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        model = get_model(model_id=<POSE_MODEL_ID>, api_key=<ROBOFLOW_API_KEY>)

        result = model.infer(image)[0]
        key_points = ov.KeyPoints.from_inference(result)
        ```

    === "MediaPipe"

        Use [`ov.KeyPoints.from_mediapipe`](/keypoint/core/#overwatch.keypoint.core.KeyPoints.from_mediapipe)
        method, which accepts [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        pose result.

        ```python
        import cv2
        import mediapipe as mp
        import overwatch as ov

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        image_height, image_width, _ = image.shape
        mediapipe_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path="pose_landmarker_heavy.task"
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=2)

        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        with PoseLandmarker.create_from_options(options) as landmarker:
            pose_landmarker_result = landmarker.detect(mediapipe_image)

        key_points = ov.KeyPoints.from_mediapipe(
            pose_landmarker_result, (image_width, image_height))
        ```

    Attributes:
        xy (np.ndarray): An array of shape `(n, 2)` containing
            the bounding boxes coordinates in format `[x1, y1]`
        confidence (Optional[np.ndarray]): An array of shape
            `(n,)` containing the confidence scores of the keypoint keypoints.
        class_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the class ids of the keypoint keypoints.
        data (Dict[str, Union[np.ndarray, List]]): A dictionary containing additional
            data where each key is a string representing the data type, and the value
            is either a NumPy array or a list of corresponding data.
    """  # noqa: E501 // docs

    xy: npt.NDArray[np.float32]
    class_id: Optional[npt.NDArray[np.int_]] = None
    confidence: Optional[npt.NDArray[np.float32]] = None
    data: Dict[str, Union[npt.NDArray[Any], List]] = field(default_factory=dict)

    def __post_init__(self):
        validate_keypoints_fields(
            xy=self.xy,
            confidence=self.confidence,
            class_id=self.class_id,
            data=self.data,
        )

    def __len__(self) -> int:
        """
        Returns the number of keypoints in the `ov.KeyPoints` object.
        """
        return len(self.xy)

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            np.ndarray,
            Optional[np.ndarray],
            Optional[float],
            Optional[int],
            Optional[int],
            Dict[str, Union[np.ndarray, List]],
        ]
    ]:
        """
        Iterates over the Keypoint object and yield a tuple of
        `(xy, confidence, class_id, data)` for each keypoint detection.
        """
        for i in range(len(self.xy)):
            yield (
                self.xy[i],
                self.confidence[i] if self.confidence is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                get_data_item(self.data, i),
            )

    def __eq__(self, other: KeyPoints) -> bool:
        return all(
            [
                np.array_equal(self.xy, other.xy),
                np.array_equal(self.class_id, other.class_id),
                np.array_equal(self.confidence, other.confidence),
                is_data_equal(self.data, other.data),
            ]
        )

    @classmethod
    def from_inference(cls, inference_result: Union[dict, Any]) -> KeyPoints:
        """
        Create a `ov.KeyPoints` object from the [Roboflow](https://roboflow.com/)
        API inference result or the [Inference](https://inference.roboflow.com/)
        package results.

        Args:
            inference_result (dict, any): The result from the
                Roboflow API or Inference package containing predictions with keypoints.

        Returns:
            A `ov.KeyPoints` object containing the keypoint coordinates, class IDs,
                and class names, and confidences of each keypoint.

        Examples:
            ```python
            import cv2
            import overwatch as ov
            from inference import get_model

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = get_model(model_id=<POSE_MODEL_ID>, api_key=<ROBOFLOW_API_KEY>)

            result = model.infer(image)[0]
            key_points = ov.KeyPoints.from_inference(result)
            ```

            ```python
            import cv2
            import overwatch as ov
            from inference_sdk import InferenceHTTPClient

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=<ROBOFLOW_API_KEY>
            )

            result = client.infer(image, model_id=<POSE_MODEL_ID>)
            key_points = ov.KeyPoints.from_inference(result)
            ```
        """
        if isinstance(inference_result, list):
            raise ValueError(
                "from_inference() operates on a single result at a time."
                "You can retrieve it like so:  inference_result = model.infer(image)[0]"
            )

        with suppress(AttributeError):
            inference_result = inference_result.dict(exclude_none=True, by_alias=True)

        if not inference_result.get("predictions"):
            return cls.empty()

        xy = []
        confidence = []
        class_id = []
        class_names = []

        for prediction in inference_result["predictions"]:
            prediction_xy = []
            prediction_confidence = []
            for keypoint in prediction["keypoints"]:
                prediction_xy.append([keypoint["x"], keypoint["y"]])
                prediction_confidence.append(keypoint["confidence"])
            xy.append(prediction_xy)
            confidence.append(prediction_confidence)

            class_id.append(prediction["class_id"])
            class_names.append(prediction["class"])

        data = {CLASS_NAME_DATA_FIELD: np.array(class_names)}

        return cls(
            xy=np.array(xy, dtype=np.float32),
            confidence=np.array(confidence, dtype=np.float32),
            class_id=np.array(class_id, dtype=int),
            data=data,
        )

    @classmethod
    def from_mediapipe(
        cls, mediapipe_results, resolution_wh: Tuple[int, int]
    ) -> KeyPoints:
        """
        Creates a `ov.KeyPoints` instance from a
        [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        pose landmark detection inference result.

        Args:
            mediapipe_results (Union[PoseLandmarkerResult, SolutionOutputs]):
                The output results from Mediapipe. It supports both: the inference
                result `PoseLandmarker` and the legacy one from `Pose`.
            resolution_wh (Tuple[int, int]): A tuple of the form `(width, height)`
                representing the resolution of the frame.

        Returns:
            A `ov.KeyPoints` object containing the keypoint coordinates and
                confidences of each keypoint.

        !!! tip
            Before you start, download model bundles from the
            [MediaPipe website](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models).

        Examples:
            ```python
            import cv2
            import mediapipe as mp
            import overwatch as ov

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            image_height, image_width, _ = image.shape
            mediapipe_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path="pose_landmarker_heavy.task"
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=2)

            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            with PoseLandmarker.create_from_options(options) as landmarker:
                pose_landmarker_result = landmarker.detect(mediapipe_image)

            key_points = ov.KeyPoints.from_mediapipe(
                pose_landmarker_result, (image_width, image_height))
            ```
        """  # noqa: E501 // docs
        results = mediapipe_results.pose_landmarks
        if not isinstance(mediapipe_results.pose_landmarks, list):
            if mediapipe_results.pose_landmarks is None:
                results = []
            else:
                results = [
                    [landmark for landmark in mediapipe_results.pose_landmarks.landmark]
                ]

        if len(results) == 0:
            return cls.empty()

        xy = []
        confidence = []
        for pose in results:
            prediction_xy = []
            prediction_confidence = []
            for landmark in pose:
                keypoint_xy = [
                    landmark.x * resolution_wh[0],
                    landmark.y * resolution_wh[1],
                ]
                prediction_xy.append(keypoint_xy)
                prediction_confidence.append(landmark.visibility)

            xy.append(prediction_xy)
            confidence.append(prediction_confidence)

        return cls(
            xy=np.array(xy, dtype=np.float32),
            confidence=np.array(confidence, dtype=np.float32),
        )

    @classmethod
    def from_ultralytics(cls, ultralytics_results) -> KeyPoints:
        """
        Creates a `ov.KeyPoints` instance from a
        [YOLOv8](https://github.com/ultralytics/ultralytics) pose inference result.

        Args:
            ultralytics_results (ultralytics.engine.results.Keypoints):
                The output Results instance from YOLOv8

        Returns:
            A `ov.KeyPoints` object containing the keypoint coordinates, class IDs,
                and class names, and confidences of each keypoint.

        Examples:
            ```python
            import cv2
            import overwatch as ov
            from ultralytics import YOLO

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = YOLO('yolov8s-pose.pt')

            result = model(image)[0]
            key_points = ov.KeyPoints.from_ultralytics(result)
            ```
        """
        if ultralytics_results.keypoints.xy.numel() == 0:
            return cls.empty()

        xy = ultralytics_results.keypoints.xy.cpu().numpy()
        class_id = ultralytics_results.boxes.cls.cpu().numpy().astype(int)
        class_names = np.array([ultralytics_results.names[i] for i in class_id])

        confidence = ultralytics_results.keypoints.conf.cpu().numpy()
        data = {CLASS_NAME_DATA_FIELD: class_names}
        return cls(xy, class_id, confidence, data)

    @classmethod
    def from_yolo_nas(cls, yolo_nas_results) -> KeyPoints:
        """
        Create a `ov.KeyPoints` instance from a [YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS-POSE.md)
        pose inference results.

        Args:
            yolo_nas_results (ImagePoseEstimationPrediction): The output object from
                YOLO NAS.

        Returns:
            A `ov.KeyPoints` object containing the keypoint coordinates, class IDs,
                and class names, and confidences of each keypoint.

        Examples:
            ```python
            import cv2
            import torch
            import overwatch as ov
            import super_gradients

            image = cv2.imread(<SOURCE_IMAGE_PATH>)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = super_gradients.training.models.get(
                "yolo_nas_pose_s", pretrained_weights="coco_pose").to(device)

            results = model.predict(image, conf=0.1)
            key_points = ov.KeyPoints.from_yolo_nas(results)
            ```
        """  # noqa: E501 // docs
        if len(yolo_nas_results.prediction.poses) == 0:
            return cls.empty()

        xy = yolo_nas_results.prediction.poses[:, :, :2]
        confidence = yolo_nas_results.prediction.poses[:, :, 2]

        # yolo_nas_results treats params differently.
        # prediction.labels may not exist, whereas class_names might be None
        if hasattr(yolo_nas_results.prediction, "labels"):
            class_id = yolo_nas_results.prediction.labels  # np.array[int]
        else:
            class_id = None

        data = {}
        if class_id is not None and yolo_nas_results.class_names is not None:
            class_names = []
            for c_id in class_id:
                name = yolo_nas_results.class_names[c_id]  # tuple[str]
                class_names.append(name)
            data[CLASS_NAME_DATA_FIELD] = class_names

        return cls(
            xy=xy,
            confidence=confidence,
            class_id=class_id,
            data=data,
        )

    def __getitem__(
        self, index: Union[int, slice, List[int], np.ndarray, str]
    ) -> Union[KeyPoints, List, np.ndarray, None]:
        """
        Get a subset of the `ov.KeyPoints` object or access an item from its data field.

        When provided with an integer, slice, list of integers, or a numpy array, this
        method returns a new `ov.KeyPoints` object that represents a subset of the
        original `ov.KeyPoints`. When provided with a string, it accesses the
        corresponding item in the data dictionary.

        Args:
            index (Union[int, slice, List[int], np.ndarray, str]): The index, indices,
                or key to access a subset of the `ov.KeyPoints` or an item from the
                data.

        Returns:
            A subset of the `ov.KeyPoints` object or an item from the data field.

        Examples:
            ```python
            import overwatch as ov

            key_points = ov.KeyPoints()

            # access the first keypoint using an integer index
            key_points[0]

            # access the first 10 keypoints using index slice
            key_points[0:10]

            # access selected keypoints using a list of indices
            key_points[[0, 2, 4]]

            # access keypoints with selected class_id
            key_points[key_points.class_id == 0]

            # access keypoints with confidence greater than 0.5
            key_points[key_points.confidence > 0.5]
            ```
        """
        if isinstance(index, str):
            return self.data.get(index)
        if isinstance(index, int):
            index = [index]
        return KeyPoints(
            xy=self.xy[index],
            confidence=self.confidence[index] if self.confidence is not None else None,
            class_id=self.class_id[index] if self.class_id is not None else None,
            data=get_data_item(self.data, index),
        )

    def __setitem__(self, key: str, value: Union[np.ndarray, List]):
        """
        Set a value in the data dictionary of the `ov.KeyPoints` object.

        Args:
            key (str): The key in the data dictionary to set.
            value (Union[np.ndarray, List]): The value to set for the key.

        Examples:
            ```python
            import cv2
            import overwatch as ov
            from ultralytics import YOLO

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = YOLO('yolov8s.pt')

            result = model(image)[0]
            keypoints = ov.KeyPoints.from_ultralytics(result)

            keypoints['class_name'] = [
                 model.model.names[class_id]
                 for class_id
                 in keypoints.class_id
             ]
            ```
        """
        if not isinstance(value, (np.ndarray, list)):
            raise TypeError("Value must be a np.ndarray or a list")

        if isinstance(value, list):
            value = np.array(value)

        self.data[key] = value

    @classmethod
    def empty(cls) -> KeyPoints:
        """
        Create an empty Keypoints object with no keypoints.

        Returns:
            An empty `ov.KeyPoints` object.

        Examples:
            ```python
            import overwatch as ov

            key_points = ov.KeyPoints.empty()
            ```
        """
        return cls(xy=np.empty((0, 0, 2), dtype=np.float32))
