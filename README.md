# Overwatch Sayak

## Overview
The powers of this library  `overwatch-sayak` package can be used for object detection in images or video, loading datasets, detection tracking, counting, detecting, Slicing Aided Hyper Inferencing for small object detection, etc. 

## Installation
you need to install the `overwatch-sayak` package. You can do this using pip:

```bash
pip install overwatch-sayak
```
For importing the library for your code, use the import command with:

```python
import overwatch_sayak
```
## Quickstart
### Models
overwatch-sayak was designed to be model-friendly. Just plug in any classification, detection, or segmentation model. We have created connectors for the most popular libraries like Ultralytics, Transformers, or MMDetection for your convenience.

#### Inference
```python
import cv2
import overwatch_sayak as ov
from inference import get_model

image = cv2.imread("path/to/your/image.jpg")
model = get_model("yolov8s-640")
result = model.infer(image)[0]
detections = ov.Detections.from_inference(result)

len(detections)
# Output: Number of detections
```
### Annotators
overwatch-sayak offers a wide range of highly customizable annotators, allowing you to compose the perfect visualization for your use case.

```python
import cv2
import overwatch_sayak as ov

image = cv2.imread("path/to/your/image.jpg")
detections = ov.Detections(...)

bounding_box_annotator = ov.BoundingBoxAnnotator()
annotated_frame = bounding_box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

# Display or save the annotated image
cv2.imshow("Annotated Image", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Or save the image
cv2.imwrite("annotated_image.jpg", annotated_frame)
```
### Slicing Inference for Small Object Detection
Slicing the image into smaller pieces can improve object detection accuracy for detecting small objects. The following example demonstrates how to use the InferenceSlicer for this purpose.

```python
import overwatch_sayak as ov
from inference import get_model
import cv2
import numpy as np

# Load the image
image = cv2.imread("/path/to/image.jpg")

# Load the model
model = get_model("yolov8s-640")

# Define the callback function for slicing inference
def slicer_callback(slice: np.ndarray) -> ov.Detections:
    result = model.infer(slice)[0]
    detections = ov.Detections.from_inference(result)
    return detections

# Create the slicer
slicer = ov.InferenceSlicer(
    callback=slicer_callback,
    slice_wh=(512, 512),
    overlap_ratio_wh=(0.4, 0.4),
    overlap_filter_strategy=ov.OverlapFilter.NONE
)

# Run the slicer on the image
detections = slicer(image)

# Annotate the image
annotated_frame = ov.BoundingBoxAnnotator().annotate(
    scene=image.copy(),
    detections=detections
)

# Display or save the annotated image
cv2.imshow("Annotated Image", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Or save the image
cv2.imwrite("annotated_image.jpg", annotated_frame)
```
### Datasets
overwatch-sayak offers a suite of utilities that enable you to load, split, merge, and save datasets in various supported formats..

```python
import overwatch_sayak as ov

dataset = ov.DetectionDataset.from_yolo(
    images_directory_path="path/to/images",
    annotations_directory_path="path/to/annotations",
    data_yaml_path="path/to/data.yaml"
)

dataset.classes
# Output: ['dog', 'person']

len(dataset)
# Output: Number of images in the dataset
```
#### Split
```python
train_dataset, test_dataset = dataset.split(split_ratio=0.7)
test_dataset, valid_dataset = test_dataset.split(split_ratio=0.5)

len(train_dataset), len(test_dataset), len(valid_dataset)
# Output: (Number of training images, Number of test images, Number of validation images)
```
#### Merge
```python
ds_1 = ov.DetectionDataset(...)
len(ds_1)
# Output: Number of images in ds_1
ds_1.classes
# Output: ['dog', 'person']

ds_2 = ov.DetectionDataset(...)
len(ds_2)
# Output: Number of images in ds_2
ds_2.classes
# Output: ['cat']

ds_merged = ov.DetectionDataset.merge([ds_1, ds_2])
len(ds_merged)
# Output: Number of images in the merged dataset
ds_merged.classes
# Output: ['cat', 'dog', 'person']
```
#### Save
```python
dataset.as_yolo(
    images_directory_path="path/to/save/images",
    annotations_directory_path="path/to/save/annotations",
    data_yaml_path="path/to/save/data.yaml"
)

dataset.as_pascal_voc(
    images_directory_path="path/to/save/images",
    annotations_directory_path="path/to/save/annotations"
)

dataset.as_coco(
    images_directory_path="path/to/save/images",
    annotations_path="path/to/save/annotations"
)
```
#### Convert
```python
ov.DetectionDataset.from_yolo(
    images_directory_path="path/to/load/images",
    annotations_directory_path="path/to/load/annotations",
    data_yaml_path="path/to/load/data.yaml"
).as_pascal_voc(
    images_directory_path="path/to/save/images",
    annotations_directory_path="path/to/save/annotations"
)
```

## Contributing
If you want to contribute to this project, feel free to open an issue or submit a pull request on GitHub.

## License
This project is licensed under the MIT License.



 
