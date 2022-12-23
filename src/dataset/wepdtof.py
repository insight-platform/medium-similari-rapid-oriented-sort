from typing import Optional
from pathlib import Path
from collections import defaultdict
import json
import math
import re
import numpy as np

SEQUENCES = frozenset(
    (
        "call_center",
        "convenience_store",
        "empty_store",
        "exhibition_setup",
        "exhibition",
        "it_office",
        "jewelry_store",
        "jewelry_store_2",
        "kindergarten",
        "large_office",
        "large_office_2",
        "printing_store",
        "repair_store",
        "street_grocery",
        "tech_store",
        "warehouse",
    )
)

ANNOTATION_SOURCES = frozenset(
    ("GT", "rapid", "similari_iou", "similari_maha")
)

FRAME_N_PATTERN = re.compile(r"_(?P<frame_n>[0-9]+)$")


class WEPDTOFAnnotationsPaths:
    def __init__(self, data_root_path, sequence_name, annotations_source) -> None:
        self.data_root_path = data_root_path
        self.sequence_name = sequence_name
        self.annotations_source = annotations_source

    @property
    def frames_dir_path(self):
        return (
            self.data_root_path
            / "frames"
            / "_".join((self.sequence_name, self.annotations_source))
        )

    @property
    def json_file_path(self):
        if self.annotations_source == "GT":
            return self.data_root_path / "annotations" / f"{self.sequence_name}.json"
        if self.annotations_source == "rapid":
            return (
                self.data_root_path / "annotations_rapid" / f"{self.sequence_name}.json"
            )

        return (
            self.data_root_path
            / "trackers"
            / self.sequence_name
            / f"{self.annotations_source}.json"
        )


class WEPDTOFPrediction:
    def __init__(
        self,
        dataset_path: Path,
        sequence_name: str,
        annotations_src: Optional[str] = None,
        annotations_dst: Optional[str] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.sequence_name = sequence_name

        self.input_ann_paths = None
        if annotations_src is not None:
            self.input_ann_paths = WEPDTOFAnnotationsPaths(
                dataset_path, sequence_name, annotations_src
            )

        self.output_ann_paths = None
        if annotations_dst is not None:
            self.output_ann_paths = WEPDTOFAnnotationsPaths(
                dataset_path, sequence_name, annotations_dst
            )

        if self.input_ann_paths is not None:
            print(f"Reading annotations from {self.input_ann_paths.json_file_path}")
            with open(
                self.input_ann_paths.json_file_path, "r", encoding="utf8"
            ) as file_obj:
                detections_dict = json.load(file_obj)

            self.frame_annotations = defaultdict(list)
            for detection in detections_dict["annotations"]:
                # Each bounding box is represented by five numbers [cx, cy, w, h, d]
                # where “cx” and “cy” are coordinates of the center of the bounding box in pixels from the top-left image corner at [0,0]
                # “w” and “h” are its width and height in pixels
                # and “d” is its clock-wise rotation angle from the vertical axis pointing up, in degrees.
                # In order to avoid ambiguity, the bounding-box width is constrained to be less or equal to its height (w<=h)
                # the rotation angle is set to be within -90 to +90 degrees
                bbox = detection["bbox"]
                obj_id = detection["person_id"]

                match = FRAME_N_PATTERN.search(detection["image_id"])
                frame_n = int(match["frame_n"])

                obj = bbox + [obj_id]
                if "confidence" in detection:
                    obj += [detection["confidence"]]
                self.frame_annotations[frame_n].append(obj)

            self.data = tuple(
                np.array(objects)
                for _, objects in sorted(self.frame_annotations.items())
            )
        else:
            print("No input annotations specified.")
            self.data = []

    @property
    def original_frames_paths(self):
        frame_dir_path = self.dataset_path / "frames" / self.sequence_name
        return sorted(frame_dir_path.glob("*.jpg"))

    @property
    def visualization_dir(self):
        return self.input_ann_paths.frames_dir_path

    def write_results(self, results):
        self.output_ann_paths.json_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(
            self.output_ann_paths.json_file_path, "w", encoding="utf8"
        ) as file_obj:
            json.dump(results, file_obj)
        return self.output_ann_paths.json_file_path

    def convert_similari_results(self, results):
        annotations = {"annotations": []}
        for i, tracks in enumerate(results):
            for track in tracks:
                if track.predicted_bbox.angle is not None:
                    angle = track.predicted_bbox.angle * 180 / math.pi
                else:
                    angle = 0

                bbox = [
                    track.predicted_bbox.xc,
                    track.predicted_bbox.yc,
                    track.predicted_bbox.aspect * track.predicted_bbox.height,
                    track.predicted_bbox.height,
                    angle,
                ]

                annotations["annotations"].append(
                    {
                        "bbox": bbox,
                        "person_id": track.id,
                        "image_id": f"{self.sequence_name}_{i+1:06d}",
                    }
                )

        return annotations

    def convert_rapid_results(self, results):
        annotations = {"annotations": []}
        for i, bboxes in enumerate(results):
            for bbox in bboxes:
                if len(bbox) == 6:
                    x, y, w, h, a, conf = bbox
                else:
                    x, y, w, h, a = bbox[:5]
                    conf = 0.01

                bbox_annotation = [x.item(), y.item(), w.item(), h.item(), a.item()]
                annotations["annotations"].append(
                    {
                        "bbox": bbox_annotation,
                        "person_id": -1,
                        "image_id": f"{self.sequence_name}_{i+1:06d}",
                        "confidence": conf
                    }
                )

        return annotations
