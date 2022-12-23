"""Custom dataset module for TrackEval. Works with rotated bboxes."""
from pathlib import Path
from collections import defaultdict
import json
import re
import numpy as np
import cv2
import pycocotools.mask as mask_utils
from trackeval import _timing
from trackeval.utils import init_config
from trackeval.datasets._base_dataset import _BaseDataset

FRAME_N_PATTERN = re.compile(r"_(?P<frame_n>[0-9]+)$")


def get_frame_n(obj_annotation):
    match = FRAME_N_PATTERN.search(obj_annotation["image_id"])
    return int(match["frame_n"])


class WEPDTOFEval(_BaseDataset):
    @staticmethod
    def get_default_dataset_config():
        dataset_dir = Path("/") / "opt" / "data" / "WEPDTOF"
        return {
            "GT_FOLDER": dataset_dir / "annotations",
            "TRACKERS_FOLDER": dataset_dir / "trackers" / "annotations",
            "OUTPUT_FOLDER": dataset_dir / "eval_output",
            "TRACKERS_TO_EVAL": [],
            "CLASSES_TO_EVAL": ["person"],
            "SEQUENCES": ["exhibition"],
            "PRINT_CONFIG": True,
        }

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()

        self.config = init_config(
            config, self.get_default_dataset_config(), self.get_name()
        )

        self.gt_fol = self.config["GT_FOLDER"]
        self.trackers_fol = self.config["TRACKERS_FOLDER"]

        self.output_fol = self.config["OUTPUT_FOLDER"]
        self.output_sub_fol = ""

        self.seq_list = self.config["SEQUENCES"]
        self.class_list = self.config["CLASSES_TO_EVAL"]

        self.tracker_list = self.config["TRACKERS_TO_EVAL"]

        self.image_sizes = {
            "call_center": (1920, 1080),
            "convenience_store": (960, 720),
            "empty_store": (1440, 1080),
            "exhibition_setup": (1920, 1080),
            "exhibition": (1920, 1080),
            "it_office": (1440, 1080),
            "jewelry_store": (1620, 1080),
            "jewelry_store_2": (1620, 1080),
            "kindergarten": (900, 720),
            "large_office": (1350, 1080),
            "large_office_2": (1350, 1080),
            "printing_store": (1440, 1440),
            "repair_store": (900, 720),
            "street_grocery": (944, 1080),
            "tech_store": (2592, 1944),
            "warehouse": (1920, 1080),
        }

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker)
        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets]: list (for each timestep) of lists of detections.
        [gt_ignore_region]: list (for each timestep) of masks for the ignore regions
        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """

        num_timesteps = self.__get_num_timesteps(seq)

        if is_gt:
            data_path = self.gt_fol / f"{seq}.json"
        else:
            data_path = self.trackers_fol / seq / f"{tracker}.json"

        with open(data_path, "r", encoding="utf8") as file_obj:
            read_data = json.load(file_obj)

        printing_store_dup_skip = True
        # group objects by frame
        frames_objects = defaultdict(list)
        for ann in read_data["annotations"]:
            correct_annotation = True
            # Each bounding box is represented by five numbers [cx, cy, w, h, d]
            # where “cx” and “cy” are coordinates of the center of the bounding box in pixels from the top-left image corner at [0,0]
            # “w” and “h” are its width and height in pixels
            # and “d” is its clock-wise rotation angle from the vertical axis pointing up, in degrees.
            # In order to avoid ambiguity, the bounding-box width is constrained to be less or equal to its height (w<=h)
            # the rotation angle is set to be within -90 to +90 degrees
            bbox = ann["bbox"]
            obj_id = ann["person_id"]
            frame_n = get_frame_n(ann) - 1

            # hacky workarounds for bugs in GT
            # there are duplicated obj_ids on some frames of call center video
            if seq == 'call_center' and is_gt and obj_id == 33 and ann["area"] < 300:
                obj_id = 330

            # printing store video has one duplicated object on one frame
            if seq == 'printing_store' and is_gt and obj_id == 7 and frame_n == 1008 and printing_store_dup_skip:
                correct_annotation = False
                printing_store_dup_skip = False

            if correct_annotation:
                frames_objects[frame_n].append(bbox + [obj_id])

        data_keys = ["ids", "classes", "dets"]
        if is_gt:
            data_keys += ["gt_ignore_region"]

        raw_data = {key: [None] * num_timesteps for key in data_keys}

        for frame_n in range(num_timesteps):
            if frame_n in frames_objects:
                objs = np.array(frames_objects[frame_n])

                bboxes_as_points = []
                for bbox in objs[:, :5]:
                    cx, cy, width, height, angle = bbox
                    rot_rect = ((cx, cy), (width, height), angle)
                    points = cv2.boxPoints(rot_rect)
                    points_flattened = []
                    for pt in points:
                        points_flattened.extend(pt)
                    bboxes_as_points.append(points_flattened)

                img_width, img_height = self.image_sizes[seq]
                raw_data["dets"][frame_n] = mask_utils.frPyObjects(
                    bboxes_as_points, img_height, img_width
                )
                raw_data["ids"][frame_n] = np.atleast_1d(objs[:, 5]).astype(int)
                raw_data["classes"][frame_n] = np.ones_like(raw_data["ids"][frame_n])

            else:
                raw_data["dets"][frame_n] = []
                raw_data["ids"][frame_n] = np.empty(0).astype(int)
                raw_data["classes"][frame_n] = np.empty(0).astype(int)

            if is_gt:
                raw_data["gt_ignore_region"][frame_n] = mask_utils.merge(
                    [], intersect=False
                )

        if is_gt:
            key_map = {"ids": "gt_ids", "classes": "gt_classes", "dets": "gt_dets"}
        else:
            key_map = {
                "ids": "tracker_ids",
                "classes": "tracker_classes",
                "dets": "tracker_dets",
            }
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data["num_timesteps"] = num_timesteps
        raw_data["seq"] = seq
        return raw_data

    def __get_num_timesteps(self, seq):
        gt_path = self.gt_fol / f"{seq}.json"

        with open(gt_path, "r", encoding="utf8") as file_obj:
            gt_data = json.load(file_obj)

        frame_ns = []
        for ann in gt_data["annotations"]:
            frame_n = get_frame_n(ann)
            frame_ns.append(frame_n)

        return max(frame_ns)

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        """
        self._check_unique_ids(raw_data)

        data_keys = [
            "gt_ids",
            "tracker_ids",
            "gt_dets",
            "tracker_dets",
            "similarity_scores",
        ]
        data = {key: [None] * raw_data["num_timesteps"] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0

        for frame_n in range(raw_data["num_timesteps"]):
            data["tracker_ids"][frame_n] = raw_data["tracker_ids"][frame_n]
            data["tracker_dets"][frame_n] = raw_data["tracker_dets"][frame_n]
            data["gt_ids"][frame_n] = raw_data["gt_ids"][frame_n]
            data["gt_dets"][frame_n] = raw_data["gt_dets"][frame_n]
            data["similarity_scores"][frame_n] = raw_data["similarity_scores"][frame_n]

            unique_gt_ids += list(np.unique(data["gt_ids"][frame_n]))
            unique_tracker_ids += list(np.unique(data["tracker_ids"][frame_n]))
            num_tracker_dets += len(data["tracker_ids"][frame_n])
            num_gt_dets += len(data["gt_ids"][frame_n])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["gt_ids"][t]) > 0:
                    data["gt_ids"][t] = gt_id_map[data["gt_ids"][t]].astype(np.int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["tracker_ids"][t]) > 0:
                    data["tracker_ids"][t] = tracker_id_map[
                        data["tracker_ids"][t]
                    ].astype(np.int)

        # Record overview statistics.
        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = len(unique_gt_ids)
        data["num_timesteps"] = raw_data["num_timesteps"]
        data["seq"] = raw_data["seq"]
        data["cls"] = cls

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        """Detections format as was converted in _load_raw_file()
        Meaning, RLE masks
        """
        return self._calculate_mask_ious(
            gt_dets_t, tracker_dets_t, is_encoded=True, do_ioa=False
        )
