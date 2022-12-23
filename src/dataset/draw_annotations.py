from pathlib import Path
import argparse
import shutil
import numpy as np
import cv2
from .wepdtof import WEPDTOFPrediction, SEQUENCES


def parse_args():
    parser = argparse.ArgumentParser()
    default_dataset_path = Path("/opt/data/WEPDTOF")
    parser.add_argument(
        "annotations",
        choices=[
            "GT_similari_iou",
            "GT_similari_maha",
            "rapid_similari_iou",
            "rapid_similari_maha",
            "GT",
            "rapid",
        ],
    )
    parser.add_argument("--sequence", choices=SEQUENCES, default="exhibition")
    parser.add_argument("--dataset-path", type=Path, default=default_dataset_path)
    return parser.parse_args()


def main(args):

    dataset = WEPDTOFPrediction(
        args.dataset_path, args.sequence, annotations_src=args.annotations
    )

    shutil.rmtree(dataset.visualization_dir, ignore_errors=True)
    dataset.visualization_dir.mkdir(parents=True)

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    font_color = (255, 255, 255)
    text_offset = np.array([0, -10]).reshape((1, 1, -1)).astype(np.int32)

    for i, frame_path in enumerate(dataset.original_frames_paths):
        img = cv2.imread(str(frame_path))

        frame_n = i + 1
        try:
            objs = dataset.frame_annotations[frame_n]
        except KeyError:
            objs = []

        for obj in objs:
            xc, yc, width, height, degrees, obj_id = obj

            points = cv2.boxPoints(((xc, yc), (width, height), degrees))
            pts = np.array(points).reshape((1, -1, 1, 2)).astype(np.int32)
            cv2.polylines(img, pts, True, (0, 255, 0), 2)

            # obj_id == -1 is used as a placeholder in detections annotations
            if obj_id != -1:
                text = f"#{obj_id}"

                text_size, baseline = cv2.getTextSize(
                    text, font_face, font_scale, font_thickness
                )

                point_text_orig = pts[0, 0, :] + text_offset

                rect_tl = point_text_orig + np.array([0, baseline]).reshape(
                    (1, 1, -1)
                ).astype(np.int32)

                rect_br = point_text_orig + np.array(
                    [text_size[0], -text_size[1]]
                ).reshape((1, 1, -1)).astype(np.int32)

                cv2.rectangle(
                    img, rect_tl.ravel(), rect_br.ravel(), (0, 0, 0), cv2.FILLED
                )

                img = cv2.putText(
                    img,
                    text,
                    point_text_orig.ravel(),
                    font_face,
                    font_scale,
                    font_color,
                    font_thickness,
                    cv2.LINE_AA,
                )

        output_path = dataset.visualization_dir / f"{frame_n:06d}.jpg"
        cv2.imwrite(str(output_path), img)


if __name__ == "__main__":
    main(parse_args())
