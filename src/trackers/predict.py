from pathlib import Path
import argparse
from dataset import WEPDTOFPrediction, SEQUENCES
from .rotated_bboxes_sort import init_rotated_bboxes_sort
from .similari_utils import init_similari_sort_iou, init_similari_sort_maha


def parse_args():
    parser = argparse.ArgumentParser()
    default_dataset_path = Path("/opt/data/WEPDTOF")
    parser.add_argument(
        "trackers",
        nargs="+",
        choices=["similari_iou", "similari_maha"],
    )
    parser.add_argument(
        "--detections",
        nargs="+",
        choices=[
            "GT",
            "rapid",
        ],
        default="GT",
    )
    parser.add_argument("--sequences", nargs='+', choices=SEQUENCES, default=SEQUENCES)
    parser.add_argument("--dataset-path", type=Path, default=default_dataset_path)
    return parser.parse_args()


def main(args):
    for seq in args.sequences:
        for dets in args.detections:
            for tracker in args.trackers:
                print(f"Initializing dataset, detections {dets}, tracker {tracker}, sequence {seq}")
                dataset = WEPDTOFPrediction(
                    args.dataset_path,
                    seq,
                    annotations_src=dets,
                    annotations_dst="_".join((dets, tracker)),
                )

                results = run_tracking(tracker, dataset)

                write_path = dataset.write_results(results)
                print(f"Finished. Results were written into {write_path}")


def run_tracking(tracker_name, dataset: WEPDTOFPrediction):

    if tracker_name == "similari_iou":
        tracker, inputs = init_similari_sort_iou(dataset)
    elif tracker_name == "similari_maha":
        tracker, inputs = init_similari_sort_maha(dataset)
    results_converter = dataset.convert_similari_results
    results = []
    for frame_detections in inputs.data:
        results.append(tracker.process_frame(frame_detections))
    return results_converter(results)


if __name__ == "__main__":
    main(parse_args())
