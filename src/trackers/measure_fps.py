from pathlib import Path
import argparse
import cProfile
import pstats
import numpy as np
from dataset import WEPDTOFPrediction, SEQUENCES
from .rotated_bboxes_sort import init_rotated_bboxes_sort
from .similari_utils import init_similari_sort_iou, init_similari_sort_maha


def parse_args():
    parser = argparse.ArgumentParser()
    default_dataset_path = Path('/opt/data/WEPDTOF')
    parser.add_argument(
        'tracker',
        choices=[
            'similari_iou',
            'similari_maha'
        ],
    )
    parser.add_argument(
        '--detections',
        choices=[
            'GT',
            'rapid',
        ],
        default='GT'
    )
    parser.add_argument('--sequence', choices=SEQUENCES, default='exhibition')
    parser.add_argument('--dataset-path', type=Path, default=default_dataset_path)
    parser.add_argument('--repeat-n', type=int, default=10)
    return parser.parse_args()


def main(args):
    dataset = WEPDTOFPrediction(args.dataset_path, args.sequence, annotations_src=args.detections)
    fps_vals = []
    per_call_vals = []
    for _ in range(args.repeat_n+2):
        stats = run_tracking(dataset, args.tracker)
        stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(1)

        trk_fnc_stats = stats.get_stats_profile().func_profiles["process_frame"]
        per_call = trk_fnc_stats.cumtime / int(trk_fnc_stats.ncalls)
        fps = 1 / per_call

        fps_vals.append(fps)
        per_call_vals.append(per_call)
        
    print(f'mean per call {np.mean(per_call_vals[1:-1]) * 1000:.2f} ms, mean fps {np.mean(fps_vals[1:-1]):.2f}')


def run_tracking(dataset, tracker):

    if tracker == 'similari_iou':
        tracker, inputs = init_similari_sort_iou(dataset)
    elif tracker == 'similari_maha':
        tracker, inputs = init_similari_sort_maha(dataset)

    with cProfile.Profile() as pr:
        for frame_detections in inputs.data:
            tracker.process_frame(frame_detections)
    return pstats.Stats(pr)


if __name__ == '__main__':
    main(parse_args())
