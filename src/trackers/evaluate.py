import argparse
from pathlib import Path
import trackeval
from dataset import SEQUENCES
from .dataset_trackeval import WEPDTOFEval

NUM_CORES = 1

def parse_args():
    parser = argparse.ArgumentParser()
    default_dataset_path = Path('/opt/data/WEPDTOF')
    parser.add_argument(
        'trackers',
        nargs='+',
        choices=[
            'similari_iou',
            'similari_maha'
        ],
    )
    parser.add_argument(
        '--detections',
        nargs='+',
        choices=[
            'GT',
            'rapid',
        ],
        default=['GT']
    )
    parser.add_argument('--sequences', nargs='+', choices=SEQUENCES, default=SEQUENCES)
    parser.add_argument('--dataset-path', type=Path, default=default_dataset_path)
    return parser.parse_args()


def main(args):
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = True
    eval_config['PRINT_CONFIG'] = False
    eval_config['OUTPUT_DETAILED'] = True
    eval_config['PLOT_CURVES'] = False
    if NUM_CORES > 1:
        eval_config['USE_PARALLEL'] = True
        eval_config['NUM_PARALLEL_CORES'] = NUM_CORES

    evaluator = trackeval.Evaluator(eval_config)

    trackers = []
    for dets in args.detections:
        for tracker in args.trackers:
            trackers.append(
                '_'.join((dets, tracker))
            )

    dataset_list = [
        WEPDTOFEval({
            'GT_FOLDER': args.dataset_path / 'annotations',
            'TRACKERS_FOLDER': args.dataset_path / 'trackers',
            'OUTPUT_FOLDER': args.dataset_path / 'eval_output',
            'SEQUENCES': args.sequences,
            'TRACKERS_TO_EVAL': trackers
        })
    ]

    metrics_list = [
        # trackeval.metrics.HOTA(),
        # Similarity score threshold required for a TP match. Default 0.5.
        trackeval.metrics.CLEAR(config={'THRESHOLD': 0.5}),
        # Similarity score threshold required for a IDTP match. Default 0.5.
        trackeval.metrics.Identity(config={'THRESHOLD': 0.5}),
    ]

    evaluator.evaluate(dataset_list, metrics_list)

if __name__ == '__main__':
    main(parse_args())
