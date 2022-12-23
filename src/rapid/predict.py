from pathlib import Path
import argparse
from api import Detector
from dataset import WEPDTOFPrediction, SEQUENCES


def parse_args():
    parser = argparse.ArgumentParser()
    default_dataset_path = Path('/opt/data/WEPDTOF')
    parser.add_argument('--sequences', nargs='+', choices=SEQUENCES, default=SEQUENCES)
    parser.add_argument('--dataset-path', type=Path, default=default_dataset_path)
    return parser.parse_args()


def main(args):
    detector = Detector(
        model_name='rapid',
        weights_path='/opt/app/weights/RAPiD.ckpt',
        use_cuda=True
    )

    for seq in args.sequences:
        dataset = WEPDTOFPrediction(
            args.dataset_path,
            seq,
            annotations_dst='rapid'
        )

        results = []
        for img_path in dataset.original_frames_paths:
            detections = detector.detect_one(
                img_path=str(img_path),
                input_size=1024,
                conf_thres=0.3,
            )

            results.append(detections)

        wepdtof_results = dataset.convert_rapid_results(results)

        write_path = dataset.write_results(wepdtof_results)
        print(f'Finished. Results were written into {write_path}')


if __name__ == '__main__':
    main(parse_args())
