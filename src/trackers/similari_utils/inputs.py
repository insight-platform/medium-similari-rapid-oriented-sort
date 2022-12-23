import math
from similari import Universal2DBox


class SimilariInput:

    def __init__(self, dataset):
        self.data = []

        for datum in dataset.data:
            frame_bboxes = []
            for bbox in datum:

                if len(bbox) == 7:
                    conf = bbox[6]
                else:
                    conf = 1

                frame_bboxes.append(
                    (
                        Universal2DBox.new_with_confidence(
                            xc=bbox[0],
                            yc=bbox[1],
                            angle=bbox[4] * math.pi / 180,
                            aspect=bbox[2] / bbox[3],
                            height=bbox[3],
                            confidence=conf
                        ),
                        None
                    )
                )
            self.data.append(frame_bboxes)
