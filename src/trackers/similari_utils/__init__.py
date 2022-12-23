from similari import Sort, SpatioTemporalConstraints, PositionalMetricType
from .inputs import SimilariInput


class SimilariWrapper:
    def __init__(
        self,
        metric,
        constraints,
        min_hits: int = 1,
        max_age: int = 5,
        add_idle: bool = True,
    ) -> None:

        self.tracker = Sort(
            shards=4,
            bbox_history=2 * max_age,
            max_idle_epochs=max_age,
            method=metric,
            spatio_temporal_constraints=constraints,
        )

        self.min_hits = min_hits
        self.add_idle = add_idle

    def process_frame(self, inputs):
        tracks = self.tracker.predict(inputs)

        if self.add_idle:
            tracks.extend(self.tracker.idle_tracks())

        self.tracker.clear_wasted()
        return filter(lambda t: t.length >= self.min_hits, tracks)


def init_similari_sort_iou(dataset):
    metric = PositionalMetricType.iou(threshold=0.3)
    constraints = SpatioTemporalConstraints()
    constraints.add_constraints([(1, 1.0)])
    tracker = SimilariWrapper(metric, constraints, min_hits=3, max_age=15)
    inputs = SimilariInput(dataset)

    return tracker, inputs


def init_similari_sort_maha(dataset):
    metric = PositionalMetricType.maha()
    constraints = SpatioTemporalConstraints()
    constraints.add_constraints([(1, 1.0)])
    tracker = SimilariWrapper(metric, constraints, min_hits=3, max_age=15)
    inputs = SimilariInput(dataset)

    return tracker, inputs
