import json
from argparse import ArgumentParser

import luigi
import numpy as np
import pandas as pd

from judo_footage_analysis.utils import ensure_parent


class CombatPhaseRawToDiscrete(luigi.Task):
    input_json_path = luigi.Parameter()
    output_json_path = luigi.Parameter()
    interval_duration = luigi.IntParameter(default=5)

    def output(self):
        return luigi.LocalTarget(self.output_json_path)

    def run(self):
        df = pd.read_json(self.input_json_path)

        discrete_annotations = []
        for file, annotation in zip(df.file, df.annotations):
            # Get the min start time and max end time
            min_start_time = min([a["start"] for a in annotation])
            max_end_time = max([a["end"] for a in annotation])

            # Create the recording points
            recording_points = np.arange(
                min_start_time, max_end_time, self.interval_duration
            )
            recording_points = np.append(recording_points, max_end_time)

            for i, recording_point in enumerate(recording_points[:-1]):
                is_match = False
                is_active = False
                is_standing = False

                for a in annotation:
                    if a["labels"] == "Match" and not is_match:
                        if (
                            a["start"] <= recording_point
                            and a["end"] >= recording_points[i + 1]
                        ):
                            is_match = True
                    if a["labels"] == "Active" and not is_active:
                        if (
                            a["start"] <= recording_point
                            and a["end"] >= recording_points[i + 1]
                        ):
                            is_active = True
                    if a["labels"] == "Standing" and not is_standing:
                        if (
                            a["start"] <= recording_point
                            and a["end"] >= recording_points[i + 1]
                        ):
                            is_standing = True

                discrete_annotations.append(
                    {
                        "file": file,
                        "time": recording_point,
                        "is_match": int(is_match),
                        "is_active": int(is_active),
                        "is_standing": int(is_standing),
                    }
                )

        df = pd.DataFrame(discrete_annotations)
        print(df.head())
        # https://stackoverflow.com/questions/60029873/pandas-to-json-redundant-backslashes
        # https://stackoverflow.com/questions/71275766/backward-slash-when-converting-dataframe-to-json-file
        ensure_parent(self.output_json_path).write_text(
            json.dumps(
                df.to_dict(
                    orient="records",
                ),
                indent=2,
            )
        )


class Workflow(luigi.Task):
    input_json_path = luigi.Parameter()
    output_json_path = luigi.Parameter()
    interval_duration = luigi.IntParameter(default=5)

    def requires(self):
        return CombatPhaseRawToDiscrete(
            input_json_path=self.input_json_path,
            output_json_path=self.output_json_path,
            interval_duration=self.interval_duration,
        )

    def run(self):
        pass


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-json-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/data/combat_phase/filtered_annotations_v2.json",
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output-json-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/data/combat_phase/discrete_annotations_v2.json",
        help="Path to save the descritised annotations.",
    )
    parser.add_argument(
        "--interval-duration",
        type=int,
        default=5,
        help="Duration of each interval in seconds.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of workers for Luigi."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    luigi.build(
        [
            Workflow(
                input_json_path=args.input_json_path,
                output_json_path=args.output_json_path,
                interval_duration=args.interval_duration,
            )
        ]
    )
