import luigi

from .labelstudio_to_raw import CombatPhaseLabelStudioToRaw
from .raw_to_discrete import CombatPhaseRawToDiscrete


class Workflow(luigi.Task):
    input_json_path = luigi.Parameter()
    data_path = luigi.Parameter()
    output_path = luigi.Parameter()
    interval_duration = luigi.IntParameter(default=1)

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/labels.json")

    def run(self):
        raw_task = CombatPhaseLabelStudioToRaw(
            input_json_path=self.input_json_path,
            data_path=self.data_path,
            output_json_path=f"{self.output_path}/raw.json",
        )
        yield raw_task

        yield CombatPhaseRawToDiscrete(
            input_json_path=raw_task.output().path,
            output_json_path=self.output().path,
            interval_duration=self.interval_duration,
        )


root_path = "/cs-share/pradalier/tmp/judo"
workflow_path = f"{root_path}/data/combat_phase"
luigi.build(
    [
        Workflow(
            data_path=f"{root_path}/data",
            input_json_path=f"{workflow_path}/project-16-at-2024-03-28-14-17-d0fa284f.json",
            output_path=f"{workflow_path}/discrete_v2",
        )
    ]
)
