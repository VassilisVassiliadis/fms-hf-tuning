# Copyright The IBM Tuning Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import Any, Dict, Optional
import os
import typing

# Third Party
from aim.ext.resource import DEFAULT_SYSTEM_TRACKING_INT
from aim.hugging_face import AimCallback
import aim


class CustomAimCallback(AimCallback):

    def __init__(
        self,
        repo: Optional[str] = None,
        experiment: Optional[str] = None,
        system_tracking_interval: Optional[int] = DEFAULT_SYSTEM_TRACKING_INT,
        log_system_params: Optional[bool] = True,
        capture_terminal_logs: Optional[bool] = True,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ):

        self._additional_metrics = additional_metrics or {}

        super().__init__(
            repo,
            experiment,
            system_tracking_interval,
            log_system_params,
            capture_terminal_logs,
        )

    def on_train_end(self, args, state, control, **kwargs):

        if state.is_world_process_zero:
            run: aim.Run = self.experiment

            for k, v in self._additional_metrics.items():
                run.track(v, name=k, context={"scope": "additional_metrics"})

            env_variables = run["__system_params"]["env_variables"]
            cuda_devices = env_variables.get("CUDA_VISIBLE_DEVICES", "")
            cuda_devices = [int(x) for x in cuda_devices.split(",") if len(x) > 0]

            metrics = []

            for m in run.metrics():
                context = m.context.to_dict()

                if m.name == "__system_gpu" and context["gpu"] in cuda_devices:
                    metrics.append(
                        {
                            "name": m.name,
                            "values": m.values.values_list(),
                            "gpu": context["gpu"],
                        }
                    )
                elif m.name != "__system_gpu":
                    metrics.append({"name": m.name, "values": m.values.values_list()})

            run["metrics"] = metrics
            # Standard
            import json

            with open("aim_info.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "run_hash": run.hash,
                        "metrics": metrics,
                    },
                    f,
                )

        super().on_train_end(args=args, state=state, control=control, **kwargs)


def get_aimstack_callback(
    additional_metrics: Optional[Dict[str, Any]] = None,
):
    # Initialize a new run
    aim_server = os.environ.get("AIMSTACK_SERVER")
    aim_db = os.environ.get("AIMSTACK_DB")
    aim_experiment = os.environ.get("AIMSTACK_EXPERIMENT")
    if aim_experiment is None:
        aim_experiment = ""

    if aim_server:
        aim_callback = CustomAimCallback(
            repo="aim://" + aim_server + "/",
            experiment=aim_experiment,
            additional_metrics=additional_metrics,
        )
    elif aim_db:
        aim_callback = CustomAimCallback(repo=aim_db, experiment=aim_experiment)
    else:
        aim_callback = CustomAimCallback(
            experiment=aim_experiment, additional_metrics=additional_metrics
        )

    return aim_callback
