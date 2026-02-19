import sys

from agentlab.experiments.study import Study

n_jobs = 1
parallel_backend = "sequential"

if len(sys.argv) > 1:
    study_path = sys.argv[1]

backend_parameter = None
if len(sys.argv) > 2:
    backend_parameter = sys.argv[2]


if backend_parameter == "ray":
    n_jobs = 10
    parallel_backend = "ray"

if __name__ == "__main__":
    study = Study.load(study_path)
    study.find_incomplete(include_errors=True)
    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        n_relaunch=2,
        relaunch_errors=True,
    )
