import os
import random
import string
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from jinja2 import Template
from pydantic import BaseModel, model_validator


def submit_job(
    job_script_path, interactive, nodes, partition, time, account, environment
):
    if interactive:
        # cmd = [
        #     "srun",
        #     f"--nodes={nodes}",
        #     f"--partition={partition}",
        #     f"--time={time}",
        #     f"--account={account}",
        #     "--exclusive",
        #     "--pty",
        # ]

        # if environment:
        #     cmd.extend(["--container-writable", f"--environment={environment}"])

        # cmd.append("bash")

        # # logging.info(f"Launching interactive session with command: {' '.join(cmd)}")
        # # logging.info("Press Ctrl+D or type 'exit' to end the session")

        # try:
        #     subprocess.run(cmd, check=True)
        #     # logging.info("Interactive session ended")
        #     return None
        # except subprocess.CalledProcessError as e:
        #     # logging.error(f"Error launching interactive session: {e}")
        #     raise
        # except KeyboardInterrupt:
        #     # logging.info("\nInteractive session interrupted")
        #     return None
        pass
    else:
        try:
            result = subprocess.run(
                ["sbatch", job_script_path], capture_output=True, text=True, check=True
            )
            job_id = result.stdout.strip().split("\n")[-1].split()[-1]
            # logging.info(f"Job submitted successfully with ID: {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            # logging.error(f"Error submitting job: {e}")
            # logging.error(f"stderr: {e.stderr}")
            raise
        except (IndexError, ValueError):
            # logging.error(f"Error parsing job ID from sbatch output: {result.stdout}")
            raise


class TemplateArgs(BaseModel):
    job_name: str
    account: str = "infra01"
    nodes: int
    partition: str = "normal"
    time: str = "00:05:00"
    environment: str
    framework: str
    framework_args: str = ""
    pre_launch_cmds: str = ""
    workers: int = 1
    nodes_per_worker: Optional[int] = None
    worker_port: int = 5000
    use_router: bool = False
    router_environment: Optional[str] = None
    router_port: int = 30000
    router_args: str = ""
    disable_ocf: bool = False
    ocf_service_name: str = "llm"
    ocf_service_port: int = 8080

    @model_validator(mode="after")
    def set_defaults(self):
        if self.nodes_per_worker is None:
            self.nodes_per_worker = self.nodes // self.workers
        if self.router_environment is None:
            self.router_environment = self.environment
        return self


def generate_job_script(template_path, output_path, template_args: TemplateArgs):
    with open(template_path, "r") as f:
        template = Template(f.read())

    rendered_script = template.render(**template_args.model_dump())
    with open(output_path, "w") as f:
        f.write(rendered_script)


def main():
    whoami = os.environ.get("USER", os.popen("whoami").read().strip())
    job_name = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

    sglang_args = TemplateArgs(
        job_name=job_name,
        nodes=1,
        framework="sglang",
        environment=str(Path.cwd() / "serving/envs/sglang.toml"),
        framework_args=(
            "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 "
            f"--served-model-name swiss-ai/Apertus-8B-Instruct-2509-{whoami} "
            "--host 0.0.0.0 "
            "--port 8080 "
        ),
    )

    vllm_args = TemplateArgs(
        job_name=job_name,
        nodes=1,
        framework="vllm",
        environment=str(Path.cwd() / "serving/envs/vllm.toml"),
        framework_args=(
            "--model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 "
            f"--served-model-name swiss-ai/Apertus-8B-Instruct-2509-{whoami} "
            "--host 0.0.0.0 "
            "--port 8080 "
        ),
    )

    template_args = sglang_args

    template_path = Path(__file__).parent / "template.jinja"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as temp_file:
        generate_job_script(template_path, temp_file.name, template_args)
        submit_job(
            temp_file.name,
            interactive=False,
            nodes=template_args.nodes,
            partition=template_args.partition,
            time=template_args.time,
            account=template_args.account,
            environment=template_args.environment,
        )


if __name__ == "__main__":
    main()
