import logging
import os
import sys
import tempfile
from typing import Optional

from pydantic import BaseModel
from utils import (
    extract_model_name,
    fetch_bootstrap_addresses,
    generate_job_script,
    nanoid,
    setup_logging,
    submit_job,
)


class TemplateArgs(BaseModel):
    job_name: Optional[str] = None
    account: str = "infra01"
    nodes: int
    partition: str = "normal"
    time: str = "04:00:00"
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
    ocf_bootstrap_addr: Optional[str] = None
    ocf_service_name: str = "llm"
    ocf_service_port: int = 8080


def main():
    setup_logging()

    whoami = os.environ.get("USER", os.popen("whoami").read().strip())

    template_args = TemplateArgs(
        nodes=1,
        framework="sglang",
        environment=os.path.join(os.getcwd(), "serving/envs/sglang.toml"),
        framework_args=(
            "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509"
            f" --served-model-name swiss-ai/Apertus-8B-Instruct-2509-{whoami}"
            " --host 0.0.0.0"
            " --port 8080"
        ),
    )

    # Generate job name if not provided
    if not template_args.job_name:
        model_name = extract_model_name(template_args.framework_args)
        template_args.job_name = nanoid(model_name=model_name)

    # Log the full command
    logging.info("=" * 80)
    logging.info("Full command:")
    logging.info(" ".join(sys.argv))
    logging.info("=" * 80)
    logging.info("")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "template.jinja")

    # Calculate nodes_per_worker if not specified
    if template_args.nodes_per_worker is None:
        template_args.nodes_per_worker = template_args.nodes // template_args.workers

    # Set router_environment if not specified
    if template_args.router_environment is None:
        template_args.router_environment = template_args.environment

    # Fetch bootstrap address if not specified and OCF is enabled
    if not template_args.disable_ocf and template_args.ocf_bootstrap_addr is None:
        ocf_bootstrap_addr = fetch_bootstrap_addresses()
        if ocf_bootstrap_addr is None:
            ocf_bootstrap_addr = "/ip4/148.187.108.172/tcp/43905/p2p/QmQsNxJVa2rnidp998qAz4FCutgmjBsuZqtrxUUy5YfgBu"
            logging.warning(
                f"Falling back to hardcoded bootstrap address: {ocf_bootstrap_addr}"
            )
        template_args.ocf_bootstrap_addr = ocf_bootstrap_addr

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as temp_file:
        generate_job_script(template_path, temp_file.name, **template_args.model_dump())
        job_id = submit_job(
            temp_file.name,
            interactive=False,
            nodes=template_args.nodes,
            partition=template_args.partition,
            time=template_args.time,
            account=template_args.account,
            environment=template_args.environment,
        )

        if job_id:
            log_dir = f"logs/{job_id}"

            logging.info("")
            logging.info(f"Root job output will be available in: {log_dir}/log.out")
            logging.info("")
            logging.info(f"To view CSCS Dashboard:")
            logging.info(" https://console.mlp.cscs.ch/")
            logging.info("")
            logging.info("To tail all logs (from all nodes):")
            logging.info(f"  tail -f {log_dir}/*")
            logging.info("")
            logging.info("To cancel job:")
            logging.info(f"  scancel {job_id}")


if __name__ == "__main__":
    main()
