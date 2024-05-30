import datetime
import logging
import os

import wandb

# import weasyprint
from wandb.sdk.data_types import trace_tree

# Suppress annoying fontTools messages
logger = logging.getLogger("fontTools.subset")
logger.setLevel(logging.WARNING)


class WandbLogger:
    def __init__(self, scenario_name, configs, debug=False, tags=[]) -> None:
        run = wandb.init(
            project="EMS",
            group=scenario_name,
            config=configs,
            tags=tags,
            save_code=True,
            mode="online" if not debug else "disabled",
        )
        print(f"Storage name: {run.name}-{run.id}")
        self.run_id = run.id
        self.run_name = run.name
        self.current_agent_name = None
        self.current_agent_span = None
        self.current_phase_name = None
        self.token_usage = 0
        self.token_usage_in = 0
        self.token_usage_out = 0
        self.start_time_ms = datetime.datetime.now().timestamp() * 1000

        self.global_step = 0
        self.is_finish_pending = False

        self.html_logs = {}

    def get_agent_chain(self, agent_name, phase_name):
        start_time_ms = datetime.datetime.now().timestamp() * 1000
        if (
            self.current_agent_name != agent_name
            or self.current_phase_name != phase_name
        ):
            if self.current_agent_span is not None:
                TFS = self.token_usage_agent / (
                    (
                        self.current_agent_span._span.end_time_ms
                        - self.current_agent_span._span.start_time_ms
                    )
                    / 1000
                )
                TFS_cumulative = self.token_usage / (
                    (self.current_agent_span._span.end_time_ms - self.start_time_ms)
                    / 1000
                )
                t = trace_tree.WBTraceTree(
                    self.current_agent_span._span, self.current_agent_span._model_dict
                )
                wandb.log(
                    {
                        "experiment/trace": t,
                        "experiment/TFS": TFS,
                        "experiment/TFS_cumulative": TFS_cumulative,
                        "experiment/token_in_cumulative": self.token_usage_in,
                        "experiment/token_out_cumulative": self.token_usage_out,
                    },
                    step=self.global_step,
                    commit=True,
                )
                self.global_step += 1
            self.current_agent_name = agent_name
            self.current_phase_name = phase_name
            self.token_usage_agent = 0
            self.current_agent_span = trace_tree.Trace(
                name=agent_name,
                kind=trace_tree.SpanKind.AGENT,
                start_time_ms=start_time_ms,
                inputs={"phase": phase_name},
            )
        return self.current_agent_span

    def start_chain(self, chain_name):
        assert self.is_finish_pending == False
        self.is_finish_pending = True
        start_time_ms = datetime.datetime.now().timestamp() * 1000
        chain = trace_tree.Trace(
            name=chain_name, kind=trace_tree.SpanKind.CHAIN, start_time_ms=start_time_ms
        )
        self.chain_error = False
        self.chain_error_message = ""
        return chain

    def log_trace_llm(
        self,
        chain,
        name,
        default_value,
        start_time_ms,
        end_time_ms,
        system_message,
        prompt,
        status,
        status_message,
        response_text,
        temperature,
        top_p,
        token_usage_in,
        token_usage_out,
        model_name,
    ):
        if status == "ERROR":
            self.chain_error = True
            self.chain_error_message = f"Error in {name}."

        t = trace_tree.Trace(
            name=name,
            kind=trace_tree.SpanKind.LLM,
            status_code=status,
            status_message=status_message,
            metadata={
                "temperature": temperature,
                "top_p": top_p,
                "token_in": token_usage_in,
                "token_out": token_usage_out,
                "model_name": model_name,
            },
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            inputs={
                "system_prompt": system_message,
                "prompt": prompt,
                "default_value": default_value,
            },
            outputs={"response": response_text},
        )
        token_usage = token_usage_in + token_usage_out
        self.token_usage_in += token_usage_in
        self.token_usage_out += token_usage_out
        self.token_usage += token_usage
        self.token_usage_agent += token_usage
        chain.add_child(t)

    def end_chain(self, agent_name, chain_span, html_render):
        assert self.is_finish_pending == True
        self.is_finish_pending = False
        if agent_name != self.current_agent_name:
            raise Exception("Agent name does not match")
        chain_agent = self.current_agent_span
        end_time_ms = datetime.datetime.now().timestamp() * 1000
        chain_span._span.end_time_ms = end_time_ms
        chain_agent._span.end_time_ms = end_time_ms
        chain_span._span.add_named_result(
            inputs={}, outputs={"html_render": html_render}
        )
        if agent_name not in self.html_logs:
            self.html_logs[agent_name] = []
        self.html_logs[agent_name].append(f"<h3>{chain_span.name}</h3>\n{html_render}")
        if self.chain_error:
            chain_span._span.status_code = "ERROR"
            chain_span._span.status_message = self.chain_error_message
            chain_agent._span.status_code = "ERROR"
            chain_agent._span.status_message = self.chain_error_message

        chain_agent.add_child(chain_span)

    def save(self, base_path, agent_name_to_id: dict[str, str]):
        for k, v in self.html_logs.items():
            html = f"""
                    <html>
                    <head>
                    <title>{k}</title>
                    </head>
                    <body>
                    <h1>{k}</h1>
            """
            for i in v:
                html += i
            html += "</body></html>"
            path = os.path.join(base_path, agent_name_to_id[k], f"prompts.pdf")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # weasyprint.HTML(string=html).write_pdf(path)

    def log_game(self, kwargs, last_log=False):
        if last_log and self.current_agent_span is not None:
            TFS = self.token_usage_agent / (
                (
                    self.current_agent_span._span.end_time_ms
                    - self.current_agent_span._span.start_time_ms
                )
                / 1000
            )
            TFS_cumulative = self.token_usage / (
                (self.current_agent_span._span.end_time_ms - self.start_time_ms) / 1000
            )
            t = trace_tree.WBTraceTree(
                self.current_agent_span._span, self.current_agent_span._model_dict
            )
            wandb.log(
                {
                    "experiment/trace": t,
                    "experiment/TFS": TFS,
                    "experiment/TFS_cumulative": TFS_cumulative,
                    "experiment/token_in_cumulative": self.token_usage_in,
                    "experiment/token_out_cumulative": self.token_usage_out,
                },
                step=self.global_step,
                commit=False,
            )
        wandb.log(kwargs, step=self.global_step, commit=last_log)
