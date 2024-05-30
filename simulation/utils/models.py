import re
import traceback
import warnings
from datetime import datetime

import pathfinder
from pathfinder import Model

from .logger import WandbLogger


class ModelWandbWrapper:
    def __init__(
        self,
        base_lm,
        render,
        wanbd_logger: WandbLogger,
        temperature,
        top_p,
        seed,
        is_api=False,
    ) -> None:
        self.base_lm = base_lm
        self.render = render
        self.wanbd_logger = wanbd_logger

        self.agent_chain = None
        self.chain = None
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.is_api = is_api

    def start_chain(
        self,
        agent_name,
        phase_name,
        query_name,
    ):
        self.agent_chain = self.wanbd_logger.get_agent_chain(agent_name, phase_name)
        self.chain = self.wanbd_logger.start_chain(phase_name + "::" + query_name)
        return self.base_lm

    def end_chain(self, agent_name, lm):
        html = lm.html()  # lm._html()
        html = html.replace("<s>", "")
        html = html.replace("</s>", "")
        html = html.replace("\n", "<br/>")

        def correct_rgba(html):
            """
            Corrects rgba values in the HTML string.
            """
            # This regex finds rgba values that have a decimal in the first three values
            rgba_pattern = re.compile(
                r"rgba\((\d*\.\d+|\d+), (\d*\.\d+|\d+), (\d*\.\d+|\d+),"
                r" (\d*\.\d+|\d+)\)"
            )

            def correct_rgba_match(match):
                # Correct each of the rgba values
                r, g, b, a = match.groups()
                r = int(float(r))
                g = int(float(g))
                b = int(float(b))
                return f"rgba({r}, {g}, {b}, {a})"

            # Replace incorrect rgba values with corrected ones
            return rgba_pattern.sub(correct_rgba_match, html)

        html = correct_rgba(html)
        self.wanbd_logger.end_chain(
            agent_name,
            self.chain,
            html,
        )

    def gen(
        self,
        previous_lm: Model,
        name=None,
        default_value="",
        *,
        max_tokens=1000,
        # regex=None, TODO maybe
        stop_regex=None,
        save_stop_text=False,
        temperature=None,
        top_p=None,
    ):
        start_time_ms = datetime.now().timestamp() * 1000
        prompt = previous_lm._current_prompt()

        if temperature is None:
            temperature = self.temperature
            top_p = self.top_p
        if top_p is None:
            top_p = 1.0

        try:
            lm: Model = previous_lm + pathfinder.gen(
                name=name,
                max_tokens=max_tokens,
                stop_regex=stop_regex,
                temperature=temperature,
                top_p=top_p,
                save_stop_text=save_stop_text,
            )
            res = lm[name]
        except Exception as e:
            warnings.warn(
                f"An exception occured: {e}: {traceback.format_exc()}\nReturning default value in gen",
                RuntimeWarning,
            )
            res = default_value
            lm = previous_lm.set(name, default_value)
        finally:
            if self.render:
                print(lm)
                print("-" * 20)

            # Logging
            end_time_ms = datetime.now().timestamp() * 1000
            self.wanbd_logger.log_trace_llm(
                chain=self.chain,
                name=name,
                default_value=default_value,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                system_message="TODO",
                prompt=prompt,
                status="SUCCESS",
                status_message=f"valid: {True}",
                response_text=res,
                temperature=temperature,
                top_p=top_p,
                token_usage_in=lm.token_in,
                token_usage_out=lm.token_out,
                model_name=lm.model_name,
            )
            self.seed += 1
            return lm

    def find(
        self,
        previous_lm: Model,
        name=None,
        default_value="",
        *,
        max_tokens=100,
        regex=None,
        stop_regex=None,
        temperature=None,
        top_p=None,
    ):
        start_time_ms = datetime.now().timestamp() * 1000
        prompt = previous_lm._current_prompt()

        if temperature is None:
            temperature = self.temperature
            top_p = self.top_p
        if top_p is None:
            top_p = 1.0

        try:
            lm: Model = previous_lm + pathfinder.find(
                name=name,
                max_tokens=max_tokens,
                regex=regex,
                stop_regex=stop_regex,
                temperature=temperature,
                top_p=top_p,
            )
            res = lm[name]
        except Exception as e:
            warnings.warn(
                f"An exception occured: {e}: {traceback.format_exc()}\nReturning default value in find",
                RuntimeWarning,
            )
            res = default_value
            lm = previous_lm.set(name, default_value)
        finally:
            if self.render:
                print(lm)
                print("-" * 20)

            # Logging
            end_time_ms = datetime.now().timestamp() * 1000
            self.wanbd_logger.log_trace_llm(
                chain=self.chain,
                name=name,
                default_value=default_value,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                system_message="TODO",
                prompt=prompt,
                status="SUCCESS",
                status_message=f"valid: {True}",
                response_text=res,
                temperature=temperature,
                top_p=top_p,
                token_usage_in=lm.token_in,
                token_usage_out=lm.token_out,
                model_name=lm.model_name,
            )
            self.seed += 1
            return lm

    def select(
        self,
        previous_lm,
        options,
        default_value=None,
        name=None,
        # No sampling by select, since is used more as parsing previous generated text
    ):
        start_time_ms = datetime.now().timestamp() * 1000
        prompt = previous_lm._current_prompt()

        error_message = None
        try:
            lm: Model = previous_lm + pathfinder.select(
                options=options,
                name=name,
            )
            res = lm[name]
        except Exception as e:
            warnings.warn(
                f"An exception occured: {e}: {traceback.format_exc()}\nReturning default value in select",
                RuntimeWarning,
            )
            res = default_value
            lm = previous_lm.set(name, default_value)
        finally:
            if self.render:
                print(lm)
                print("-" * 20)

            # Logging
            end_time_ms = datetime.now().timestamp() * 1000
            self.wanbd_logger.log_trace_llm(
                chain=self.chain,
                name=name,
                default_value=default_value,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                system_message="TODO",
                prompt=prompt,
                status="SUCCESS" if error_message is None else "ERROR",
                status_message=(
                    f"valid: {True}" if error_message is None else error_message
                ),
                response_text=res,
                temperature=0.0,
                top_p=1.0,
                token_usage_in=lm.token_in,
                token_usage_out=lm.token_out,
                model_name=lm.model_name,
            )
            self.seed += 1
            return lm
