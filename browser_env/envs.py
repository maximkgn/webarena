import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union
import logging
from threading import Event
import asyncio

import numpy as np
import numpy.typing as npt
from beartype import beartype
from gymnasium import Env
from gymnasium.spaces import Box, Text
from playwright.sync_api import (
    CDPSession,
    Page,
    Playwright,
    ViewportSize,
    expect,
    sync_playwright,
)

from .actions import Action, execute_action, get_action_space, create_id_based_action
from .processors import ObservationHandler, ObservationMetadata
from .utils import (
    AccessibilityTree,
    DetachedPage,
    Observation,
    png_bytes_to_numpy,
)

import nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger(__name__)


@dataclass
class PlaywrightScript:
    function: str  # goto, get_by_role
    destination: str  # https://www.google.com/, combobox
    name: str | None = None  # Search, Avatar 2009
    operation: str | None = None  # click, fill, press
    value: str | None = None  # avatar movie, Enter


@beartype
def parse_action(action: str) -> PlaywrightScript:
    splitted = action.strip().split(" ")
    assert len(splitted) >= 2
    match splitted[:2]:
        case ["goto", url]:
            assert len(splitted) == 2
            return PlaywrightScript("goto", url)
        case ["get_by_role", destination]:
            assert len(splitted) >= 4
            match splitted[2:]:
                case [name, operation]:
                    return PlaywrightScript(
                        "get_by_role", destination, name, operation
                    )
                case [name, operation, value]:
                    return PlaywrightScript(
                        "get_by_role", destination, name, operation, value
                    )
                case _:
                    raise ValueError("Invalid action")
        case _:
            raise ValueError(f"Invalid action {action}")


class ScriptBrowserEnv(Env[dict[str, Observation], Action]):
    """
    The goal of this environment is to produce a prototype of a browser environment.
    In the end, we want to support a fully configurable browser environment with wide
    range of action spaces and observation spaces, both structured and unstructured.
    But in this prototype, we just support action space specified by Playwright script,
    and observation space is the html content of the page.
    """

    @beartype
    def __init__(
        self,
        max_page_length: int = 8192,
        headless = True,
        slow_mo: int = 0,
        observation_type: str = "html",
        current_viewport_only: bool = False,
        viewport_size: ViewportSize = {"width": 1280, "height": 720},
        save_trace_enabled: bool = False,
        sleep_after_execution: float = 0.0,
        get_image_obs: bool = False,
    ):
        # TODO: make Space[Action] = ActionSpace
        self.action_space = get_action_space()  # type: ignore[assignment]
        self.headless = headless
        self.slow_mo = slow_mo
        self.current_viewport_only = current_viewport_only
        self.reset_finished = False
        self.viewport_size = viewport_size
        self.save_trace_enabled = save_trace_enabled
        self.sleep_after_execution = sleep_after_execution

        match observation_type:
            case "html" | "accessibility_tree":
                self.text_observation_type = observation_type
                self.image_observation_type = ""
                self.main_observation_type = "text"
            case "image":
                self.image_observation_type = observation_type
                self.text_observation_type = ""  # type: ignore[assignment]
                self.main_observation_type = "image"
            case _:
                raise ValueError(
                    f"Unsupported observation type: {observation_type}"
                )

        self.observation_handler = ObservationHandler(
            self.main_observation_type,
            self.text_observation_type,
            self.image_observation_type,
            self.current_viewport_only,
            self.viewport_size,
            use_image=get_image_obs,
        )

        self.observation_space = (
            self.observation_handler.get_observation_space()
        )
        from omnitool.env.webpage import Webpage
        self.omnitools_webpage = Webpage()

    @staticmethod
    def set_page_init_script(page: Page) -> None:
        # define getEventListeners function
        page.add_init_script("""
        window.showOpenFilePicker=  function (options) {
    return new Promise((resolve) => {
        const input = document.createElement("input");
        input.type = "file";
        input.multiple = options.multiple;
        input.accept = options.types
            .map((type) => type.accept)
            .flatMap((inst) => Object.keys(inst).flatMap((key) => inst[key]))
            .join(",");

        input.addEventListener("change", () => {
            resolve(
                [...input.files].map((file) => {
                    return {
                        getFile: async () =>
                            new Promise((resolve) => {
                                resolve(file);
                            }),
                    };
                })
            );
        });

        input.click();
    });
}
        
        """)

    @beartype
    def setup(self, config_file: Path | None = None) -> None:
        if config_file:
            with open(config_file, "r") as f:
                instance_config = json.load(f)
        else:
            instance_config = {}

        browser_type = instance_config.get("browser_type", "chromium")
        self.context_manager = sync_playwright()
        self.playwright = self.context_manager.__enter__()
        if instance_config['browser_type'] == 'chromium':
            self.browser = self.playwright.chromium.launch(
                #channel="firefox",#"chrome-beta",
                headless=self.headless, slow_mo=self.slow_mo,
                # run with extension
                args=[
                    # "--disable-extensions-except=/Users/mimacadmin/projects/tmp_extension_accessibility/",
                    # "--load-extension=/Users/mimacadmin/projects/tmp_extension_accessibility/",
                    "--window-position=1000,0",  # Adjust the x coordinate as per your screen resolution
                ],
            )
        elif instance_config['browser_type'] == 'chrome':
            self.browser = self.playwright.chromium.launch(
                channel="chrome",#"chrome-beta",
                headless=self.headless, slow_mo=self.slow_mo,
            )

        elif instance_config['browser_type'] == 'firefox':
            self.browser = self.playwright.firefox.launch(
                headless=self.headless, slow_mo=self.slow_mo,
            )            
        else:
            raise ValueError(f"Unsupported browser type: {browser_type}")

        storage_state = instance_config.get("storage_state", None)
        start_url = instance_config.get("start_url", None)
        geolocation = instance_config.get("geolocation", None)

        self.context = self.browser.new_context(
            viewport=self.viewport_size,
            # no_viewport=True,
            storage_state=storage_state,
            geolocation=geolocation,
            device_scale_factor=1,
        )
        if self.save_trace_enabled:
            self.context.tracing.start(screenshots=True, snapshots=True)
        if start_url:
            start_urls = start_url.split(" |AND| ")
            for url in start_urls:
                page = self.context.new_page()
                self.set_page_init_script(page)
                client = page.context.new_cdp_session(
                    page
                )  # talk to chrome devtools
                if self.text_observation_type == "accessibility_tree":
                    client.send("Accessibility.enable")
                page.client = client  # type: ignore # TODO[shuyanzh], fix this hackey client
                self.page = page
                self.reset_finished = True
                # self.step(create_id_based_action(f"goto [{url}]"))
                self.page.goto(url)
                page.wait_for_load_state("networkidle")
            # set the first page as the current page
            self.page = self.context.pages[0]
            self.page.bring_to_front()
        else:
            self.page = self.context.new_page()
            self.set_page_init_script(self.page)
            client = self.page.context.new_cdp_session(self.page)
            if self.text_observation_type == "accessibility_tree":
                client.send("Accessibility.enable")
            self.page.client = client  # type: ignore

    @staticmethod
    def handle_nodes_updated(event: dict[str, Any]) -> None:
        # TODO[shuyanzh] handle the event
        nodes = event['nodes']
        dict_node = nodes[0]
        if len(nodes) > 1:
            print("len(nodes)", len(nodes), "len(dict_node)", len(dict_node), "counter", cntr)
        if 'Amazon' not in dict_node['name']['value']:
            print(f"{dict_node['name']['value']}")
        # print("len(nodes)", len(nodes), "len(dict_node)", len(dict_node), "counter", cntr)
        # print(f"{dict_node['nodeId']}, {dict_node['name']['value']}")
        # print(f"{dict_node}")
        # obs = self.get_obs()
        # print("len(obs)", len(obs))
        pass

    @beartype
    def get_page_client(self, page: Page) -> CDPSession:
        return page.client  # type: ignore

    @beartype
    def _get_obs(self) -> dict[str, Observation]:
        for i in range(5):
            obs = self.observation_handler.get_observation(
                self.page, self.get_page_client(self.page), self.context
            )
            if "busy: 1" in obs['text']:
                logger.warning(f"Accessibility tree is too small, retrying {i}: {obs['text']}")
                time.sleep(1)
                continue

            return obs
        
        raise RuntimeError(f"Could not get obs after {i} retries")

    @beartype
    def _get_obs_metadata(self) -> dict[str, ObservationMetadata]:
        metadata = self.observation_handler.get_observation_metadata()
        return metadata
    
    @beartype
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """
        Reset the environment.
        :param options: options for the environment. The current supported options are:
            - "storage_state": the storage state of the browser. It is a file path to a json file.
        """
        super().reset(seed=seed, options=options)
        if self.reset_finished:
            self.context_manager.__exit__()

        if options is not None and "config_file" in options:
            config_file = Path(options["config_file"])
            if config_file.exists():
                self.setup(config_file)
            else:
                raise ValueError(f"Config file {config_file} does not exist.")
        else:
            self.setup()
        self.reset_finished = True

        if self.sleep_after_execution > 0:
            time.sleep(self.sleep_after_execution)

        observation = self._get_obs()
        observation_metadata = self._get_obs_metadata()
        info = {
            "page": DetachedPage(self.page.url, ""),
            "fail_error": "",
            "observation_metadata": observation_metadata,
            "log_history": ""
        }

        return (observation, info)
    

    def get_obs(self) -> tuple[dict[str, Observation], dict[str, Any]]:
        observation = self._get_obs()
        observation_metadata = self._get_obs_metadata()
        info = {
            "page": DetachedPage(self.page.url, ""),
            "fail_error": "",
            "observation_metadata": observation_metadata,
            "log_history": ""
        }
        return (observation, info)
        

    @beartype
    def save_trace(self, trace_path: str | Path) -> None:
        if self.save_trace_enabled:
            self.context.tracing.stop(path=trace_path)

    @beartype
    def close(self) -> None:
        if self.reset_finished:
            self.context_manager.__exit__()

    @beartype
    def step(
        self, action: Action
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_step(action))

    @beartype
    async def async_step(
        self, action: Action
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any]]:
        self.load_complete_event = asyncio.Event()
        client = self.get_page_client(self.page)
        def on_load_complete(event_data):
            self.load_complete_event.set()
        if not self.reset_finished:
            raise RuntimeError("Call reset first before calling step.")

        success = False

        fail_error = ""
        try:
            client.once("Accessibility.loadComplete", on_load_complete)
            self.omnitools_webpage.set_page(self.page)
            self.page = execute_action(
                action,
                self.page,
                self.context,
                self.observation_handler.action_processor,
                self.omnitools_webpage,
            )
            from webarena.browser_env.processors import ObservationProcessor
            assert isinstance(self.observation_handler.action_processor, ObservationProcessor)
            success = True
        except Exception as e:
            e_lines = str(e).split("\n")
            if 'Timeout ' in e_lines[0] and 'ms exceeded' in e_lines[0]:
                e_lines = [e_lines[0], e_lines[2]]
            fail_error = "\n".join(e_lines)
            logger.error(f"Failed to execute action {action}. Error: {fail_error}")
            if len(fail_error) > 300:
                logger.warning(f"Truncating fail_error: {fail_error} to 300 chars: {fail_error[:300]}")
                fail_error = fail_error[:300]

        # hard sleep TODO[shuyanzh] suboptimal, may need to check network
        if self.sleep_after_execution > 0:
            await asyncio.sleep(self.sleep_after_execution)
        try:
            await asyncio.wait_for(self.load_complete_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for loadComplete event")

        observation = self._get_obs()
        observation_metadata = self._get_obs_metadata()

        info = {
            "page": DetachedPage(self.page.url, self.page.content()),
            "fail_error": fail_error,
            "observation_metadata": observation_metadata,
            "log_history": self.omnitools_webpage.log_history,
            "has_successfully_completed": self.omnitools_webpage.has_successfully_completed,
            "has_failed": self.omnitools_webpage.has_failed,
        }
        msg = (
            observation,
            float(success),  # reward
            False,  # terminated
            False,  # truncated
            info,
        )
        return msg
