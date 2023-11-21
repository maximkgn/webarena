import json
import re
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, TypedDict, Union
import logging

import numpy as np
import numpy.typing as npt
from beartype import beartype
from gymnasium import spaces
from playwright.sync_api import CDPSession, Page, ViewportSize

logger = logging.getLogger(__name__)
marked_elements = {}
node_id_to_unique_id = defaultdict(dict)

from browser_env.constants import (
    ASCII_CHARSET,
    FREQ_UNICODE_CHARSET,
    IGNORED_ACTREE_PROPERTIES,
    UTTERANCE_MAX_LENGTH,
)

from .utils import (
    AccessibilityTree,
    BrowserConfig,
    BrowserInfo,
    Observation,
    png_bytes_to_numpy,
)


class ObservationProcessor:
    def process(self, page: Page, client: CDPSession) -> Observation:
        raise NotImplementedError


class ObservationMetadata(TypedDict):
    obs_nodes_info: dict[str, Any]


def create_empty_metadata() -> ObservationMetadata:
    return {
        "obs_nodes_info": {},
    }


class TextObervationProcessor(ObservationProcessor):
    def __init__(
        self,
        observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
    ):
        self.observation_type = observation_type
        self.current_viewport_only = current_viewport_only
        self.viewport_size = viewport_size
        self.observation_tag = "text"
        self.meta_data = (
            create_empty_metadata()
        )  # use the store meta data of this observation type

    def fetch_browser_info(
        self,
        page: Page,
        client: CDPSession,
    ) -> BrowserInfo:
        # extract domtree
        tree = client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )

        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds
        # add union bound placeholder
        tree["documents"][0]["layout"]["unionBounds"] = [None for _ in bounds]

        # extract browser info
        win_upper_bound = page.evaluate("window.pageYOffset")
        win_left_bound = page.evaluate("window.pageXOffset")
        win_width = page.evaluate("window.screen.width")
        win_height = page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_upper_bound + win_height
        device_pixel_ratio = page.evaluate("window.devicePixelRatio")
        # assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

        config: BrowserConfig = {
            "win_upper_bound": win_upper_bound,
            "win_left_bound": win_left_bound,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": win_right_bound,
            "win_lower_bound": win_lower_bound,
            "device_pixel_ratio": device_pixel_ratio,
        }

        # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
        info: BrowserInfo = {"DOMTree": tree, "config": config}

        return info

    @beartype
    @staticmethod
    def partially_in_viewport(
        bound: list[float], config: BrowserConfig
    ) -> bool:
        [x, y, width, height] = bound
        elem_left_bound = x
        elem_top_bound = y
        elem_right_bound = x + width
        elem_lower_bound = y + height

        ok = (
            elem_left_bound < config["win_right_bound"]
            and elem_right_bound >= config["win_left_bound"]
            and elem_top_bound < config["win_lower_bound"]
            and elem_lower_bound >= config["win_upper_bound"]
        )

        return ok

    @beartype
    def retrieve_viewport_info(self, info: BrowserInfo) -> None:
        """Add viewport related information to the DOMTree
        1. add union bound, which is a union of all the bounds of the nodes in the subtree
        This is only used when current_viewport_only is enabled since it is quite slow

        TODO[robert1003]: improve
        """
        tree = info["DOMTree"]
        document = tree["documents"][0]
        nodes = document["nodes"]
        parent = nodes["parentIndex"]
        node_names = nodes["nodeName"]

        layout = document["layout"]
        layout_node_cursor = layout["nodeIndex"]
        bounds = layout["bounds"]

        graph = defaultdict(lambda: [])
        assert len(node_names) == len(parent)
        for node_idx in range(len(node_names)):
            parent_idx = parent[node_idx]
            if parent_idx != -1:
                graph[parent_idx].append(node_idx)

        union_bounds: list[list[float] | None] = [None for _ in bounds]

        def valid_bbox(bound: list[float] | None) -> bool:
            if bound is None:
                return False
            # no width or height
            if np.isclose(bound[2], 0):
                return False
            if np.isclose(bound[3], 0):
                return False
            return True

        def add_union_bound(idx: int) -> list[float] | None:
            if idx in layout_node_cursor:
                cursor = layout_node_cursor.index(idx)
                node_bound = bounds[cursor].copy()
                tree_bounds: list[Any] = [node_bound]
                for child_idx in graph[idx]:
                    child_bound = add_union_bound(child_idx)
                    tree_bounds.append(
                        child_bound.copy() if child_bound else None
                    )

                tree_bounds = [b for b in tree_bounds if valid_bbox(b)]
                # convert to absolute coordinates
                for i in range(len(tree_bounds)):
                    tree_bounds[i][2] = tree_bounds[i][0] + tree_bounds[i][2]
                    tree_bounds[i][3] = tree_bounds[i][1] + tree_bounds[i][3]

                if len(tree_bounds) == 0:
                    assert not valid_bbox(node_bound)
                    node_union_bound = [0.0, 0.0, 0.0, 0.0]
                else:
                    left_bound = min([b[0] for b in tree_bounds])
                    top_bound = min([b[1] for b in tree_bounds])
                    right_bound = max([b[2] for b in tree_bounds])
                    bottom_bound = max([b[3] for b in tree_bounds])
                    node_union_bound = [
                        left_bound,
                        top_bound,
                        right_bound - left_bound,
                        bottom_bound - top_bound,
                    ]

                # update the list
                union_bounds[cursor] = node_union_bound
            else:
                node_union_bound = None

            return node_union_bound

        add_union_bound(0)
        info["DOMTree"]["documents"][0]["layout"]["unionBounds"] = union_bounds

    @beartype
    def current_viewport_html(self, info: BrowserInfo) -> str:
        # adopted from [natbot](https://github.com/nat/natbot)
        tree = info["DOMTree"]
        strings = tree["strings"]
        document = tree["documents"][0]
        nodes = document["nodes"]
        attributes = nodes["attributes"]
        node_value = nodes["nodeValue"]
        parent = nodes["parentIndex"]
        node_names = nodes["nodeName"]

        layout = document["layout"]
        layout_node_cursor = layout["nodeIndex"]
        union_bounds = layout["unionBounds"]

        graph = defaultdict(lambda: [])
        for node_idx in range(len(node_names)):
            parent_idx = parent[node_idx]
            if parent_idx != -1:
                graph[parent_idx].append(node_idx)

        def dfs(idx: int) -> str:
            node_name = strings[node_names[idx]].lower().strip()
            can_skip = "#" in node_name or "::" in node_name

            inner_text = ""
            node_value_idx = node_value[idx]
            if node_value_idx >= 0 and node_value_idx < len(strings):
                inner_text = " ".join(strings[node_value_idx].split())
            node_attributes = [strings[i] for i in attributes[idx]]
            node_attributes_str = ""
            for i in range(0, len(node_attributes), 2):
                a = node_attributes[i]
                b = node_attributes[i + 1]
                b = " ".join(b.split())
                node_attributes_str += f'{a}="{b}" '
            node_attributes_str = node_attributes_str.strip()

            html = ""
            if not can_skip:
                html += f"<{node_name}"
                if {node_attributes_str}:
                    html += f" {node_attributes_str}"
                html += f">{inner_text}"
            else:
                html += f"{inner_text}"

            for child_idx in graph[idx]:
                if child_idx in layout_node_cursor:
                    cursor = layout_node_cursor.index(child_idx)
                    union_bound = union_bounds[cursor]
                    if not self.partially_in_viewport(
                        union_bound, info["config"]
                    ):
                        continue
                    html += dfs(child_idx)

            if not can_skip:
                html += f"</{node_name}>"

            return html

        html = dfs(0)
        return html

    @beartype
    def fetch_page_accessibility_tree(
        self, info: BrowserInfo, client: CDPSession, fix_parent_dom_node_id=True
    ) -> AccessibilityTree:
        accessibility_tree: AccessibilityTree = client.send(
            "Accessibility.getFullAXTree", {}
        )["nodes"]

        # query first node
        # client.send( # TODOREMOVE why this gets stuck in frames that are not the main frame????
        #     "Accessibility.queryAXTree",
        #     {
        #         'backendNodeId': int(accessibility_tree[0]['backendDOMNodeId'])
        #     }
        # )

        # a few nodes are repeated in the accessibility tree
        seen_ids = set()
        _accessibility_tree = []
        for node in accessibility_tree:
            if node["nodeId"] not in seen_ids:
                _accessibility_tree.append(node)
                seen_ids.add(node["nodeId"])
        accessibility_tree = _accessibility_tree

        # add the bounding box of each node
        tree = info["DOMTree"]
        document = tree["documents"][0]
        nodes = document["nodes"]
        backend_node_id = nodes["backendNodeId"]
        node_names = nodes["nodeName"]

        layout = document["layout"]
        layout_node_cursor = layout["nodeIndex"]
        bounds = layout["bounds"]
        union_bounds = layout["unionBounds"]
        offsetrect_bounds = layout["offsetRects"]
        backend_id_to_bound = {}

        # get the mapping between backend node id and bounding box
        for idx in range(len(node_names)):
            if idx not in layout_node_cursor:
                continue
            cursor = layout_node_cursor.index(idx)
            node_bound = bounds[cursor]
            node_union_bound = union_bounds[cursor]
            node_offsetrect_bound = offsetrect_bounds[cursor]
            node_backend_id = backend_node_id[idx]
            backend_id_to_bound[node_backend_id] = [
                node_bound,
                node_union_bound,
                node_offsetrect_bound,
            ]
        node_id_to_node = {}
        for node in accessibility_tree:
            node_id_to_node[node["nodeId"]] = node

        parent_graph: dict[str, str] = {} # son to parent
        refine_node_ids: list[str] = []
        for node in accessibility_tree:
            if "parentId" in node:
                parent_graph[node["nodeId"]] = node["parentId"]
            backend_dom_node_id = None
            if "backendDOMNodeId" in node:
                backend_dom_node_id = node["backendDOMNodeId"]
            elif fix_parent_dom_node_id:
                cnt = 0
                cur_node = node
                while 'parentId' in cur_node:
                    cnt += 1
                    cur_node = node_id_to_node[cur_node['parentId']]
                    if 'backendDOMNodeId' in cur_node:
                        backend_dom_node_id = cur_node['backendDOMNodeId']
                        logger.warning(f"Found backendDOMNodeId for node {node['nodeId']} after {cnt} iterations")
                        break
                    
            if backend_dom_node_id is None:
                node["bound"] = None
                node["union_bound"] = None
                node["offsetrect_bound"] = None
            elif backend_dom_node_id not in backend_id_to_bound:
                refine_node_ids.append(node["nodeId"])
            else:
                node["bound"] = backend_id_to_bound[backend_dom_node_id][
                    0
                ]
                node["union_bound"] = backend_id_to_bound[
                    backend_dom_node_id
                ][1]
                node["offsetrect_bound"] = backend_id_to_bound[
                    backend_dom_node_id
                ][2]

        # refine the bounding box for nodes which only appear in the accessibility tree
        node_ids = [node["nodeId"] for node in accessibility_tree]
        for refine_node_id in refine_node_ids:
            child_id = refine_node_id
            parent_idx: None | int = None
            while child_id in parent_graph:
                parent_id = parent_graph[child_id]
                parent_idx = node_ids.index(parent_id)
                child_id = parent_id
                if accessibility_tree[parent_idx]["union_bound"] is not None:
                    break

            refine_node_idx = node_ids.index(refine_node_id)

            if parent_idx is not None:
                accessibility_tree[refine_node_idx][
                    "bound"
                ] = accessibility_tree[parent_idx]["bound"]
                accessibility_tree[refine_node_idx][
                    "union_bound"
                ] = accessibility_tree[parent_idx]["union_bound"]
                accessibility_tree[refine_node_idx][
                    "offsetrect_bound"
                ] = accessibility_tree[parent_idx]["offsetrect_bound"]
            else:
                accessibility_tree[refine_node_idx]["bound"] = None
                accessibility_tree[refine_node_idx]["union_bound"] = None
                accessibility_tree[refine_node_idx]["offsetrect_bound"] = None

        return accessibility_tree

    @beartype
    def current_viewport_accessibility_tree(
        self,
        info: BrowserInfo,
        accessibility_tree: AccessibilityTree,
        page: Page,
        use_active_elem_as_bbox: bool,
    ) -> AccessibilityTree:
        config = info["config"]

        subtree = []
        for node in accessibility_tree:
            if not node["bound"]:
                continue

            [x, y, width, height] = node["bound"]
            elem_left_bound = x
            elem_top_bound = y
            elem_right_bound = x + width
            elem_lower_bound = y + height

            ok = (
                elem_left_bound <= config["win_right_bound"]
                and elem_right_bound >= config["win_left_bound"]
                and elem_top_bound <= config["win_lower_bound"]
                and elem_lower_bound >= config["win_upper_bound"]
            )

            if ok:
                subtree.append(node)

        if not use_active_elem_as_bbox:
            return subtree
        
        bbox = page.evaluate('''() => {
            const activeElement = document.activeElement;
            const rect = activeElement.getBoundingClientRect();
            return {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height
            };
        }''')
        bbox['win_right_bound'] = bbox['x'] + bbox['width']
        bbox['win_left_bound'] = bbox['x']
        bbox['win_lower_bound'] = bbox['y'] + bbox['height']
        bbox['win_upper_bound'] = bbox['y']

        # make bbox in viewport
        bbox['win_right_bound'] = min(bbox['win_right_bound'], config['win_right_bound'])
        bbox['win_left_bound'] = max(bbox['win_left_bound'], config['win_left_bound'])
        bbox['win_lower_bound'] = min(bbox['win_lower_bound'], config['win_lower_bound'])
        bbox['win_upper_bound'] = max(bbox['win_upper_bound'], config['win_upper_bound'])
        
        subtree2 = []
        for node in accessibility_tree:
            if not node["bound"]:
                continue

            [x, y, width, height] = node["bound"]
            elem_left_bound = x
            elem_top_bound = y
            elem_right_bound = x + width
            elem_lower_bound = y + height

            ok = (
                elem_left_bound <= bbox["win_right_bound"]
                and elem_right_bound >= bbox["win_left_bound"]
                and elem_top_bound <= bbox["win_lower_bound"]
                and elem_lower_bound >= bbox["win_upper_bound"]
            )

            if ok:
                subtree2.append(node)
        
        
        logger.warning(f"len(subtree): {len(subtree)}, len(subtree2): {len(subtree2)}")

        return subtree2

    @beartype
    @staticmethod
    def parse_accessibility_tree(
        accessibility_tree: AccessibilityTree, multi_roots: bool, frame, client
    ) -> tuple[str, dict[str, Any]]:
        """Parse the accessibility tree into a string text"""
        node_id_to_idx = {}
        for idx, node in enumerate(accessibility_tree):
            node_id_to_idx[node["nodeId"]] = idx

        obs_nodes_info = {}
        visited_nodes = set()
        count_nodes_seen = [0]
        count_num_invalid_nodes = [0]

        def dfs(idx: int, obs_node_id: str, depth: int, log_cycles=True) -> str:
            tree_str = ""
            if obs_node_id in visited_nodes and log_cycles:
                logger.error(f"Cycle detected in the accessibility tree")
            node = accessibility_tree[idx]
            indent = "\t" * depth
            valid_node = True
            if obs_node_id not in visited_nodes:
                count_nodes_seen[0] += 1
            visited_nodes.add(obs_node_id)
            try:
                role = node["role"]["value"]
                name = node["name"]["value"]
                global node_id_to_unique_id
                unique_id = node_id_to_unique_id[frame].get(obs_node_id, "")
                if not name.strip() and unique_id != "":
                    name = "clickable"
                node_str = f"[{unique_id}] {role} {repr(name)}"
                properties = []
                for property in node.get("properties", []):
                    try:
                        if property["name"] in IGNORED_ACTREE_PROPERTIES:
                            continue
                        properties.append(
                            f'{property["name"]}: {property["value"]["value"]}'
                        )
                    except KeyError:
                        pass

                if properties:
                    node_str += " " + " ".join(properties)

                # check valid
                if not node_str.strip():
                    valid_node = False

                # empty generic node
                if not name.strip():
                    if not properties:
                        if role in [
                            "generic",
                            # "img",
                            "list",
                            "strong",
                            "paragraph",
                            "banner",
                            "navigation",
                            "Section",
                            "LabelText",
                            "Legend",
                            "listitem",
                        ]:
                            valid_node = False

                    elif role in ["listitem"]:
                        valid_node = False

                if role == 'IframePresentational':  
                    html = client.send("DOM.getOuterHTML", {"backendNodeId": node['backendDOMNodeId']})['outerHTML']
                    iframe_name = re.search(r'name="([^"]+)"', html).group(1)
                    # Add iframe name to the node representing string
                    node_str += f" {repr(iframe_name)}"

                if valid_node:
                    tree_str += f"{indent}{node_str}"
                    obs_nodes_info[obs_node_id] = {
                        "backend_id": node["backendDOMNodeId"],
                        "bound": node["bound"],
                        "union_bound": node["union_bound"],
                        "offsetrect_bound": node["offsetrect_bound"],
                        "text": node_str,
                        "role": role,
                        "name": name,
                    }

            except Exception as e:
                valid_node = False

            count_num_invalid_nodes[0] += 1 - int(valid_node)
            if role == 'MenuListPopup' and len(node['childIds']) > 10:
                # Declutter the accessibility tree from many options
                tree_str += f"{indent}... ({len(node['childIds'])} options)"
                return tree_str

            for _, child_node_id in enumerate(node["childIds"]):
                if child_node_id not in node_id_to_idx:
                    continue
                # mark this to save some tokens
                child_depth = depth + 1 if valid_node else depth
                child_str = dfs(
                    node_id_to_idx[child_node_id], child_node_id, child_depth
                )
                if child_str.strip():
                    if tree_str.strip():
                        tree_str += "\n"
                    tree_str += child_str

            return tree_str

        tree_str = dfs(0, accessibility_tree[0]["nodeId"], 0)
        unvisited_nodes = set(node_id_to_idx.keys()) - visited_nodes
        if multi_roots:
            while unvisited_nodes:
                node_id = unvisited_nodes.pop()
                additional_tree = dfs(node_id_to_idx[node_id], node_id, 0, log_cycles=False)
                if additional_tree.strip():
                    tree_str += "\n" + additional_tree
                    unvisited_nodes = set(node_id_to_idx.keys()) - visited_nodes
        count_num_invalid_nodes = count_num_invalid_nodes[0]
        count_nodes_seen = count_nodes_seen[0]
        if count_nodes_seen != len(accessibility_tree):
            logger.error(f"count_nodes_seen: {count_nodes_seen}, len(accessibility_tree): {len(accessibility_tree)}")
        if count_num_invalid_nodes + len(obs_nodes_info) != count_nodes_seen:
            logger.warning(f"count_num_invalid_nodes: {count_num_invalid_nodes}, len(obs_nodes_info): {len(obs_nodes_info)}, count_nodes_seen: {count_nodes_seen}")
        tree_num_lines = len(tree_str.split("\n"))
        if len(obs_nodes_info) != tree_num_lines:
            logger.warning(f"len(obs_nodes_info): {len(obs_nodes_info)}, tree_num_lines: {tree_num_lines}")
        return tree_str, obs_nodes_info

    @beartype
    @staticmethod
    def clean_accesibility_tree(tree_str: str, clean_ids: bool = False) -> str:
        """further clean accesibility tree"""
        clean_lines: list[str] = []
        for line in tree_str.split("\n"):
            if "statictext" in line.lower():
                prev_lines = clean_lines[-3:]
                static_text = line.split("StaticText")[1].strip()
                quote_char = static_text[0]
                assert quote_char in ["'", '"']
                assert static_text.count(quote_char) == 2
                static_text = static_text.split(quote_char)[1]
                unique_text = all(
                    static_text not in prev_line
                    for prev_line in prev_lines
                )
                if unique_text:
                    clean_lines.append(line)
            else:
                clean_lines.append(line)
        
        if clean_ids: # remove [id] --> []
            for idx, line in enumerate(clean_lines):
                # replace only the first occurence. remove the id but keep the brackets
                clean_lines[idx] = re.sub(r"\[[^\]]+\]", "[]", line, count=1)

        return "\n".join(clean_lines)

    @beartype
    def process(self, page: Page, client: CDPSession, context) -> str:
        # get the tab info
        open_tabs = page.context.pages
        try:
            tab_titles = [tab.title() for tab in open_tabs]
            current_tab_idx = open_tabs.index(page)
            for idx in range(len(open_tabs)):
                if idx == current_tab_idx:
                    tab_titles[
                        idx
                    ] = f"Tab {idx} (current): {open_tabs[idx].title()}"
                else:
                    tab_titles[idx] = f"Tab {idx}: {open_tabs[idx].title()}"
            tab_title_str = " | ".join(tab_titles)
        except Exception:
            tab_title_str = " | ".join(
                ["Tab {idx}" for idx in range(len(open_tabs))]
            )

        try:
            browser_info = self.fetch_browser_info(page, client)
        except Exception:
            page.wait_for_load_state("load", timeout=500)
            browser_info = self.fetch_browser_info(page, client)

        if self.current_viewport_only:
            self.retrieve_viewport_info(browser_info)

        if self.observation_type == "html":
            if self.current_viewport_only:
                html = self.current_viewport_html(browser_info)
                content = html
            else:
                content = page.content()
        elif self.observation_type == "accessibility_tree":
            content = ""
            obs_nodes_info = {}

            for frame in page.frames:
                try: 
                    client_frame = context.new_cdp_session(frame)
                    browser_info = self.fetch_browser_info(frame, client_frame)
                except:
                    continue

                accessibility_tree_frame = self.fetch_page_accessibility_tree(
                    browser_info, client_frame
                )
                if not accessibility_tree_frame:
                    continue

                nodes_marked_in_frame = node_id_to_unique_id[frame].keys()
                nodes_in_frame = [node['nodeId'] for node in accessibility_tree_frame]
                num_nodes_marked = len(set(nodes_marked_in_frame) & set(nodes_in_frame)) 
                if num_nodes_marked == 0:
                    logger.debug(f"No marked nodes in frame {frame.name or frame.url}, skipping")
                    continue
                
                use_active_elem_as_bbox = False # expiment in limiting the accessibility tree to the active element
                multi_roots = use_active_elem_as_bbox
                if self.current_viewport_only:
                    accessibility_tree_frame = self.current_viewport_accessibility_tree(
                        browser_info, accessibility_tree_frame, page, use_active_elem_as_bbox=use_active_elem_as_bbox,
                    )
                    if not accessibility_tree_frame:
                        continue
                content_frame, obs_nodes_info_frame = self.parse_accessibility_tree(
                    accessibility_tree_frame, multi_roots, frame, client_frame
                )
                content_frame = self.clean_accesibility_tree(content_frame)
                content_frame += "\n"
                if frame.name:
                    content_frame = content_frame.split("\n")  # Add frame name to root node of the tree 
                    content_frame = content_frame[0] + str(frame.name) + "\n" + "\n".join(content_frame[1:]) + "\n"

                content += content_frame
                obs_nodes_info = {**obs_nodes_info, **obs_nodes_info_frame}
            self.obs_nodes_info = obs_nodes_info
            self.meta_data["obs_nodes_info"] = obs_nodes_info
        else:
            raise ValueError(
                f"Invalid observatrion type: {self.observation_type}"
            )

        self.browser_config = browser_info["config"]
        content = f"{tab_title_str}\n\n{content}"
        return content

    @beartype
    def get_element_center(self, element_id: str) -> tuple[float, float]:
        node_info = self.obs_nodes_info[element_id]
        node_bound = node_info["bound"]
        x, y, width, height = node_bound
        browser_config = self.browser_config
        b_x, b_y = (
            browser_config["win_left_bound"],
            browser_config["win_upper_bound"],
        )
        center_x = (x - b_x) + width / 2
        center_y = (y - b_y) + height / 2
        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )


class ImageObservationProcessor(ObservationProcessor):
    def __init__(self, observation_type: str):
        self.observation_type = observation_type
        self.observation_tag = "image"
        self.meta_data = create_empty_metadata()

    def process(self, page: Page, client: CDPSession) -> npt.NDArray[np.uint8]:
        try:
            screenshot = png_bytes_to_numpy(page.screenshot())
        except:
            page.wait_for_event("load")
            screenshot = png_bytes_to_numpy(page.screenshot())
        return screenshot
    

class ImageObservationProcessorWithSetOfMarks(ImageObservationProcessor):
    
    def place_som(self, page: Page, client: CDPSession, context):
        # Read the JavaScript script from the file
        with open("omnitool/mark_borders.js", 'r') as file:
            js_script = file.read()

        def run_script_in_frame(frame, counter, iframe_name=None):
            # Modify the script to start with the specific counter
            modified_script = js_script.replace('let counter = 0;', f'let counter = {counter};')
            elements = frame.evaluate(modified_script)

            # Add iframe origin information to each element
            for element in elements:
                element['iframe'] = frame
                element['iframe_name'] = iframe_name

            return elements

        marked_elements = []
        counter = 0
        
        # Iterate overeach iframe and run the script
        unique_ids_in_tree = {}
        assert page.main_frame in page.frames
        for frame in page.frames:
            iframe_name = frame.name or frame.url  # Use the frame's name or URL as an identifier
            marked_elements_iframe = run_script_in_frame(frame, counter, iframe_name=iframe_name)
            marked_elements.extend(marked_elements_iframe)
            counter += len(marked_elements_iframe)

            try:
                client_frame = context.new_cdp_session(frame)
            except:
                if len(marked_elements_iframe) > 0:
                    logger.warning(f"Found marked elements in frame {iframe_name} but could not get CDP session")
                continue
            
            # Get accessibility items for each marked element
            accessibility_tree: AccessibilityTree = client_frame.send(
                "Accessibility.getFullAXTree", {}
            )["nodes"]
            accessibility_tree_w_uniqueid = [item for item in accessibility_tree if 'uniqueid__' in str(item)]
            uniqueid_pattern = re.compile(r'uniqueid__(\d+)__')
            for ax_item in accessibility_tree_w_uniqueid:
                name = ax_item['name']['value']
                match = uniqueid_pattern.search(name)
                if match is None:
                    continue
                unique_id = int(match.group(1))
                unique_ids_in_tree[unique_id] = (ax_item, frame)

            xpaths_array = ','.join([f'"{elem["xpath"]}"' for elem in marked_elements_iframe])
            labels_array = ','.join([f'"{elem["old_aria_label"]}"' for elem in marked_elements_iframe])
            frame.evaluate(f'''
            (function() {{
                const xpaths = [{xpaths_array}];
                const oldLabels = [{labels_array}];
                xpaths.forEach((xpath, index) => {{
                    const element = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                    if (element) {{
                        const oldLabel = oldLabels[index];
                        if (oldLabel != 'None') {{
                            element.setAttribute('aria-label', oldLabel);
                        }} else {{
                            element.removeAttribute('aria-label');
                        }}
                    }}
                }});
            }})();
            ''')





        # Convert to a dictionary with IDs as keys
        marked_elements_dict = {element['id']: element for element in marked_elements}

        # Add accessibility tree items to the marked elements dictionary
        for unique_id, (ax_item, frame) in unique_ids_in_tree.items():
            marked_elements_dict[unique_id]['ax_item'] = ax_item
        
        # create nodeID to uniqueID mapping
        global node_id_to_unique_id
        node_id_to_unique_id = defaultdict(dict)
        for unique_id, (ax_item, frame) in unique_ids_in_tree.items():
            node_id_to_unique_id[frame][ax_item['nodeId']] = unique_id


        #     role = node["role"]["value"]
        #     name = node["name"]["value"]
        #     node_str = f"{role} {repr(name)}"
        #     properties = []
        #     for property in node.get("properties", []):
        #         try:
        #             if property["name"] in IGNORED_ACTREE_PROPERTIES:
        #                 continue
        #             properties.append(
        #                 f'{property["name"]}: {property["value"]["value"]}'
        #             )
        #         except KeyError:
        #                 pass
        #         if properties:
        #             node_str += " " + " ".join(properties)
        #     marked_elements_dict[unique_id]['node_str'] = node_str



        # always show scroller for the page
        css_to_inject = """
            /* This CSS ensures that scrollbars are always shown */
            html {
                overflow-y: scroll;
            }
            body {
                overflow-y: scroll;
            }
            ::-webkit-scrollbar {
                -webkit-appearance: none;
                width: 7px;
            }
            ::-webkit-scrollbar-thumb {
                border-radius: 4px;
                background-color: rgba(0,0,0,.5);
                box-shadow: 0 0 1px rgba(255,255,255,.5);
            }
        """       # Inject CSS into the page
        page.add_style_tag(content=css_to_inject)

        
        return marked_elements_dict

    def remove_som(self, page: Page):
        """Removes Set Of Marks from screen"""
        with open("omnitool/remove_mark_borders.js", 'r') as file:
            js_script = file.read()
        
        def run_removal_script_in_frame(frame):
            return frame.evaluate(js_script)
        run_removal_script_in_frame(page.main_frame)
        # Iterate over each iframe and run the removal script
        for frame in page.frames:
            if frame != page.main_frame:  # Skip the main frame as it's already processed
                run_removal_script_in_frame(frame)

    def process(self, page: Page, client: CDPSession, context) -> npt.NDArray[np.uint8]:
        global marked_elements
        try:
            marked_elements = self.place_som(page, client, context)
        except:
            import time
            time.sleep(3) # TODOREMOVE?
            marked_elements = self.place_som(page, client, context)
        # import asyncio

        screenshot = super().process(page, client)
        try:
            self.remove_som(page)
        except:
            import time
            time.sleep(3) # TODOREMOVE?
            self.remove_som(page)
        return screenshot


class ObservationHandler:
    """Main entry point to access all observation processor"""

    def __init__(
        self,
        main_observation_type: str,
        text_observation_type: str,
        image_observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
        use_image: bool = True,
    ) -> None:
        self.main_observation_type = main_observation_type
        self.text_processor = TextObervationProcessor(
            text_observation_type, current_viewport_only, viewport_size
        )
        if use_image:
            self.image_processor = ImageObservationProcessorWithSetOfMarks(
                image_observation_type
            )
        self.use_image = use_image
        self.viewport_size = viewport_size

    @beartype
    def get_observation_space(self) -> spaces.Dict:
        text_space = spaces.Text(
            min_length=0,
            max_length=UTTERANCE_MAX_LENGTH,
            charset=ASCII_CHARSET + FREQ_UNICODE_CHARSET,
        )

        image_space = spaces.Box(
            # Each position stores the RGB values. Note the swapped axes (height first).
            np.zeros(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            ),
            np.ones(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            )
            * 255.0,
            dtype=np.uint8,
        )

        return spaces.Dict({"text": text_space, "image": image_space})

    @beartype
    def get_observation(
        self, page: Page, client: CDPSession, context
    ) -> dict[str, Observation]:
        res = {}
        if self.use_image:
            res["image"] = self.image_processor.process(page, client, context)
        res["text"] = self.text_processor.process(page, client, context)
        return res

    @beartype
    def get_observation_metadata(self) -> dict[str, ObservationMetadata]:
        res = {"text": self.text_processor.meta_data}
        if self.use_image:
            res["image"] = self.image_processor.meta_data
        return res

    @property
    def action_processor(self) -> ObservationProcessor:
        """Return the main processor that is associated with the action space"""
        if self.main_observation_type == "text":
            return self.text_processor
        elif self.main_observation_type == "image":
            return self.image_processor
        else:
            raise ValueError("Invalid main observation type")
