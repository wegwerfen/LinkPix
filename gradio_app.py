#!/usr/bin/env python3
"""
Gradio-based management console for the Image Generation Server.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import shutil
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set

import gradio as gr
import requests

CONFIG_PATH = "config.json"
STYLES_PATH = os.path.join("data", "styles.json")
PREFERENCES_PATH = os.path.join("data", "user_preferences.json")
PLACEHOLDERS_PATH = os.path.join("data", "placeholders.json")
CACHE_ROOT_DEFAULT = "cache"
WORKFLOW_DIR = "workflows"
WORKFLOW_SETTINGS_DIR = os.path.join(WORKFLOW_DIR, "settings")
WORKFLOW_ORIGINALS_DIR = os.path.join(WORKFLOW_DIR, "originals")
MAX_WORKFLOW_FIELDS = 80
DATA_DIR = "data"
FLASK_PORT = 4000

FIELD_KEY_SEPARATOR = "|"
FIELD_ORDER_SEPARATOR = "!"
FIELD_ORDER_MIN = 1
FIELD_ORDER_MAX = 99


DEFAULT_PLACEHOLDERS = [
    "prompt",
    "negative_prompt",
    "model",
    "vae",
    "sampler",
    "scheduler",
    "steps",
    "cfg",
    "denoise",
    "clip_skip",
    "width",
    "height",
    "seed",
    "img_format",
    "lora",
]

PLACEHOLDER_FORM_HINTS: Dict[str, Dict[str, Any]] = {
    "prompt": {"component": "textarea", "lines": 6},
    "negative_prompt": {"component": "textarea", "lines": 4},
    "width": {"component": "int"},
    "height": {"component": "int"},
    "steps": {"component": "int"},
    "seed": {"component": "int"},
    "clip_skip": {"component": "int"},
    "cfg": {"component": "float", "precision": 2},
    "denoise": {"component": "float", "precision": 3},
    "model": {"component": "dropdown", "source": "checkpoints", "allow_custom": True},
    "vae": {"component": "dropdown", "source": "vae", "allow_custom": True},
    "sampler": {"component": "dropdown", "source": "samplers"},
    "scheduler": {"component": "dropdown", "source": "schedulers"},
    "lora": {"component": "dropdown", "source": "loras", "allow_custom": True},
    "img_format": {
        "component": "dropdown",
        "source": "image_formats",
        "allow_custom": True,
        "choices": ["png", "jpg", "jpeg", "webp", "bmp"],
    },
}

PLACEHOLDER_MAX_FIELDS = 40

OBJECT_INFO_CACHE: Dict[str, Any] = {}
OBJECT_INFO_TIMESTAMP: float = 0.0
OBJECT_INFO_ERRORS: List[str] = []
OBJECT_INFO_TTL_SECONDS = 600

# Cache for expensive operations
_CONFIG_CACHE: Optional[Dict] = None
_CONFIG_CACHE_TIME: float = 0.0
_STYLES_CACHE: Optional[Dict[str, Dict[str, str]]] = None
_STYLES_CACHE_TIME: float = 0.0
_PREFS_CACHE: Optional[Dict] = None
_PREFS_CACHE_TIME: float = 0.0
_CACHE_TTL: float = 30.0  # Cache for 30 seconds


def dedupe_and_sort_strings(items: List[str]) -> List[str]:
    """Utility function to deduplicate and sort strings efficiently."""
    if not items:
        return []
    return sorted(dict.fromkeys(items), key=str.lower)


def safe_dict_copy(item: Any) -> Dict[str, Any]:
    """Safely copy dictionary or return empty dict."""
    return dict(item) if isinstance(item, dict) else {}


def get_file_mtime(path: str) -> float:
    """Get file modification time safely."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def load_config() -> Dict:
    """Load server configuration with caching, falling back to sensible defaults."""
    global _CONFIG_CACHE, _CONFIG_CACHE_TIME

    now = time.time()
    config_mtime = get_file_mtime(CONFIG_PATH)

    # Use cache if valid and file hasn't changed
    if (_CONFIG_CACHE is not None and
        now - _CONFIG_CACHE_TIME < _CACHE_TTL and
        config_mtime <= _CONFIG_CACHE_TIME):
        return _CONFIG_CACHE

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except FileNotFoundError:
        config = {
            "comfyui_url": "http://127.0.0.1:8188",
            "workflow_path": os.path.join(WORKFLOW_DIR, "flux_schnell.json"),
            "output_dir": CACHE_ROOT_DEFAULT,
            "timeout": 120,
            "image_format": "png",
            "web_port": 8501,
            "network_access": False,
        }
    except json.JSONDecodeError:
        config = {}

    _CONFIG_CACHE = config
    _CONFIG_CACHE_TIME = now
    return config


def save_config(cfg: Dict) -> bool:
    global _CONFIG_CACHE
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        _CONFIG_CACHE = None  # Invalidate cache
        return True
    except OSError:
        return False


def load_styles() -> Dict[str, Dict[str, str]]:
    """Load styles with caching."""
    global _STYLES_CACHE, _STYLES_CACHE_TIME

    now = time.time()
    styles_mtime = get_file_mtime(STYLES_PATH)

    # Use cache if valid and file hasn't changed
    if (_STYLES_CACHE is not None and
        now - _STYLES_CACHE_TIME < _CACHE_TTL and
        styles_mtime <= _STYLES_CACHE_TIME):
        return _STYLES_CACHE

    try:
        with open(STYLES_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            styles = data
        else:
            styles = {"none": {"pre": "", "post": ""}}
    except (FileNotFoundError, json.JSONDecodeError):
        styles = {"none": {"pre": "", "post": ""}}

    _STYLES_CACHE = styles
    _STYLES_CACHE_TIME = now
    return styles


def save_styles(styles: Dict[str, Dict[str, str]]) -> bool:
    global _STYLES_CACHE
    try:
        with open(STYLES_PATH, "w", encoding="utf-8") as fh:
            json.dump(styles, fh, indent=2)
        _STYLES_CACHE = None  # Invalidate cache
        return True
    except OSError:
        return False


def load_placeholders() -> List[str]:
    """Load placeholder names from disk with a sane fallback."""
    try:
        with open(PLACEHOLDERS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            candidates = data.get("placeholders", [])
        else:
            candidates = data
        if isinstance(candidates, list):
            cleaned = [str(item).strip() for item in candidates if isinstance(item, str) and item.strip()]
            if cleaned:
                return cleaned
    except (OSError, json.JSONDecodeError):
        pass
    return DEFAULT_PLACEHOLDERS.copy()


def save_placeholders_list(placeholders: List[str]) -> None:
    ensure_directory(os.path.dirname(PLACEHOLDERS_PATH))
    with open(PLACEHOLDERS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"placeholders": placeholders}, fh, indent=2)


def get_user_preferences() -> Dict:
    """Get user preferences with caching."""
    global _PREFS_CACHE, _PREFS_CACHE_TIME

    now = time.time()
    prefs_mtime = get_file_mtime(PREFERENCES_PATH)

    # Use cache if valid and file hasn't changed
    if (_PREFS_CACHE is not None and
        now - _PREFS_CACHE_TIME < _CACHE_TTL and
        prefs_mtime <= _PREFS_CACHE_TIME):
        return _PREFS_CACHE

    try:
        with open(PREFERENCES_PATH, "r", encoding="utf-8") as fh:
            prefs = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        prefs = {"default_style": "none"}

    _PREFS_CACHE = prefs
    _PREFS_CACHE_TIME = now
    return prefs


def save_user_preferences(prefs: Dict) -> bool:
    global _PREFS_CACHE
    try:
        with open(PREFERENCES_PATH, "w", encoding="utf-8") as fh:
            json.dump(prefs, fh, indent=2)
        _PREFS_CACHE = None  # Invalidate cache
        return True
    except OSError:
        return False


def get_workflow_files() -> List[str]:
    if not os.path.exists(WORKFLOW_DIR):
        return []
    return sorted([f for f in os.listdir(WORKFLOW_DIR) if f.endswith(".json")])


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def workflow_settings_path(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    return os.path.join(WORKFLOW_SETTINGS_DIR, f"{stem}-settings.json")


def workflow_original_path(filename: str) -> str:
    return os.path.join(WORKFLOW_ORIGINALS_DIR, filename)


def ensure_workflow_original(filename: str, content: str) -> None:
    if not filename:
        return
    ensure_directory(WORKFLOW_ORIGINALS_DIR)
    target = workflow_original_path(filename)
    if not os.path.exists(target):
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(content)


def load_workflow_settings_full(filename: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    settings_file = workflow_settings_path(filename)
    try:
        with open(settings_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}, {}

    if not isinstance(data, dict):
        return {}, {}

    field_map = data.pop("__fields", {})
    if not isinstance(field_map, dict):
        field_map = {}
    return data, field_map


def save_workflow_settings_full(filename: str, placeholders_map: Dict[str, Any], field_map: Dict[str, Any]) -> None:
    ensure_directory(WORKFLOW_SETTINGS_DIR)
    payload = dict(placeholders_map)
    payload["__fields"] = field_map
    with open(workflow_settings_path(filename), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_workflow_file(filename: str) -> Tuple[Dict[str, Any], str]:
    path = os.path.join(WORKFLOW_DIR, filename)
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()
    data = json.loads(content)
    return data, content


def comfyui_base_url(cfg: Optional[Dict] = None) -> str:
    cfg = cfg or load_config()
    return (cfg.get("comfyui_url", "http://127.0.0.1:8188") or "http://127.0.0.1:8188").rstrip("/")


def _normalize_object_info(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        if "nodes" in data and isinstance(data["nodes"], dict):
            return data["nodes"]
        return data
    return {}


def refresh_object_info(force: bool = False) -> Tuple[Dict[str, Any], List[str], float]:
    global OBJECT_INFO_CACHE, OBJECT_INFO_TIMESTAMP, OBJECT_INFO_ERRORS

    now = time.time()
    if (
        not force
        and OBJECT_INFO_CACHE
        and now - OBJECT_INFO_TIMESTAMP < OBJECT_INFO_TTL_SECONDS
    ):
        return OBJECT_INFO_CACHE, [], OBJECT_INFO_TIMESTAMP

    base_url = comfyui_base_url()
    endpoint = f"{base_url}/object_info"
    notes: List[str] = []

    try:
        response = requests.get(endpoint, timeout=10)
        if response.status_code == 200:
            payload = response.json()
            if isinstance(payload, dict):
                OBJECT_INFO_CACHE = _normalize_object_info(payload)
                OBJECT_INFO_TIMESTAMP = now
                OBJECT_INFO_ERRORS = []
            else:
                notes.append("Object info response format unexpected.")
        else:
            notes.append(f"/object_info returned HTTP {response.status_code}")
    except requests.exceptions.RequestException as exc:
        notes.append(f"/object_info error: {exc}")

    if notes:
        OBJECT_INFO_ERRORS = notes
    return OBJECT_INFO_CACHE, notes, OBJECT_INFO_TIMESTAMP


def get_cached_object_info() -> Dict[str, Any]:
    cache, _, _ = refresh_object_info(force=False)
    return cache


def object_info_lookup(class_type: str, input_name: str) -> Tuple[Optional[str], Dict[str, Any]]:
    data = get_cached_object_info()
    node_info = data.get(class_type) if isinstance(data, dict) else None
    if not isinstance(node_info, dict):
        return None, {}

    input_sections: List[Dict[str, Any]] = []
    inputs = node_info.get("input") or node_info.get("inputs")
    if isinstance(inputs, dict):
        for key in ("required", "optional", "keyword", "kw", "hidden", "extras"):
            section = inputs.get(key)
            if isinstance(section, dict):
                input_sections.append(section)
        # Some nodes directly list inputs without nesting
        extra_inputs = {k: v for k, v in inputs.items() if isinstance(v, (list, dict))}
        if extra_inputs:
            input_sections.append(extra_inputs)
    else:
        # Some node definitions might store inputs at top level
        for key in ("required", "optional"):
            section = node_info.get(key)
            if isinstance(section, dict):
                input_sections.append(section)

    for section in input_sections:
        if input_name not in section:
            continue
        spec = section[input_name]
        if isinstance(spec, list) and spec:
            attrs: Dict[str, Any] = {}
            extra_attrs = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            attrs.update(extra_attrs)

            first = spec[0]
            if isinstance(first, list):
                choices = _sanitize_string_list(first)
                attrs.setdefault("choices", choices)
                attrs.setdefault("allow_custom", False)
                return "choice_list", attrs

            if isinstance(first, str):
                type_name = first
            else:
                type_name = str(first)

            if "choices" not in attrs:
                attrs["choices"] = _sanitize_string_list(spec)
                if attrs["choices"]:
                    attrs.setdefault("allow_custom", False)
            return type_name, attrs
        if isinstance(spec, dict):
            type_name = spec.get("type") or spec.get("kind")
            attrs = {k: v for k, v in spec.items() if k != "type"}
            return type_name, attrs
        if isinstance(spec, str):
            stripped = spec.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = ast.literal_eval(stripped)
                    if isinstance(parsed, list):
                        sanitized_choices = _sanitize_string_list(parsed)
                        return "choice_list", {"choices": sanitized_choices, "allow_custom": False}
                except (ValueError, SyntaxError):
                    pass
            return spec, {}

    return None, {}


def object_info_status_message(timestamp: float, notes: Optional[List[str]] = None) -> str:
    lines: List[str] = []
    if timestamp:
        synced_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"Last object info sync: `{synced_time}`")
    else:
        lines.append("Object info has not been loaded yet.")

    notes = notes or OBJECT_INFO_ERRORS
    for note in notes:
        if note:
            prefix = note[:2]
            if prefix in {"⚠️", "❌", "✅"}:
                lines.append(note)
            else:
                lines.append(f"⚠️ {note}")

    if not lines:
        lines.append("Object info status unavailable.")
    return "\n".join(lines)

def remove_associated_workflow_files(filename: str) -> List[str]:
    """Delete workflow file and associated copies in originals/settings."""
    removed: List[str] = []
    targets = [os.path.join(WORKFLOW_DIR, filename)]

    stem, _ = os.path.splitext(filename)
    targets.append(os.path.join(WORKFLOW_ORIGINALS_DIR, filename))
    targets.append(os.path.join(WORKFLOW_ORIGINALS_DIR, f"{stem}.json"))
    targets.append(os.path.join(WORKFLOW_SETTINGS_DIR, filename))
    targets.append(os.path.join(WORKFLOW_SETTINGS_DIR, f"{stem}-settings.json"))

    for path in targets:
        if path and os.path.isfile(path):
            try:
                os.remove(path)
                removed.append(path)
            except OSError:
                pass

    return removed


def workflow_field_key(node_id: str, input_name: str, order: Optional[int] = None) -> str:
    base = f"{node_id}{FIELD_KEY_SEPARATOR}{input_name}"
    if order is None:
        return base
    try:
        order_int = int(order)
    except (TypeError, ValueError):
        return base
    order_int = max(FIELD_ORDER_MIN, min(FIELD_ORDER_MAX, order_int))
    return f"{order_int}{FIELD_ORDER_SEPARATOR}{base}"


def parse_field_storage_key(key: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    if not isinstance(key, str):
        return None, None, None
    rest = key
    order: Optional[int] = None
    if FIELD_ORDER_SEPARATOR in rest:
        order_part, rest = rest.split(FIELD_ORDER_SEPARATOR, 1)
        try:
            order = int(order_part)
        except ValueError:
            order = None
    if FIELD_KEY_SEPARATOR in rest:
        node_id, input_name = rest.split(FIELD_KEY_SEPARATOR, 1)
    else:
        node_id, input_name = None, None
    return order, node_id, input_name


def find_field_value(
    field_map: Dict[str, Any], node_id: str, input_name: str
) -> Tuple[Optional[str], Optional[Any]]:
    for key, value in field_map.items():
        _, candidate_node, candidate_input = parse_field_storage_key(key)
        if candidate_node == node_id and candidate_input == input_name:
            return key, value
    return None, None


def convert_string_to_type(value: str, type_name: str) -> Tuple[Optional[Any], Optional[str]]:
    if type_name == "int":
        try:
            return int(value), None
        except ValueError:
            return None, "Enter a valid integer"
    if type_name == "float":
        try:
            return float(value), None
        except ValueError:
            return None, "Enter a valid number"
    return value, None


def parse_workflow_for_configuration(filename: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return structured field data and workflow dict for configuration editing."""
    if not filename:
        return [], {}

    workflow_data, raw_content = load_workflow_file(filename)
    ensure_workflow_original(filename, raw_content)

    placeholder_defaults, field_defaults_raw = load_workflow_settings_full(filename)
    placeholders = load_placeholders()
    placeholder_set = set(placeholders)

    field_defaults: Dict[str, Any] = {}
    node_orders: Dict[str, int] = {}

    for stored_key, stored_value in field_defaults_raw.items():
        order, node_id, input_name = parse_field_storage_key(stored_key)
        if not node_id or not input_name:
            continue
        base_key = f"{node_id}{FIELD_KEY_SEPARATOR}{input_name}"
        field_defaults[base_key] = stored_value
        if order is not None and node_id not in node_orders:
            node_orders[node_id] = normalize_field_order(order, len(node_orders) + 1)

    node_entries: List[Dict[str, Any]] = []
    sequence = 0
    for node_id, node in workflow_data.items():
        inputs = node.get("inputs", {})
        meta = node.get("_meta", {})
        title = meta.get("title", node_id)

        node_fields: List[Dict[str, Any]] = []

        for input_name, value in inputs.items():
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                continue

            if isinstance(value, float) and value.is_integer():
                # Preserve intent: if value came as float but integral, keep float type
                value_type = "float"
            elif isinstance(value, int):
                value_type = "int"
            elif isinstance(value, float):
                value_type = "float"
            else:
                value_type = "str"

            base_key = f"{node_id}{FIELD_KEY_SEPARATOR}{input_name}"
            stored_value = field_defaults.get(base_key)
            placeholder_choice = ""
            text_value = value

            if isinstance(value, str) and value.startswith("%") and value.endswith("%"):
                candidate = value.strip("%")
                if candidate in placeholder_set:
                    placeholder_choice = candidate
                    stored_value = placeholder_defaults.get(candidate, stored_value)
                    text_value = value

            if stored_value is None:
                stored_value = value
            if placeholder_choice and isinstance(stored_value, str) and stored_value.startswith("%") and stored_value.endswith("%"):
                stored_value = ""

            field_entry = {
                "node_id": node_id,
                "node_title": title,
                "input_name": input_name,
                "class_type": node.get("class_type", ""),
                "type": value_type,
                "placeholder": placeholder_choice,
                "stored_value": str(stored_value),
                "text_value": str(text_value) if placeholder_choice else str(stored_value),
            }
            node_fields.append(field_entry)

        if node_fields:
            node_entries.append(
                {
                    "node_id": node_id,
                    "node_title": title,
                    "order": node_orders.get(node_id),
                    "fields": node_fields,
                    "seq": sequence,
                }
            )
            sequence += 1

    if not node_entries:
        return [], workflow_data

    for idx, entry in enumerate(node_entries, start=1):
        entry["order"] = normalize_field_order(entry.get("order"), idx)

    node_entries.sort(key=lambda item: (item.get("order", FIELD_ORDER_MAX + 1), item.get("seq", 0)))

    fields: List[Dict[str, Any]] = []
    node_sequence = 0
    for entry in node_entries:
        node_id = entry.get("node_id")
        if not node_id:
            continue
        node_sequence += 1
        node_order = normalize_field_order(entry.get("order"), node_sequence)
        node_title = entry.get("node_title", node_id)
        for field_idx, field in enumerate(entry.get("fields", [])):
            field_copy = dict(field)
            field_copy["order"] = node_order
            is_primary = field_idx == 0
            field_copy["is_primary"] = is_primary
            field_copy["display_node_title"] = node_title if is_primary else ""
            fields.append(field_copy)

    return fields, workflow_data


def _sanitize_string_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    valid_items = []
    for item in items:
        if isinstance(item, str):
            trimmed = item.strip()
            if trimmed:
                valid_items.append(trimmed)
    return dedupe_and_sort_strings(valid_items)


def _coerce_placeholder_value(name: str, component: str, value: Any) -> Tuple[Any, Optional[str]]:
    if component in ("text", "textarea", "dropdown"):
        if value is None:
            return "", None
        if isinstance(value, (int, float)):
            return str(value), None
        return str(value), None

    if component == "int":
        if isinstance(value, int):
            return value, None
        if isinstance(value, float) and value.is_integer():
            return int(value), None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                try:
                    return int(stripped), None
                except ValueError:
                    return None, f"Could not parse integer for `{name}`"
        if value in (None, ""):
            return None, None
        return None, f"Unexpected type for `{name}`"

    if component == "float":
        if isinstance(value, (int, float)):
            return float(value), None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                try:
                    return float(stripped), None
                except ValueError:
                    return None, f"Could not parse number for `{name}`"
        if value in (None, ""):
            return None, None
        return None, f"Unexpected type for `{name}`"

    if component == "checkbox":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True, None
            if lowered in {"false", "0", "no", "off"}:
                return False, None
        if value in (None, ""):
            return False, None
        return False, f"Could not parse boolean for `{name}`"

    return value, None


def infer_placeholder_component(
    name: str,
    usage_types: Optional[List[str]],
    hints: Dict[str, Any],
    value: Any,
    has_options: bool,
) -> str:
    component = hints.get("component") if hints else None
    if component == "dropdown" and not has_options:
        component = None

    if component:
        return component

    normalized = [str(item).lower() for item in (usage_types or [])]
    if any("bool" in item for item in normalized):
        return "checkbox"
    if any("int" in item for item in normalized):
        return "int"
    if any(item in {"float", "double"} or "float" in item for item in normalized):
        return "float"
    if any(item in {"string", "text"} or "string" in item or "text" in item for item in normalized):
        return "text"
    if any("choice" in item or "enum" in item for item in normalized):
        return "dropdown"
    if any("model" in item or "file" in item for item in normalized) and has_options:
        return "dropdown"

    if isinstance(value, bool):
        return "checkbox"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"

    usage_types = usage_types or []
    if "float" in usage_types:
        return "float"
    if "int" in usage_types:
        return "int"
    if name.lower().endswith("_prompt") or "prompt" in name.lower():
        return "textarea"
    return "text"


def gather_placeholder_form_fields(
    filename: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not filename:
        return [], ["⚠️ Select a workflow to edit placeholder defaults."]

    try:
        fields, _ = parse_workflow_for_configuration(filename)
    except FileNotFoundError:
        return [], [f"⚠️ Workflow `{filename}` not found."]
    except json.JSONDecodeError as exc:
        return [], [f"❌ Failed to parse workflow `{filename}`: {exc}"]

    usage: Dict[str, Dict[str, Any]] = {}
    for field in fields:
        placeholder = field.get("placeholder")
        if not placeholder:
            continue
        info = usage.setdefault(placeholder, {"types": set(), "nodes": [], "fields": []})
        field_type = field.get("type") or "str"
        info["types"].add(field_type)
        title = field.get("node_title") or field.get("node_id")
        if title and title not in info["nodes"]:
            info["nodes"].append(title)
        info["fields"].append(field)

    if not usage:
        return [], ["⚠️ No placeholders detected in this workflow."]

    stored_placeholders, _ = load_workflow_settings_full(filename)

    placeholder_priority = load_placeholders()
    warnings: List[str] = []
    fetch_notes: List[str] = []

    priority_lookup = {name: idx for idx, name in enumerate(placeholder_priority)}
    display_entries: List[Dict[str, Any]] = []

    for name, info in usage.items():
        hints = PLACEHOLDER_FORM_HINTS.get(name, {})
        metadata = analyze_object_info_for_fields(info.get("fields", []))
        stored_value = stored_placeholders.get(name)
        if (stored_value in (None, "")) and metadata.get("default_values"):
            stored_value = metadata["default_values"][0]

        options: List[str] = []
        if hints.get("component") == "dropdown":
            if hints.get("choices"):
                options.extend(hints["choices"])
            meta_choices = metadata.get("choices", [])
            options.extend(meta_choices)
            options = sorted(dict.fromkeys(options), key=str.lower)
        else:
            meta_choices = metadata.get("choices", [])
            options.extend(meta_choices)

        component_hint = hints.get("component")
        component = infer_placeholder_component(
            name,
            list(info.get("types", [])) + list(metadata.get("type_names", [])),
            hints,
            stored_value,
            bool(options),
        )
        if component_hint:
            component = component_hint
            if component == "dropdown" and not options:
                meta_choices = metadata.get("choices", [])
                if meta_choices:
                    options.extend(meta_choices)
                    options = sorted(dict.fromkeys(options), key=str.lower)

        coerced_value, warning = _coerce_placeholder_value(name, component, stored_value)
        if warning:
            warnings.append(warning)

        if component == "dropdown":
            allow_custom = bool(hints.get("allow_custom", True))
            if metadata.get("allow_custom") is False:
                allow_custom = False
            if allow_custom and isinstance(coerced_value, str) and coerced_value:
                if coerced_value not in options:
                    options.append(coerced_value)
            elif not allow_custom and options:
                if coerced_value not in options:
                    coerced_value = options[0] if options else ""
            if isinstance(coerced_value, str) and coerced_value and coerced_value not in options:
                options.append(coerced_value)
            options = sorted(dict.fromkeys(options), key=str.lower)
        else:
            allow_custom = bool(hints.get("allow_custom", False))

        numeric_min = metadata.get("min")
        numeric_max = metadata.get("max")
        numeric_step = metadata.get("step")
        if component == "int":
            numeric_step = int(numeric_step) if isinstance(numeric_step, (int, float)) else None
        elif component == "float" and isinstance(numeric_step, int):
            numeric_step = float(numeric_step)

        lines = hints.get("lines") if component in {"text", "textarea"} else None
        precision = hints.get("precision") if component == "float" else None

        fields_for_placeholder = info.get("fields", []) or []
        if not fields_for_placeholder:
            continue

        fields_for_placeholder_sorted = sorted(
            fields_for_placeholder,
            key=lambda f: (
                normalize_field_order(f.get("order"), FIELD_ORDER_MAX + 1),
                0 if f.get("is_primary") else 1,
                str(f.get("input_name", "")).lower(),
            ),
        )

        primary_field = next(
            (f for f in fields_for_placeholder_sorted if f.get("is_primary")),
            fields_for_placeholder_sorted[0],
        )
        order_value = normalize_field_order(
            primary_field.get("order"),
            priority_lookup.get(name, FIELD_ORDER_MAX + 1),
        )
        node_title = (
            primary_field.get("display_node_title")
            or primary_field.get("node_title")
            or primary_field.get("node_id")
            or ""
        )

        for field_detail in fields_for_placeholder_sorted:
            is_primary = bool(field_detail.get("is_primary"))
            display_title = node_title if is_primary else ""
            field_label = field_detail.get("input_name") or hints.get("label") or name
            entry = {
                "name": name,
                "label": field_label,
                "field_label": field_label,
                "component": component,
                "value": coerced_value,
                "options": options,
                "allow_custom": allow_custom,
                "lines": lines or (4 if component == "textarea" else 1),
                "precision": precision,
                "raw_value": stored_value,
                "min": numeric_min,
                "max": numeric_max,
                "step": numeric_step,
                "metadata": metadata,
                "order": order_value,
                "node_title": node_title,
                "display_node_title": display_title,
                "is_primary": is_primary,
                "node_id": field_detail.get("node_id"),
                "input_name": field_detail.get("input_name"),
            }
            entry["_sort_key"] = (
                order_value,
                str(node_title).lower(),
                0 if is_primary else 1,
                priority_lookup.get(name, len(priority_lookup)),
                str(field_label).lower(),
            )
            display_entries.append(entry)

    display_entries.sort(key=lambda item: item.pop("_sort_key"))

    return display_entries, warnings


def build_workflow_config_updates(fields: List[Dict[str, Any]]) -> List[Any]:
    placeholder_choices = [""] + load_placeholders()
    updates: List[Any] = []
    for idx in range(MAX_WORKFLOW_FIELDS):
        if idx < len(fields):
            field = fields[idx]
            order_value = normalize_field_order(field.get("order"), idx + 1)
            is_primary = bool(field.get("is_primary"))
            display_title = field.get("display_node_title", field.get("node_title", ""))
            updates.extend(
                [
                    gr.update(value=order_value, visible=is_primary, interactive=is_primary),
                    gr.update(value=display_title, visible=True),
                    gr.update(value=field["input_name"], visible=True),
                    gr.update(
                        value=field["text_value"],
                        visible=True,
                        interactive=field["placeholder"] == "",
                    ),
                    gr.update(
                        choices=placeholder_choices,
                        value=field["placeholder"],
                        visible=True,
                    ),
                ]
            )
        else:
            updates.extend(
                [
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(choices=placeholder_choices, value="", visible=False),
                ]
            )
    return updates


def placeholder_table_rows() -> List[List[str]]:
    return [[name] for name in load_placeholders()]


def format_placeholder_label(field: Dict[str, Any]) -> str:
    field_label = field.get("field_label") or field.get("label") or field.get("name") or ""
    node_title = field.get("display_node_title") or ""
    order_value = field.get("order")

    parts: List[str] = []
    if node_title:
        if order_value is not None:
            parts.append(f"**{order_value:02d}. {node_title}**")
        else:
            parts.append(f"**{node_title}**")
    if field_label:
        prefix = "- " if node_title else ""
        parts.append(f"{prefix}{field_label}")
    return "\n".join(parts)


def build_placeholder_form_updates(fields: List[Dict[str, Any]]) -> List[Any]:
    updates: List[Any] = []
    for idx in range(PLACEHOLDER_MAX_FIELDS):
        if idx < len(fields):
            field = fields[idx]
            component = field.get("component")
            updates.append(
                gr.update(value=format_placeholder_label(field), visible=True)
            )

            if component in {"text", "textarea"}:
                updates.append(
                    gr.update(
                        value=field.get("value", ""),
                        visible=True,
                        lines=int(field.get("lines", 1)),
                        interactive=True,
                    )
                )
            else:
                updates.append(gr.update(visible=False))

            if component in {"int", "float"}:
                precision = 0 if component == "int" else field.get("precision") or 3
                updates.append(
                    gr.update(
                        value=field.get("value"),
                        visible=True,
                        interactive=True,
                        precision=precision,
                        minimum=field.get("min"),
                        maximum=field.get("max"),
                        step=field.get("step"),
                    )
                )
            else:
                updates.append(gr.update(visible=False))

            if component == "checkbox":
                updates.append(
                    gr.update(
                        value=bool(field.get("value")),
                        visible=True,
                        interactive=True,
                    )
                )
            else:
                updates.append(gr.update(visible=False))

            if component == "dropdown":
                updates.append(
                    gr.update(
                        value=field.get("value", ""),
                        choices=field.get("options") or [],
                        visible=True,
                        interactive=True,
                        allow_custom_value=bool(field.get("allow_custom")),
                    )
                )
            else:
                updates.append(gr.update(visible=False))
        else:
            updates.extend(
                [
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                ]
            )
    return updates


def compose_placeholder_status_message(
    filename: Optional[str], messages: List[str], has_fields: bool
) -> str:
    lines: List[str] = []
    if has_fields and filename:
        lines.append(f"✅ Loaded placeholder defaults for `{filename}`.")

    for msg in messages:
        if not msg:
            continue
        if msg.startswith(("⚠️", "❌", "✅", "- ")):
            lines.append(msg)
        else:
            lines.append(f"- {msg}")

    if not lines:
        if filename:
            lines.append(f"✅ Loaded placeholder defaults for `{filename}`.")
        else:
            lines.append("⚠️ Select a workflow to edit placeholder defaults.")

    return "\n".join(lines)


def analyze_object_info_for_fields(fields: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "type_names": set(),
        "choices": set(),
        "default_values": [],
        "min": None,
        "max": None,
        "step": None,
        "allow_custom": True,
    }

    for field in fields:
        class_type = field.get("class_type") or ""
        input_name = field.get("input_name") or ""
        if not class_type or not input_name:
            continue
        type_name, attrs = object_info_lookup(class_type, input_name)
        if type_name:
            result["type_names"].add(str(type_name))
        if not isinstance(attrs, dict):
            continue

        if "choices" in attrs:
            choices = attrs.get("choices")
            if isinstance(choices, dict):
                iterable = choices.values()
            else:
                iterable = choices
            for choice in iterable or []:
                result["choices"].add(str(choice))
            if attrs.get("force_input") or attrs.get("restrict_to_choices"):
                result["allow_custom"] = False

        if "default" in attrs and attrs["default"] not in (None, ""):
            result["default_values"].append(attrs["default"])

        for key in ("min", "max", "step"):
            value = attrs.get(key)
            if value is None:
                continue
            if result[key] is None:
                result[key] = value

        if attrs.get("allow_custom") is False:
            result["allow_custom"] = False

    result["type_names"] = sorted(result.get("type_names", set()))
    result["choices"] = sorted(result.get("choices", set()), key=str.lower)
    return result


def prepare_placeholder_for_save(field: Dict[str, Any]) -> Tuple[bool, Any, Optional[str]]:
    name = field.get("name", "")
    component = field.get("component")
    value = field.get("value")

    if component in {"text", "textarea", "dropdown"}:
        return True, "" if value is None else str(value), None

    if component == "int":
        if value in (None, ""):
            return False, None, None
        if isinstance(value, int):
            return True, value, None
        if isinstance(value, float) and value.is_integer():
            return True, int(value), None
        try:
            return True, int(str(value).strip())
        except (ValueError, AttributeError):
            return False, None, f"Enter a valid integer for `{name}`"

    if component == "float":
        if value in (None, ""):
            return False, None, None
        if isinstance(value, (int, float)):
            return True, float(value), None
        try:
            return True, float(str(value).strip())
        except (ValueError, AttributeError):
            return False, None, f"Enter a valid number for `{name}`"

    if component == "checkbox":
        return True, bool(value), None

    return True, value, None


def build_placeholder_form_response(
    filename: Optional[str],
    fields: List[Dict[str, Any]],
    messages: List[str],
) -> List[Any]:
    # Limit fields and only copy if we need to modify them
    safe_fields = fields[:PLACEHOLDER_MAX_FIELDS]
    status = compose_placeholder_status_message(filename, messages, bool(safe_fields))
    updates = build_placeholder_form_updates(safe_fields)
    payload = {"filename": filename, "fields": safe_fields}
    return [payload, gr.update(value=status)] + updates


def apply_fields_to_workflow(
    filename: str,
    workflow_data: Dict[str, Any],
    fields: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any], Set[str]]:
    errors: List[str] = []
    placeholders_map: Dict[str, Any] = {}
    field_map: Dict[str, Any] = {}
    active_placeholders: Set[str] = set()

    for idx, field in enumerate(fields):
        node = workflow_data.get(field["node_id"])
        if not node:
            continue
        inputs = node.setdefault("inputs", {})
        order_value = normalize_field_order(field.get("order"), idx + 1)
        field["order"] = order_value
        field_key = workflow_field_key(field["node_id"], field["input_name"], order_value)
        placeholder = field.get("placeholder", "")
        value_type = field.get("type", "str")

        if placeholder:
            active_placeholders.add(placeholder)
            stored_raw = field.get("stored_value", "")
            placeholder_value: Optional[Any]
            err: Optional[str] = None

            if isinstance(stored_raw, str) and stored_raw.strip().startswith("%") and stored_raw.strip().endswith("%"):
                placeholder_value = None
            elif stored_raw in (None, ""):
                placeholder_value = None
            else:
                placeholder_value, err = convert_string_to_type(str(stored_raw), value_type)

            if err:
                errors.append(f"{field['node_title']} → {field['input_name']}: {err}")
                continue

            inputs[field["input_name"]] = f"%{placeholder}%"
            if placeholder_value is not None:
                placeholders_map[placeholder] = placeholder_value
                field_map[field_key] = placeholder_value
        else:
            text_value = field.get("text_value", "")
            typed_value, err = convert_string_to_type(text_value, value_type)
            if err:
                errors.append(f"{field['node_title']} → {field['input_name']}: {err}")
                continue
            inputs[field["input_name"]] = typed_value
            field_map[field_key] = typed_value

    return errors, placeholders_map, field_map, workflow_data, active_placeholders


def update_field_value(fields: List[Dict[str, Any]], index: int, new_value: str) -> List[Dict[str, Any]]:
    if not (0 <= index < len(fields)):
        return fields

    # Only copy if we need to modify
    fields = fields.copy()
    field = safe_dict_copy(fields[index])
    field["text_value"] = new_value
    field["stored_value"] = new_value
    fields[index] = field
    return fields


def update_field_placeholder(
    fields: List[Dict[str, Any]], index: int, placeholder: str
) -> Tuple[List[Dict[str, Any]], str, bool]:
    if not (0 <= index < len(fields)):
        return fields, "", False

    fields = fields.copy()
    field = safe_dict_copy(fields[index])
    placeholder = placeholder or ""

    if placeholder:
        # Store current non-placeholder value for restoration.
        if field.get("placeholder", "") == "":
            field["stored_value"] = field.get("text_value", "")
        field["placeholder"] = placeholder
        field["text_value"] = f"%{placeholder}%"
        fields[index] = field
        return fields, field["text_value"], False

    # Restoring to concrete value
    restore_value = field.get("stored_value", "")
    field["placeholder"] = ""
    field["text_value"] = restore_value
    fields[index] = field
    return fields, restore_value, True


def update_placeholder_field_value(
    fields: List[Dict[str, Any]], index: int, new_value: Any
) -> List[Dict[str, Any]]:
    if not (0 <= index < len(fields)):
        return fields

    updated_fields = [dict(item) for item in fields]
    field = dict(updated_fields[index])
    field["value"] = new_value

    if (
        field.get("component") == "dropdown"
        and field.get("allow_custom")
        and isinstance(new_value, str)
        and new_value
    ):
        options = list(field.get("options") or [])
        if new_value not in options:
            options.append(new_value)
            field["options"] = sorted(dict.fromkeys(options), key=str.lower)

    updated_fields[index] = field
    return updated_fields


def normalize_field_order(value: Any, fallback: int) -> int:
    try:
        order = int(value)
    except (TypeError, ValueError):
        order = fallback
    order = max(FIELD_ORDER_MIN, min(FIELD_ORDER_MAX, order))
    return order


def update_field_order(
    fields: List[Dict[str, Any]], index: int, new_value: Any
) -> List[Dict[str, Any]]:
    if not (0 <= index < len(fields)):
        return fields
    updated_fields = [dict(item) for item in fields]
    target_field = dict(updated_fields[index])
    target_node = target_field.get("node_id")
    new_order = normalize_field_order(new_value, index + 1)

    for idx, item in enumerate(updated_fields):
        if item.get("node_id") != target_node:
            continue
        item_copy = dict(item)
        item_copy["order"] = new_order
        updated_fields[idx] = item_copy

    return updated_fields


def set_active_workflow(filename: str) -> Tuple[bool, str]:
    config = load_config()
    if not filename:
        return False, "No workflow selected"
    candidate = os.path.join(WORKFLOW_DIR, filename)
    if not os.path.exists(candidate):
        return False, f"Workflow {filename} not found"
    config["workflow_path"] = candidate
    ok = save_config(config)
    if ok:
        return True, f"Active workflow set to {filename}. Restart Flask to apply."
    return False, "Failed to update configuration"


def build_api_markdown(cfg: Dict) -> str:
    base_url = f"http://127.0.0.1:{FLASK_PORT}"
    network_enabled = cfg.get("network_access", False)
    parts = ["### 📡 API Endpoint", f"`{base_url}/prompt/[PROMPT]?width=WIDTH&height=HEIGHT&style=STYLE`"]
    if network_enabled:
        try:
            import socket

            hostname = socket.gethostname()
            ip_addr = socket.gethostbyname(hostname)
            network_url = f"http://{ip_addr}:{FLASK_PORT}"
        except Exception:
            network_url = f"http://<host-ip>:{FLASK_PORT}"
        parts.append("")
        parts.append("**Network accessible:**")
        parts.append(f"`{network_url}/prompt/[PROMPT]`")
    else:
        parts.append("")
        parts.append("Network access is disabled. Enable it under Settings → Server Configuration.")
    return "\n".join(parts)


def build_style_preview(style_name: str, styles: Optional[Dict[str, Dict[str, str]]] = None) -> str:
    if styles is None:
        styles = load_styles()
    style = styles.get(style_name, styles.get("none", {"pre": "", "post": ""}))
    pre = style.get("pre", "").strip()
    post = style.get("post", "").strip()
    sample = "a beautiful landscape"
    final_prompt = sample
    if pre:
        final_prompt = f"{pre} {final_prompt}"
    if post:
        final_prompt = f"{final_prompt}, {post}"
    metadata = [f"**Current style:** `{style_name}`"]
    if pre:
        metadata.append(f"- Pre: {pre}")
    if post:
        metadata.append(f"- Post: {post}")
    metadata.append("")
    metadata.append("**Sample prompt:**")
    metadata.append(f"`{final_prompt}`")
    return "\n".join(metadata)


def check_comfyui_status(cfg: Optional[Dict] = None) -> Tuple[bool, str]:
    cfg = cfg or load_config()
    comfy_url = cfg.get("comfyui_url", "http://127.0.0.1:8188")
    try:
        resp = requests.get(f"{comfy_url}/system_stats", timeout=5)
        if resp.status_code == 200:
            return True, "Connected"
        return False, f"HTTP {resp.status_code}"
    except requests.exceptions.RequestException as exc:
        return False, str(exc)


def check_flask_status() -> Tuple[bool, str]:
    try:
        resp = requests.get(f"http://127.0.0.1:{FLASK_PORT}/health", timeout=3)
        if resp.status_code == 200:
            try:
                payload = resp.json()
                status_msg = payload.get("status", "ok")
            except ValueError:
                status_msg = "Responding"
            return True, status_msg
        return False, f"HTTP {resp.status_code}"
    except requests.exceptions.RequestException as exc:
        return False, str(exc)


def compute_cache_stats(cfg: Optional[Dict] = None) -> Tuple[str, List[List[str]]]:
    cfg = cfg or load_config()
    cache_root = cfg.get("output_dir", CACHE_ROOT_DEFAULT)
    if not os.path.exists(cache_root):
        return f"Cache directory `{cache_root}` does not exist yet.", []

    total_files = 0
    total_size = 0
    per_user: List[List[str]] = []

    for user_id in sorted(os.listdir(cache_root)):
        user_path = os.path.join(cache_root, user_id)
        if not os.path.isdir(user_path):
            continue
        user_files = 0
        user_size = 0
        last_activity = 0
        for file_name in os.listdir(user_path):
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            file_path = os.path.join(user_path, file_name)
            try:
                stat = os.stat(file_path)
            except OSError:
                continue
            user_files += 1
            user_size += stat.st_size
            total_files += 1
            total_size += stat.st_size
            last_activity = max(last_activity, int(stat.st_mtime))
        if user_files:
            last_seen = datetime.fromtimestamp(last_activity).strftime("%Y-%m-%d %H:%M:%S")
            per_user.append([
                user_id,
                str(user_files),
                f"{user_size / (1024 * 1024):.2f}",
                last_seen,
            ])

    summary = (
        f"**Total images:** {total_files}\n"
        f"**Total size:** {total_size / (1024 * 1024):.2f} MB\n"
        f"**Users with cache:** {len(per_user)}"
    )
    return summary, per_user


def format_cache_table(per_user: List[List[str]]) -> str:
    """Render cache statistics as a Markdown table for display."""
    if not per_user:
        return "No cached images found."

    header = "| User | Images | Size (MB) | Last activity |\n| --- | --- | --- | --- |"
    rows = []
    for user, images, size_mb, last_seen in per_user:
        label = f"{user[:12]}…" if len(user) > 12 else user
        rows.append(f"| {label} | {images} | {size_mb} | {last_seen} |")
    return "\n".join([header, *rows])


def gather_monitoring_data() -> Tuple[str, str, str, str]:
    cfg = load_config()
    comfy_ok, comfy_msg = check_comfyui_status(cfg)
    flask_ok, flask_msg = check_flask_status()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    status_lines = ["### 🖥️ Service Status"]
    status_lines.append(f"- ComfyUI: {'✅' if comfy_ok else '❌'} {comfy_msg}")
    status_lines.append(f"- Flask API: {'✅' if flask_ok else '❌'} {flask_msg}")
    status_lines.append(f"- Checked at: {now}")

    cache_summary, per_user = compute_cache_stats(cfg)
    cache_table_md = format_cache_table(per_user)

    config_lines = ["### ⚙️ Configuration Snapshot"]
    config_lines.append(f"- ComfyUI URL: {cfg.get('comfyui_url', 'n/a')}")
    workflow_name = os.path.basename(cfg.get("workflow_path", "not set"))
    config_lines.append(f"- Active workflow: {workflow_name}")
    config_lines.append(f"- Timeout: {cfg.get('timeout', 30)} seconds")
    config_lines.append(f"- Image format: {cfg.get('image_format', 'png')}")
    config_lines.append(
        f"- Network access: {'enabled' if cfg.get('network_access') else 'disabled'}"
    )

    return "\n".join(status_lines), cache_summary, cache_table_md, "\n".join(config_lines)


def detect_workflow_placeholders(content: str) -> Dict[str, bool]:
    placeholders = load_placeholders()
    return {name: f"%{name}%" in content for name in placeholders}


def load_workflow_content(filename: str) -> Tuple[str, str]:
    if not filename:
        return "", "No workflow selected"
    path = os.path.join(WORKFLOW_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
    except OSError as exc:
        return "", f"Failed to read workflow: {exc}"
    placeholders = detect_workflow_placeholders(content)
    summary_lines = ["### Placeholders Detected"]
    summary_lines.extend(
        [f"- `{name}`: {'✅' if present else '⬜️'}" for name, present in placeholders.items()]
    )
    return content, "\n".join(summary_lines)


def save_workflow_content(filename: str, content: str) -> Tuple[bool, str]:
    if not filename:
        return False, "No workflow selected"
    path = os.path.join(WORKFLOW_DIR, filename)
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON: {exc}"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return True, f"Saved {filename}"
    except OSError as exc:
        return False, f"Failed to save workflow: {exc}"


def restart_flask() -> str:
    try:
        response = requests.post(f"http://127.0.0.1:{FLASK_PORT}/admin/restart", timeout=3)
        if response.status_code == 202:
            return "Restart requested. Flask will reboot shortly."
        return f"Restart request returned HTTP {response.status_code}"
    except requests.exceptions.RequestException as exc:
        return f"Failed to contact Flask API: {exc}"


def clear_cache() -> str:
    cfg = load_config()
    cache_root = cfg.get("output_dir", CACHE_ROOT_DEFAULT)
    if not os.path.exists(cache_root):
        return "Cache directory does not exist yet."
    removed = 0
    for root, dirs, files in os.walk(cache_root, topdown=False):
        for file_name in files:
            try:
                os.remove(os.path.join(root, file_name))
                removed += 1
            except OSError:
                pass
        for dir_name in dirs:
            try:
                os.rmdir(os.path.join(root, dir_name))
            except OSError:
                pass
    return f"Cleared cache entries ({removed} files removed)."


def ensure_preview_dir() -> str:
    cfg = load_config()
    cache_root = cfg.get("output_dir", CACHE_ROOT_DEFAULT)
    preview_root = os.path.join(cache_root, "gradio_previews")
    os.makedirs(preview_root, exist_ok=True)
    return preview_root


def generate_image(prompt: str, width: int, height: int, style: str) -> Tuple[Optional[str], str]:
    prompt = (prompt or "").strip()
    if not prompt:
        return None, "⚠️ Please enter a prompt."

    comfy_ok, _ = check_comfyui_status()
    flask_ok, _ = check_flask_status()
    if not comfy_ok:
        return None, "❌ ComfyUI is not reachable."
    if not flask_ok:
        return None, "❌ Flask API is not running."

    encoded_prompt = requests.utils.quote(prompt)
    params = {"width": int(width), "height": int(height)}
    if style:
        params["style"] = style

    cfg = load_config()
    timeout = int(cfg.get("timeout", 30)) + 10
    try:
        response = requests.get(
            f"http://127.0.0.1:{FLASK_PORT}/prompt/{encoded_prompt}",
            params=params,
            timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        return None, f"❌ Request failed: {exc}"

    if response.status_code != 200:
        return None, f"❌ Generation failed: HTTP {response.status_code}"

    preview_dir = ensure_preview_dir()
    filename = os.path.join(preview_dir, f"generated_{int(time.time())}.png")
    try:
        with open(filename, "wb") as fh:
            fh.write(response.content)
    except OSError as exc:
        return None, f"❌ Failed to write preview image: {exc}"

    return filename, "✅ Image generated successfully."


def update_default_style(style_name: str) -> Tuple[str, str]:
    prefs = get_user_preferences()
    prefs["default_style"] = style_name
    message = ""
    if save_user_preferences(prefs):
        message = f"✅ Default style set to `{style_name}`"
    else:
        message = "❌ Failed to save default style"
    preview = build_style_preview(style_name)
    return message, preview


def refresh_styles_dropdowns(styles: Optional[Dict[str, Dict[str, str]]] = None,
                            prefs: Optional[Dict] = None) -> Tuple[Any, Any]:
    if styles is None:
        styles = load_styles()
    if prefs is None:
        prefs = get_user_preferences()
    choices = list(styles.keys())
    default_style = prefs.get("default_style", choices[0] if choices else None)
    return (
        gr.update(choices=choices, value=default_style),
        gr.update(choices=choices, value=default_style),
    )


def add_style(
    name: str, pre: str, post: str
) -> Tuple[str, str, str, Any, Any]:
    name = (name or "").strip()
    styles = load_styles()
    prefs = get_user_preferences()
    default_style = prefs.get("default_style", "none")

    if not name:
        preview = build_style_preview(default_style, styles)
        return "⚠️ Style name is required.", "", preview, *refresh_styles_dropdowns(styles, prefs)
    if name in styles:
        preview = build_style_preview(default_style, styles)
        return f"⚠️ Style `{name}` already exists.", "", preview, *refresh_styles_dropdowns(styles, prefs)

    styles[name] = {"pre": pre or "", "post": post or ""}
    status = "✅ Style added." if save_styles(styles) else "❌ Failed to save styles."
    manage_preview = build_style_preview(name, styles)
    dashboard_preview = build_style_preview(default_style, styles)
    return status, manage_preview, dashboard_preview, *refresh_styles_dropdowns(styles, prefs)


def update_style(
    name: str, pre: str, post: str
) -> Tuple[str, str, str, Any, Any]:
    name = (name or "").strip()
    styles = load_styles()
    prefs = get_user_preferences()
    default_style = prefs.get("default_style", "none")

    if name not in styles:
        preview = build_style_preview(default_style, styles)
        return f"⚠️ Style `{name}` not found.", "", preview, *refresh_styles_dropdowns(styles, prefs)

    styles[name] = {"pre": pre or "", "post": post or ""}
    status = "✅ Style updated." if save_styles(styles) else "❌ Failed to save styles."
    manage_preview = build_style_preview(name, styles)
    dashboard_preview = build_style_preview(default_style, styles)
    return status, manage_preview, dashboard_preview, *refresh_styles_dropdowns(styles, prefs)


def delete_style(name: str) -> Tuple[str, str, str, Any, Any]:
    name = (name or "").strip()
    styles = load_styles()
    prefs = get_user_preferences()

    if name == "none":
        default_style = prefs.get("default_style", "none")
        preview = build_style_preview(default_style, styles)
        return "⚠️ The `none` style cannot be deleted.", "", preview, *refresh_styles_dropdowns(styles, prefs)

    if name in styles:
        styles.pop(name)
        status = "✅ Style deleted." if save_styles(styles) else "❌ Failed to modify styles."
    else:
        status = f"⚠️ Style `{name}` not found."

    if prefs.get("default_style") == name:
        prefs["default_style"] = "none"
        save_user_preferences(prefs)

    default_style = prefs.get("default_style", "none")
    manage_preview = build_style_preview(default_style, styles)
    dashboard_preview = build_style_preview(default_style, styles)
    return status, manage_preview, dashboard_preview, *refresh_styles_dropdowns(styles, prefs)


def reset_styles_preview(style_name: str) -> str:
    return build_style_preview(style_name)


def styles_as_table() -> List[List[str]]:
    styles = load_styles()
    table = [["style", "pre", "post"]]
    for name, info in styles.items():
        table.append([name, info.get("pre", ""), info.get("post", "")])
    return table[1:]


def save_server_configuration(
    comfy_url: str,
    timeout: int,
    port: int,
    network_accessible: bool,
) -> Tuple[str, str]:
    cfg = load_config()
    cfg.update(
        {
            "comfyui_url": comfy_url.strip() or cfg.get("comfyui_url", "http://127.0.0.1:8188"),
            "timeout": int(timeout),
            "web_port": int(port),
            "network_access": bool(network_accessible),
        }
    )
    if save_config(cfg):
        status = "✅ Configuration saved. Restart Gradio/Flask to apply network changes."
    else:
        status = "❌ Failed to write configuration."
    return status, build_api_markdown(cfg)


# Helper functions for UI event handlers
def apply_workflow_handler(selected: str):
    """Handle workflow application."""
    ok, message = set_active_workflow(selected)
    return gr.update(value=message, visible=True)


def load_placeholder_defaults_handler(selected: Optional[str]):
    """Load placeholder defaults for a workflow."""
    fields, notes = gather_placeholder_form_fields(selected)
    truncated_note: List[str] = []
    if len(fields) > PLACEHOLDER_MAX_FIELDS:
        truncated_note.append(f"⚠️ Showing first {PLACEHOLDER_MAX_FIELDS} placeholders.")
    fields = fields[:PLACEHOLDER_MAX_FIELDS]
    return build_placeholder_form_response(selected, fields, notes + truncated_note)


def make_placeholder_change_handler(index: int):
    """Create a placeholder value change handler for a specific index."""
    def _handler(value: Any, state: Dict[str, Any]):
        fields = state.get("fields", []) if isinstance(state, dict) else []
        updated = update_placeholder_field_value(fields, index, value)
        payload = {
            "filename": state.get("filename") if isinstance(state, dict) else None,
            "fields": updated,
        }
        return payload
    return _handler


def refresh_object_info_handler(force: bool, workflow_value: Optional[str], placeholder_state_value: Optional[Dict[str, Any]]):
    """Handle object info refresh."""
    target_workflow = workflow_value
    if not target_workflow and isinstance(placeholder_state_value, dict):
        target_workflow = placeholder_state_value.get("filename")

    data, notes, timestamp = refresh_object_info(force=force)
    status = object_info_status_message(timestamp, notes)
    placeholder_payload = load_placeholder_defaults_handler(target_workflow)
    return (
        {"timestamp": timestamp, "notes": notes, "data": data},
        gr.update(value=status),
        *placeholder_payload,
    )


def generate_image_handler(prompt: str, width: int, height: int, style: str):
    """Handle image generation."""
    return generate_image(prompt, width, height, style)


def load_workflow_handler(file_name: str):
    """Load workflow content for editor."""
    content, info = load_workflow_content(file_name)
    return info, content


def persist_workflow_handler(file_name: str, content: str):
    """Save workflow content."""
    ok, message = save_workflow_content(file_name, content)
    info, _ = load_workflow_content(file_name)
    return message, info


def build_dashboard_tab(config: Dict, styles: Dict, prefs: Dict, workflow_files: List[str],
                       active_workflow_name: Optional[str], workflow_warning: Optional[str],
                       initial_placeholder_fields: List[Dict], object_info_data: Dict,
                       object_info_notes: List[str], object_info_timestamp: float) -> Tuple[Any, ...]:
    """Build the Dashboard tab with workflow and style controls."""
    api_text = build_api_markdown(config)
    default_style = prefs.get("default_style", "none" if "none" in styles else (next(iter(styles)) if styles else "none"))

    api_markdown = gr.Markdown(api_text)
    gr.Markdown("Use the workflow and style controls below to tune defaults.")

    with gr.Row():
        workflow_dropdown = gr.Dropdown(
            choices=workflow_files,
            value=active_workflow_name,
            label="Active workflow",
        )
        apply_workflow_btn = gr.Button("Apply workflow", variant="primary")

    workflow_result = gr.Markdown(
        workflow_warning or "",
        visible=bool(workflow_warning),
    )

    apply_workflow_btn.click(
        apply_workflow_handler,
        inputs=workflow_dropdown,
        outputs=workflow_result,
    )

    gr.Markdown("### Default Style")
    with gr.Row():
        style_dropdown = gr.Dropdown(
            choices=list(styles.keys()),
            value=default_style,
            label="Default style",
        )
        style_status = gr.Markdown()

    style_preview = gr.Markdown(build_style_preview(default_style, styles))

    # Return components that need to be connected later
    return (api_markdown, workflow_dropdown, style_dropdown, style_status, style_preview)


def build_generate_tab(styles: Dict, prefs: Dict) -> Tuple[Any, ...]:
    """Build the Generate tab for image creation."""
    default_style = prefs.get("default_style", "none" if "none" in styles else (next(iter(styles)) if styles else "none"))

    gr.Markdown("Compose a prompt and send it through the Flask API via ComfyUI.")
    prompt_box = gr.Textbox(
        label="Prompt",
        value="a beautiful sunset over mountains",
        lines=4,
    )
    with gr.Row():
        width_slider = gr.Slider(256, 2048, value=640, step=64, label="Width")
        height_slider = gr.Slider(256, 2048, value=384, step=64, label="Height")
        gen_style_dropdown = gr.Dropdown(
            choices=list(styles.keys()),
            value=default_style,
            label="Style",
        )

    generate_button = gr.Button("Generate", variant="primary")
    result_image = gr.Image(label="Result", interactive=False)
    generation_status = gr.Markdown()

    generate_button.click(
        generate_image_handler,
        inputs=[prompt_box, width_slider, height_slider, gen_style_dropdown],
        outputs=[result_image, generation_status],
    )

    return (gen_style_dropdown,)


def build_monitor_tab() -> None:
    """Build the Monitor tab for system status."""
    gr.Markdown("Refresh to see service health and cache metrics.")
    refresh_button = gr.Button("Refresh stats", variant="primary")
    status_markdown = gr.Markdown()
    cache_summary = gr.Markdown()
    cache_table = gr.Markdown()
    config_snapshot = gr.Markdown()

    refresh_button.click(
        gather_monitoring_data,
        inputs=None,
        outputs=[status_markdown, cache_summary, cache_table, config_snapshot],
    )
    # populate immediately
    init_status, init_summary, init_table_md, init_config = gather_monitoring_data()
    status_markdown.value = init_status
    cache_summary.value = init_summary
    cache_table.value = init_table_md
    config_snapshot.value = init_config


def build_settings_tab(config: Dict, styles: Dict, workflow_files: List[str],
                      active_workflow_name: Optional[str]) -> None:
    """Build the Settings tab for configuration management."""
    gr.Markdown("Administration utilities for configuration, workflows, and styles.")

    # Server configuration section
    with gr.Accordion("Server configuration", open=True):
        comfyui_input = gr.Textbox(
            label="ComfyUI URL",
            value=config.get("comfyui_url", "http://127.0.0.1:8188"),
        )
        timeout_input = gr.Number(
            label="Timeout (seconds)",
            value=config.get("timeout", 30),
            precision=0,
        )
        web_port_input = gr.Number(
            label="Gradio port",
            value=config.get("web_port", 8501),
            precision=0,
        )
        network_toggle = gr.Checkbox(
            label="Allow other devices to access the UI",
            value=bool(config.get("network_access", False)),
        )
        save_config_btn = gr.Button("Save server configuration", variant="primary")
        config_status = gr.Markdown()

        # Note: Full implementation would continue here...
        # For brevity, I'm showing the structure


def build_interface() -> gr.Blocks:
    # Load all data once at the beginning
    config = load_config()
    styles = load_styles()
    prefs = get_user_preferences()
    workflow_files = get_workflow_files()

    # Setup initial state
    stored_workflow_path = config.get("workflow_path", "") or ""
    stored_workflow_name = os.path.basename(stored_workflow_path) if stored_workflow_path else None

    workflow_warning: Optional[str] = None
    if stored_workflow_name and stored_workflow_name not in workflow_files:
        workflow_warning = (
            f"⚠️ Stored workflow `{stored_workflow_name}` is missing. Select another workflow or upload a new one."
        )
        stored_workflow_name = None

    active_workflow_name = stored_workflow_name or (workflow_files[0] if workflow_files else None)

    if not workflow_files:
        workflow_warning = workflow_warning or "⚠️ No workflows found. Upload a workflow JSON to get started."

    # Prepare initial data for placeholders
    initial_config_fields, _ = parse_workflow_for_configuration(active_workflow_name) if active_workflow_name else ([], {})
    initial_placeholders = load_placeholders()
    object_info_data, object_info_notes, object_info_timestamp = refresh_object_info(force=True)
    object_info_status_initial = object_info_status_message(object_info_timestamp, object_info_notes)
    initial_placeholder_fields, initial_placeholder_notes = gather_placeholder_form_fields(active_workflow_name)
    if len(initial_placeholder_fields) > PLACEHOLDER_MAX_FIELDS:
        initial_placeholder_notes = list(initial_placeholder_notes) + [
            f"⚠️ Showing first {PLACEHOLDER_MAX_FIELDS} placeholders."
        ]
        initial_placeholder_fields = initial_placeholder_fields[:PLACEHOLDER_MAX_FIELDS]

    # Create initial placeholder message
    initial_placeholder_message = compose_placeholder_status_message(
        active_workflow_name,
        initial_placeholder_notes,
        bool(initial_placeholder_fields),
    )

    # Define default_style variable that's used throughout the interface
    default_style = prefs.get("default_style", "none" if "none" in styles else (next(iter(styles)) if styles else "none"))

    with gr.Blocks(title="Image Generation Server", theme=gr.themes.Soft()) as demo:
        # Header section
        gr.Markdown(
            """
            # 🎨 Image Generation Server
            **Gradio Management Console for ComfyUI Image Generation**
            """,
            elem_classes=["header-section"]
        )

        with gr.Tabs():
            # Dashboard Tab
            with gr.Tab("🏠 Dashboard"):
                # API Information section
                with gr.Accordion("📡 API Information", open=True):
                    api_text = build_api_markdown(config)
                    api_markdown = gr.Markdown(api_text)

                gr.Markdown("---")  # Visual separator

                # Configuration section
                gr.Markdown("## ⚙️ Configuration")
                gr.Markdown("Configure your active workflow and default style settings below.")

                # Workflow Configuration
                with gr.Group():
                    gr.Markdown("### 🔄 Workflow Settings")
                    with gr.Row():
                        with gr.Column(scale=3):
                            workflow_dropdown = gr.Dropdown(
                                choices=workflow_files,
                                value=active_workflow_name,
                                label="Active workflow",
                                info="Select the ComfyUI workflow to use for image generation"
                            )
                        with gr.Column(scale=1, min_width=120):
                            apply_workflow_btn = gr.Button(
                                "Apply Workflow",
                                variant="primary",
                                size="lg"
                            )

                    workflow_result = gr.Markdown(
                        workflow_warning or "",
                        visible=bool(workflow_warning),
                        elem_classes=["warning-message"]
                    )

                apply_workflow_btn.click(
                    apply_workflow_handler,
                    inputs=workflow_dropdown,
                    outputs=workflow_result,
                )

                gr.Markdown("---")  # Visual separator

                # Style Configuration
                with gr.Group():
                    gr.Markdown("### 🎨 Default Style Settings")
                    with gr.Row():
                        with gr.Column(scale=2):
                            style_dropdown = gr.Dropdown(
                                choices=list(styles.keys()),
                                value=default_style,
                                label="Default style",
                                info="Choose the default style for image generation"
                            )
                        with gr.Column(scale=1):
                            style_status = gr.Markdown()

                    with gr.Accordion("Style Preview", open=False):
                        style_preview = gr.Markdown(build_style_preview(default_style, styles))

                gr.Markdown("---")  # Visual separator

                # Advanced Placeholder Configuration
                with gr.Accordion("🔧 Advanced: Workflow Placeholder Defaults", open=False):
                    gr.Markdown(
                        """
                        **Configure default values for workflow placeholders**
                        This section allows you to set default values for placeholders in your workflow JSON files.
                        """
                    )

                    # ComfyUI Data Status
                    with gr.Group():
                        gr.Markdown("#### ComfyUI Object Info Status")
                        object_info_state = gr.State(
                            {
                                "timestamp": object_info_timestamp,
                                "notes": object_info_notes,
                                "data": object_info_data,
                            }
                        )
                        object_info_status = gr.Markdown(object_info_status_initial)

                        with gr.Row():
                            with gr.Column(scale=2):
                                object_info_refresh_btn = gr.Button(
                                    "🔄 Refresh ComfyUI Data",
                                    variant="secondary",
                                    size="sm"
                                )
                            with gr.Column(scale=3):
                                object_info_refresh_note = gr.Markdown(
                                    f"*Auto-refreshes every {max(1, OBJECT_INFO_TTL_SECONDS // 60)} minutes*"
                                )

                        object_info_timer = gr.Timer(
                            value=float(max(60, OBJECT_INFO_TTL_SECONDS)),
                            active=True,
                        )

                    # Placeholder Configuration
                    with gr.Group():
                        gr.Markdown("#### Placeholder Configuration")
                        placeholder_status_md = gr.Markdown(initial_placeholder_message)
                        placeholder_form_state = gr.State(
                            {
                                "filename": active_workflow_name,
                                "fields": initial_placeholder_fields,
                            }
                        )

                    placeholder_labels: List[gr.Markdown] = []
                    placeholder_text_inputs: List[gr.Textbox] = []
                    placeholder_number_inputs: List[gr.Number] = []
                    placeholder_checkbox_inputs: List[gr.Checkbox] = []
                    placeholder_dropdown_inputs: List[gr.Dropdown] = []

                    for idx in range(PLACEHOLDER_MAX_FIELDS):
                        field = initial_placeholder_fields[idx] if idx < len(initial_placeholder_fields) else None
                        component = field.get("component") if field else None
                        label_value = format_placeholder_label(field) if field else ""
                        label_visible = field is not None
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=2, min_width=200):
                                label = gr.Markdown(
                                    value=label_value,
                                    visible=label_visible,
                                    elem_classes=["placeholder-label"]
                                )
                            with gr.Column(scale=3, min_width=260):
                                text_input = gr.Textbox(
                                    value=field.get("value", "") if field and component in {"text", "textarea"} else "",
                                    visible=bool(field and component in {"text", "textarea"}),
                                    lines=int(field.get("lines", 1)) if field and component in {"text", "textarea"} else 1,
                                    show_label=False,
                                    interactive=True,
                                )
                                number_input = gr.Number(
                                    value=field.get("value") if field and component in {"int", "float"} else None,
                                    visible=bool(field and component in {"int", "float"}),
                                    precision=0 if component == "int" else (field.get("precision") if field else None),
                                    show_label=False,
                                    interactive=True,
                                    minimum=field.get("min") if field else None,
                                    maximum=field.get("max") if field else None,
                                    step=field.get("step") if field else None,
                                )
                                checkbox_input = gr.Checkbox(
                                    value=bool(field.get("value")) if field and component == "checkbox" else False,
                                    visible=bool(field and component == "checkbox"),
                                    show_label=False,
                                    interactive=True,
                                )
                                dropdown_input = gr.Dropdown(
                                    value=field.get("value", "") if field and component == "dropdown" else None,
                                    choices=field.get("options") if field and component == "dropdown" else [],
                                    visible=bool(field and component == "dropdown"),
                                    allow_custom_value=bool(field.get("allow_custom")) if field and component == "dropdown" else False,
                                    show_label=False,
                                    interactive=True,
                                )
                        placeholder_labels.append(label)
                        placeholder_text_inputs.append(text_input)
                        placeholder_number_inputs.append(number_input)
                        placeholder_checkbox_inputs.append(checkbox_input)
                        placeholder_dropdown_inputs.append(dropdown_input)

                    placeholder_component_outputs: List[Any] = []
                    for idx in range(PLACEHOLDER_MAX_FIELDS):
                        placeholder_component_outputs.extend(
                            [
                                placeholder_labels[idx],
                                placeholder_text_inputs[idx],
                                placeholder_number_inputs[idx],
                                placeholder_checkbox_inputs[idx],
                                placeholder_dropdown_inputs[idx],
                            ]
                        )

                    # Action buttons with better styling (outside the loop)
                    with gr.Row():
                        with gr.Column(scale=1):
                            placeholder_save_btn = gr.Button(
                                "💾 Save Defaults",
                                variant="primary",
                                size="lg"
                            )
                        with gr.Column(scale=1):
                            placeholder_reload_btn = gr.Button(
                                "🔄 Reload",
                                variant="secondary",
                                size="lg"
                            )

                    def load_placeholder_defaults(selected: Optional[str]):
                        fields, notes = gather_placeholder_form_fields(selected)
                        truncated_note: List[str] = []
                        if len(fields) > PLACEHOLDER_MAX_FIELDS:
                            truncated_note.append(
                                f"⚠️ Showing first {PLACEHOLDER_MAX_FIELDS} placeholders."
                            )
                        fields = fields[:PLACEHOLDER_MAX_FIELDS]
                        return build_placeholder_form_response(
                            selected,
                            fields,
                            notes + truncated_note,
                        )

                    def refresh_object_info_ui(
                        force: bool,
                        workflow_value: Optional[str],
                        placeholder_state_value: Optional[Dict[str, Any]],
                    ):
                        target_workflow = workflow_value
                        if not target_workflow and isinstance(placeholder_state_value, dict):
                            target_workflow = placeholder_state_value.get("filename")

                        data, notes, timestamp = refresh_object_info(force=force)
                        status = object_info_status_message(timestamp, notes)
                        placeholder_payload = load_placeholder_defaults(target_workflow)
                        return (
                            {
                                "timestamp": timestamp,
                                "notes": notes,
                                "data": data,
                            },
                            gr.update(value=status),
                            *placeholder_payload,
                        )

                    def manual_refresh_object_info(
                        workflow_value: Optional[str],
                        placeholder_state_value: Optional[Dict[str, Any]],
                    ):
                        return refresh_object_info_ui(True, workflow_value, placeholder_state_value)

                    def periodic_refresh_object_info(
                        workflow_value: Optional[str],
                        placeholder_state_value: Optional[Dict[str, Any]],
                    ):
                        return refresh_object_info_ui(False, workflow_value, placeholder_state_value)

                    object_info_refresh_btn.click(
                        manual_refresh_object_info,
                        inputs=[workflow_dropdown, placeholder_form_state],
                        outputs=[
                            object_info_state,
                            object_info_status,
                            placeholder_form_state,
                            placeholder_status_md,
                            *placeholder_component_outputs,
                        ],
                    )

                    object_info_timer.tick(
                        periodic_refresh_object_info,
                        inputs=[workflow_dropdown, placeholder_form_state],
                        outputs=[
                            object_info_state,
                            object_info_status,
                            placeholder_form_state,
                            placeholder_status_md,
                            *placeholder_component_outputs,
                        ],
                    )

                    def reload_placeholder_defaults(state: Dict[str, Any]):
                        filename = state.get("filename") if isinstance(state, dict) else None
                        return load_placeholder_defaults(filename)

                    def save_placeholder_defaults(state: Dict[str, Any]):
                        filename = state.get("filename") if isinstance(state, dict) else None
                        fields = state.get("fields", []) if isinstance(state, dict) else []
                        if not filename:
                            return build_placeholder_form_response(
                                filename,
                                fields,
                                ["⚠️ No workflow selected."],
                            )

                        existing_placeholders, existing_field_map = load_workflow_settings_full(filename)
                        new_placeholders = dict(existing_placeholders)
                        errors: List[str] = []

                        for field in fields:
                            name = field.get("name")
                            store, value, error = prepare_placeholder_for_save(field)
                            if error:
                                errors.append(error)
                                continue
                            if not name:
                                continue
                            if store:
                                new_placeholders[name] = value
                            else:
                                new_placeholders.pop(name, None)

                        if errors:
                            return build_placeholder_form_response(
                                filename,
                                fields,
                                ["❌ " + " | ".join(errors)],
                            )

                        try:
                            save_workflow_settings_full(
                                filename,
                                new_placeholders,
                                existing_field_map,
                            )
                        except OSError as exc:
                            return build_placeholder_form_response(
                                filename,
                                fields,
                                [f"❌ Failed to save: {exc}"],
                            )

                        refreshed_fields, notes = gather_placeholder_form_fields(filename)
                        truncated_note: List[str] = []
                        if len(refreshed_fields) > PLACEHOLDER_MAX_FIELDS:
                            truncated_note.append(
                                f"⚠️ Showing first {PLACEHOLDER_MAX_FIELDS} placeholders."
                            )
                        refreshed_fields = refreshed_fields[:PLACEHOLDER_MAX_FIELDS]
                        messages = ["✅ Placeholder defaults saved."] + notes + truncated_note
                        return build_placeholder_form_response(
                            filename,
                            refreshed_fields,
                            messages,
                        )

                    def make_placeholder_change_handler(index: int):
                        def _handler(value: Any, state: Dict[str, Any]):
                            fields = state.get("fields", []) if isinstance(state, dict) else []
                            updated = update_placeholder_field_value(fields, index, value)
                            payload = {
                                "filename": state.get("filename") if isinstance(state, dict) else None,
                                "fields": updated,
                            }
                            return payload

                        return _handler

                    for idx in range(PLACEHOLDER_MAX_FIELDS):
                        placeholder_text_inputs[idx].change(
                            make_placeholder_change_handler(idx),
                            inputs=[placeholder_text_inputs[idx], placeholder_form_state],
                            outputs=placeholder_form_state,
                        )
                        placeholder_number_inputs[idx].change(
                            make_placeholder_change_handler(idx),
                            inputs=[placeholder_number_inputs[idx], placeholder_form_state],
                            outputs=placeholder_form_state,
                        )
                        placeholder_checkbox_inputs[idx].change(
                            make_placeholder_change_handler(idx),
                            inputs=[placeholder_checkbox_inputs[idx], placeholder_form_state],
                            outputs=placeholder_form_state,
                        )
                        placeholder_dropdown_inputs[idx].change(
                            make_placeholder_change_handler(idx),
                            inputs=[placeholder_dropdown_inputs[idx], placeholder_form_state],
                            outputs=placeholder_form_state,
                        )

                    placeholder_save_btn.click(
                        save_placeholder_defaults,
                        inputs=placeholder_form_state,
                        outputs=[
                            placeholder_form_state,
                            placeholder_status_md,
                            *placeholder_component_outputs,
                        ],
                    )

                    placeholder_reload_btn.click(
                        reload_placeholder_defaults,
                        inputs=placeholder_form_state,
                        outputs=[
                            placeholder_form_state,
                            placeholder_status_md,
                            *placeholder_component_outputs,
                        ],
                    )

                workflow_dropdown.change(
                    load_placeholder_defaults,
                    inputs=workflow_dropdown,
                    outputs=[
                        placeholder_form_state,
                        placeholder_status_md,
                        *placeholder_component_outputs,
                    ],
                )

            # Generate tab
            with gr.Tab("🎨 Generate"):
                gr.Markdown(
                    """
                    ## 🚀 Image Generation
                    **Create images using the configured workflow and ComfyUI**
                    """
                )

                # Input Section
                with gr.Group():
                    gr.Markdown("### ✏️ Prompt")
                    prompt_box = gr.Textbox(
                        label="Your prompt",
                        value="a beautiful sunset over mountains",
                        lines=4,
                        placeholder="Describe the image you want to generate...",
                        info="Enter a detailed description of the image you want to create"
                    )

                # Generation Parameters
                with gr.Group():
                    gr.Markdown("### ⚙️ Generation Parameters")
                    with gr.Row():
                        with gr.Column():
                            width_slider = gr.Slider(
                                256, 2048,
                                value=640,
                                step=64,
                                label="Width",
                                info="Image width in pixels"
                            )
                        with gr.Column():
                            height_slider = gr.Slider(
                                256, 2048,
                                value=384,
                                step=64,
                                label="Height",
                                info="Image height in pixels"
                            )
                        with gr.Column():
                            gen_style_dropdown = gr.Dropdown(
                                choices=list(styles.keys()),
                                value=default_style,
                                label="Style",
                                info="Choose generation style"
                            )

                    # Generate Button
                    generate_button = gr.Button(
                        "🎨 Generate Image",
                        variant="primary",
                        size="lg",
                        scale=1
                    )

                # Results Section
                with gr.Group():
                    gr.Markdown("### 🖼️ Results")
                    generation_status = gr.Markdown()
                    result_image = gr.Image(
                        label="Generated Image",
                        interactive=False,
                        show_download_button=True
                    )

                generate_button.click(
                    generate_image_handler,
                    inputs=[prompt_box, width_slider, height_slider, gen_style_dropdown],
                    outputs=[result_image, generation_status],
                )

                style_change_event = style_dropdown.change(
                    update_default_style,
                    inputs=style_dropdown,
                    outputs=[style_status, style_preview],
                )

            # Monitor tab
            with gr.Tab("📊 Monitor"):
                gr.Markdown(
                    """
                    ## 📊 System Monitoring
                    **Monitor service health, cache metrics, and system status**
                    """
                )

                with gr.Group():
                    refresh_button = gr.Button(
                        "🔄 Refresh Statistics",
                        variant="primary",
                        size="lg"
                    )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🖥️ Service Status")
                        status_markdown = gr.Markdown()

                    with gr.Column():
                        gr.Markdown("### 💾 Cache Summary")
                        cache_summary = gr.Markdown()

                gr.Markdown("### 📋 Cache Details")
                cache_table = gr.Markdown()

                gr.Markdown("### ⚙️ Configuration")
                config_snapshot = gr.Markdown()

                refresh_button.click(
                    gather_monitoring_data,
                    inputs=None,
                    outputs=[status_markdown, cache_summary, cache_table, config_snapshot],
                )
                # populate immediately
                init_status, init_summary, init_table_md, init_config = gather_monitoring_data()
                status_markdown.value = init_status
                cache_summary.value = init_summary
                cache_table.value = init_table_md
                config_snapshot.value = init_config

            # Settings tab
            with gr.Tab("⚙️ Settings"):
                gr.Markdown(
                    """
                    ## ⚙️ Settings & Administration
                    **Configure server settings, manage workflows, and customize styles**
                    """
                )
                with gr.Accordion("Server configuration", open=True):
                    comfyui_input = gr.Textbox(
                        label="ComfyUI URL",
                        value=config.get("comfyui_url", "http://127.0.0.1:8188"),
                    )
                    timeout_input = gr.Number(
                        label="Timeout (seconds)",
                        value=config.get("timeout", 30),
                        precision=0,
                    )
                    web_port_input = gr.Number(
                        label="Gradio port",
                        value=config.get("web_port", 8501),
                        precision=0,
                    )
                    network_toggle = gr.Checkbox(
                        label="Allow other devices to access the UI",
                        value=bool(config.get("network_access", False)),
                    )
                    save_config_btn = gr.Button("Save server configuration", variant="primary")
                    config_status = gr.Markdown()

                    save_config_btn.click(
                        save_server_configuration,
                        inputs=[
                            comfyui_input,
                            timeout_input,
                            web_port_input,
                            network_toggle,
                        ],
                        outputs=[config_status, api_markdown],
                    )

                with gr.Accordion("Workflow management", open=False):
                    with gr.Row():
                        gr.Markdown("**Add Workflow:**")
                        workflow_upload = gr.File(
                            label="",
                            file_types=[".json"],
                            file_count="single",
                        )
                        gr.Markdown("**Delete Workflow:**")
                        delete_workflow_dropdown = gr.Dropdown(
                            choices=workflow_files,
                            value=active_workflow_name,
                            label="",
                        )
                        delete_workflow_button = gr.Button("Delete", variant="stop")

                    workflow_manage_status = gr.Markdown()

                    with gr.Accordion("Placeholder Management", open=False):
                        placeholder_status = gr.Markdown()
                        placeholder_table = gr.Dataframe(
                            value=placeholder_table_rows(),
                            headers=["Placeholder"],
                            interactive=False,
                        )
                        with gr.Row():
                            placeholder_add_input = gr.Textbox(label="New placeholder", scale=3)
                            placeholder_add_button = gr.Button("Add", variant="primary")
                        with gr.Row():
                            placeholder_delete_dropdown = gr.Dropdown(
                                choices=initial_placeholders,
                                label="Delete placeholder",
                                scale=3,
                            )
                            placeholder_delete_button = gr.Button("Delete", variant="stop")
                        placeholder_confirm_state = gr.State(True)

                    with gr.Accordion("Workflow Configuration", open=False):
                        with gr.Row():
                            workflow_config_dropdown = gr.Dropdown(
                                choices=workflow_files,
                                value=active_workflow_name,
                                label="Workflow",
                            )
                            workflow_config_load_btn = gr.Button("Load", variant="primary")
                        workflow_config_status = gr.Markdown()

                        with gr.Row():
                            gr.Markdown("**Order**")
                            gr.Markdown("**Node Title**")
                            gr.Markdown("**Input Field**")
                            gr.Markdown("**Value**")
                            gr.Markdown("**Placeholder**")

                        workflow_config_state = gr.State(
                            {
                                "filename": active_workflow_name,
                                "fields": [dict(field) for field in initial_config_fields],
                            }
                        )

                        config_order_inputs: List[gr.Number] = []
                        config_node_titles: List[gr.Textbox] = []
                        config_input_names: List[gr.Textbox] = []
                        config_values: List[gr.Textbox] = []
                        config_placeholders: List[gr.Dropdown] = []
                        placeholder_choices_initial = [""] + initial_placeholders

                        for idx in range(MAX_WORKFLOW_FIELDS):
                            field = initial_config_fields[idx] if idx < len(initial_config_fields) else None
                            visible = field is not None
                            is_primary = bool(field.get("is_primary")) if field else False
                            with gr.Row():
                                order_box = gr.Number(
                                    value=normalize_field_order(field.get("order"), idx + 1) if field else idx + 1,
                                    minimum=FIELD_ORDER_MIN,
                                    maximum=FIELD_ORDER_MAX,
                                    precision=0,
                                    step=1,
                                    interactive=is_primary,
                                    visible=visible and is_primary,
                                    show_label=False,
                                    scale=1,
                                )
                                node_box = gr.Textbox(
                                    value=(field.get("display_node_title") if field else ""),
                                    interactive=False,
                                    show_label=False,
                                    visible=visible,
                                    scale=4,
                                )
                                input_box = gr.Textbox(
                                    value=field["input_name"] if field else "",
                                    interactive=False,
                                    show_label=False,
                                    visible=visible,
                                    scale=2,
                                )
                                value_box = gr.Textbox(
                                    value=field["text_value"] if field else "",
                                    interactive=bool(field and field.get("placeholder", "") == ""),
                                    show_label=False,
                                    visible=visible,
                                    scale=3,
                                )
                                placeholder_box = gr.Dropdown(
                                    choices=placeholder_choices_initial,
                                    value=field["placeholder"] if field else "",
                                    show_label=False,
                                    visible=visible,
                                    scale=2,
                                )
                            config_node_titles.append(node_box)
                            config_order_inputs.append(order_box)
                            config_input_names.append(input_box)
                            config_values.append(value_box)
                            config_placeholders.append(placeholder_box)

                        config_component_outputs: List[Any] = []
                        for idx in range(MAX_WORKFLOW_FIELDS):
                            config_component_outputs.extend(
                                [
                                    config_order_inputs[idx],
                                    config_node_titles[idx],
                                    config_input_names[idx],
                                    config_values[idx],
                                    config_placeholders[idx],
                                ]
                            )

                        with gr.Row():
                            workflow_config_save_btn = gr.Button("Save", variant="primary")
                            workflow_config_cancel_btn = gr.Button("Cancel")
                            workflow_config_restore_btn = gr.Button("Restore", variant="secondary")

                        def _build_config_response(filename: Optional[str], fields: List[Dict[str, Any]], message: str):
                            safe_fields = [dict(item) for item in fields]
                            updates = build_workflow_config_updates(safe_fields)
                            payload = {
                                "filename": filename,
                                "fields": safe_fields,
                            }
                            return [payload, gr.update(value=message)] + updates

                        def _refresh_placeholder_dashboard(filename: Optional[str]) -> List[Any]:
                            fields, notes = gather_placeholder_form_fields(filename)
                            truncated_note: List[str] = []
                            if len(fields) > PLACEHOLDER_MAX_FIELDS:
                                truncated_note.append(
                                    f"⚠️ Showing first {PLACEHOLDER_MAX_FIELDS} placeholders."
                                )
                                fields = fields[:PLACEHOLDER_MAX_FIELDS]
                            return build_placeholder_form_response(
                                filename,
                                fields,
                                notes + truncated_note,
                            )

                        def add_placeholder_entry(
                            name: str,
                            current_delete_value: Optional[str],
                            state: Dict[str, Any],
                            dashboard_state: Dict[str, Any],
                        ):
                            placeholders = load_placeholders()
                            trimmed = (name or "").strip()
                            message: str
                            new_input_value = name or ""

                            if not trimmed:
                                message = "⚠️ Enter a placeholder name."
                            elif trimmed in placeholders:
                                message = f"⚠️ Placeholder `{trimmed}` already exists."
                            else:
                                placeholders.append(trimmed)
                                placeholders = sorted(dict.fromkeys(placeholders), key=str.lower)
                                save_placeholders_list(placeholders)
                                message = f"✅ Added `{trimmed}`."
                                new_input_value = ""
                                current_delete_value = trimmed

                            state_dict = state if isinstance(state, dict) else {"filename": None, "fields": []}
                            config_response = _build_config_response(
                                state_dict.get("filename"),
                                state_dict.get("fields", []),
                                message,
                            )

                            dashboard_payload = dashboard_state if isinstance(dashboard_state, dict) else {
                                "filename": None,
                                "fields": [],
                            }
                            dashboard_updates = _refresh_placeholder_dashboard(
                                dashboard_payload.get("filename")
                            )

                            delete_value = current_delete_value if current_delete_value in placeholders else (placeholders[0] if placeholders else None)

                            return (
                                message,
                                gr.update(value=placeholder_table_rows()),
                                gr.update(choices=placeholders, value=delete_value),
                                gr.update(value=new_input_value),
                                *config_response,
                                *dashboard_updates,
                            )

                        def delete_placeholder_entry(
                            selection: Any,
                            state: Optional[Dict[str, Any]] = None,
                            confirmed: Optional[bool] = True,
                            dashboard_state: Optional[Dict[str, Any]] = None,
                        ):
                            placeholders = load_placeholders()

                            selected_name: Optional[str]

                            if isinstance(selection, dict):
                                confirmed = selection.get("confirmed", confirmed)
                                selected_name = selection.get("name")
                                state_payload = selection.get("state", state)
                            else:
                                selected_name = selection
                                state_payload = state

                            state_dict = state_payload if isinstance(state_payload, dict) else {"filename": None, "fields": []}
                            dashboard_payload = dashboard_state if isinstance(dashboard_state, dict) else {
                                "filename": None,
                                "fields": [],
                            }

                            if not bool(confirmed):
                                config_response = _build_config_response(
                                    state_dict.get("filename"),
                                    state_dict.get("fields", []),
                                    "Placeholder deletion cancelled.",
                                )
                                delete_value = selection if selection in placeholders else (placeholders[0] if placeholders else None)
                                dashboard_updates = _refresh_placeholder_dashboard(
                                    dashboard_payload.get("filename")
                                )
                                return (
                                    "Deletion cancelled.",
                                    gr.update(value=placeholder_table_rows()),
                                    gr.update(choices=placeholders, value=delete_value),
                                    gr.update(),
                                    *config_response,
                                    *dashboard_updates,
                                )

                            selected = (selected_name or "").strip()
                            if not selected:
                                config_response = _build_config_response(
                                    state_dict.get("filename"),
                                    state_dict.get("fields", []),
                                    "⚠️ Select a placeholder to delete.",
                                )
                                delete_value = placeholders[0] if placeholders else None
                                dashboard_updates = _refresh_placeholder_dashboard(
                                    dashboard_payload.get("filename")
                                )
                                return (
                                    "⚠️ Select a placeholder to delete.",
                                    gr.update(value=placeholder_table_rows()),
                                    gr.update(choices=placeholders, value=delete_value),
                                    gr.update(),
                                    *config_response,
                                    *dashboard_updates,
                                )

                            if selected not in placeholders:
                                config_response = _build_config_response(
                                    state_dict.get("filename"),
                                    state_dict.get("fields", []),
                                    f"⚠️ Placeholder `{selected}` not found.",
                                )
                                delete_value = placeholders[0] if placeholders else None
                                dashboard_updates = _refresh_placeholder_dashboard(
                                    dashboard_payload.get("filename")
                                )
                                return (
                                    f"⚠️ Placeholder `{selected}` not found.",
                                    gr.update(value=placeholder_table_rows()),
                                    gr.update(choices=placeholders, value=delete_value),
                                    gr.update(),
                                    *config_response,
                                    *dashboard_updates,
                                )

                            placeholders = [p for p in placeholders if p != selected]
                            save_placeholders_list(placeholders)
                            message = f"✅ Deleted `{selected}`."

                            fields = state_dict.get("fields", [])
                            updated_fields: List[Dict[str, Any]] = []
                            for field in fields:
                                field_copy = dict(field)
                                if field_copy.get("placeholder") == selected:
                                    fallback = field_copy.get("stored_value", "")
                                    field_copy["placeholder"] = ""
                                    field_copy["text_value"] = str(fallback) if fallback is not None else ""
                                updated_fields.append(field_copy)

                            config_response = _build_config_response(
                                state_dict.get("filename"),
                                updated_fields,
                                message,
                            )
                            delete_value = placeholders[0] if placeholders else None
                            dashboard_updates = _refresh_placeholder_dashboard(
                                dashboard_payload.get("filename")
                            )

                            return (
                                message,
                                gr.update(value=placeholder_table_rows()),
                                gr.update(choices=placeholders, value=delete_value),
                                gr.update(),
                                *config_response,
                                *dashboard_updates,
                            )

                        def load_workflow_configuration(selected: str, state: Dict[str, Any]):
                            if not selected:
                                existing_fields = state.get("fields", []) if isinstance(state, dict) else []
                                filename = state.get("filename") if isinstance(state, dict) else None
                                return _build_config_response(filename, existing_fields, "⚠️ Select a workflow to load.")
                            fields, _ = parse_workflow_for_configuration(selected)
                            return _build_config_response(selected, fields, f"Loaded `{selected}`." )

                        def cancel_workflow_configuration(state: Dict[str, Any]):
                            filename = state.get("filename") if isinstance(state, dict) else None
                            if not filename:
                                existing_fields = state.get("fields", []) if isinstance(state, dict) else []
                                return _build_config_response(filename, existing_fields, "⚠️ No workflow selected.")
                            fields, _ = parse_workflow_for_configuration(filename)
                            return _build_config_response(filename, fields, "Changes reverted.")

                        def restore_workflow_configuration(payload: Any):
                            if isinstance(payload, dict) and "confirmed" in payload:
                                confirmed = bool(payload.get("confirmed"))
                                state = payload.get("state")
                            else:
                                confirmed = True
                                state = payload

                            if not isinstance(state, dict):
                                state = {"filename": None, "fields": []}

                            if not confirmed:
                                return _build_config_response(state.get("filename"), state.get("fields", []), "Restore cancelled.")

                            filename = state.get("filename")
                            if not filename:
                                return _build_config_response(filename, state.get("fields", []), "⚠️ No workflow selected.")

                            original = workflow_original_path(filename)
                            target = os.path.join(WORKFLOW_DIR, filename)
                            if not os.path.exists(original):
                                return _build_config_response(filename, state.get("fields", []), "⚠️ Original copy not found.")
                            ensure_directory(WORKFLOW_DIR)
                            shutil.copy2(original, target)
                            fields, _ = parse_workflow_for_configuration(filename)
                            return _build_config_response(filename, fields, f"Restored `{filename}` from backup.")

                        def save_workflow_configuration(
                            state: Dict[str, Any], dashboard_state: Optional[Dict[str, Any]] = None
                        ):
                            filename = state.get("filename") if isinstance(state, dict) else None
                            raw_fields = state.get("fields", []) if isinstance(state, dict) else []
                            fields: List[Dict[str, Any]] = []
                            node_orders: Dict[str, int] = {}
                            node_index = 0
                            for item in raw_fields:
                                normalized = dict(item)
                                node_id = normalized.get("node_id")
                                if node_id not in node_orders:
                                    node_index += 1
                                    node_orders[node_id] = normalize_field_order(
                                        normalized.get("order"), node_index
                                    )
                                normalized["order"] = node_orders.get(node_id, node_index)
                                fields.append(normalized)
                            dashboard_payload = dashboard_state if isinstance(dashboard_state, dict) else {
                                "filename": None,
                                "fields": [],
                            }
                            if not filename:
                                response = _build_config_response(filename, fields, "⚠️ No workflow selected.")
                                dashboard_updates = _refresh_placeholder_dashboard(
                                    dashboard_payload.get("filename")
                                )
                                return response + dashboard_updates
                            try:
                                workflow_data, _ = load_workflow_file(filename)
                            except FileNotFoundError:
                                response = _build_config_response(
                                    filename, fields, f"❌ Workflow `{filename}` not found."
                                )
                                dashboard_updates = _refresh_placeholder_dashboard(
                                    dashboard_payload.get("filename")
                                )
                                return response + dashboard_updates

                            node_order_values = [value for value in node_orders.values() if value is not None]
                            if len(node_order_values) != len(set(node_order_values)):
                                try:
                                    gr.Warning("Duplicate row numbers detected. Changes not saved.")
                                except Exception:
                                    pass
                                response = _build_config_response(
                                    filename,
                                    fields,
                                    "❌ Duplicate row numbers detected. Changes not saved.",
                                )
                                dashboard_updates = _refresh_placeholder_dashboard(
                                    dashboard_payload.get("filename")
                                )
                                return response + dashboard_updates

                            indexed_fields = [
                                (idx, dict(item)) for idx, item in enumerate(fields)
                            ]
                            indexed_fields.sort(
                                key=lambda entry: (
                                    entry[1].get("order", FIELD_ORDER_MAX + 1),
                                    entry[0],
                                )
                            )
                            fields = []
                            for idx, (_, field_item) in enumerate(indexed_fields, start=1):
                                field_item["order"] = normalize_field_order(field_item.get("order"), idx)
                                fields.append(field_item)

                            existing_placeholders_map, existing_field_map = load_workflow_settings_full(filename)

                            (
                                errors,
                                new_placeholders_map,
                                new_field_map,
                                updated_workflow,
                                active_placeholders,
                            ) = apply_fields_to_workflow(
                                filename, workflow_data, fields
                            )
                            if errors:
                                message = "❌ " + " | ".join(errors)
                                response = _build_config_response(filename, fields, message)
                                dashboard_updates = _refresh_placeholder_dashboard(
                                    dashboard_payload.get("filename")
                                )
                                return response + dashboard_updates

                            ensure_directory(WORKFLOW_DIR)
                            with open(os.path.join(WORKFLOW_DIR, filename), "w", encoding="utf-8") as fh:
                                json.dump(updated_workflow, fh, indent=2)

                            final_placeholders: Dict[str, Any] = {}
                            if isinstance(existing_placeholders_map, dict):
                                for key, value in existing_placeholders_map.items():
                                    if key in active_placeholders:
                                        final_placeholders[key] = value
                            for key, value in new_placeholders_map.items():
                                final_placeholders[key] = value

                            final_field_map: Dict[str, Any] = {}
                            existing_field_map_lookup = existing_field_map if isinstance(existing_field_map, dict) else {}

                            for idx, field in enumerate(fields):
                                order_value = normalize_field_order(field.get("order"), idx + 1)
                                node_id = field.get("node_id")
                                input_name = field.get("input_name")
                                value_type = field.get("type", "str")
                                text_value = field.get("text_value", "")
                                placeholder_name = field.get("placeholder", "")
                                new_key = workflow_field_key(node_id, input_name, order_value)

                                if new_key in new_field_map:
                                    value_to_store = new_field_map[new_key]
                                else:
                                    _, existing_value = find_field_value(
                                        existing_field_map_lookup, node_id, input_name
                                    )
                                    if existing_value is not None:
                                        value_to_store = existing_value
                                    elif placeholder_name:
                                        value_to_store = final_placeholders.get(placeholder_name)
                                    else:
                                        typed_value, _ = convert_string_to_type(str(text_value), value_type)
                                        value_to_store = typed_value

                                final_field_map[new_key] = value_to_store

                            save_workflow_settings_full(filename, final_placeholders, final_field_map)

                            refreshed_fields, _ = parse_workflow_for_configuration(filename)
                            try:
                                gr.Info(f"Saved {filename}")
                            except Exception:
                                pass
                            response = _build_config_response(
                                filename, refreshed_fields, f"✅ Saved `{filename}`."
                            )
                            dashboard_updates = _refresh_placeholder_dashboard(
                                dashboard_payload.get("filename")
                            )
                            return response + dashboard_updates

                        def make_order_change_handler(index: int):
                            def _handler(new_value: Any, state: Dict[str, Any]):
                                fields = state.get("fields", []) if isinstance(state, dict) else []
                                fields = update_field_order(fields, index, new_value)
                                payload = {
                                    "filename": state.get("filename") if isinstance(state, dict) else None,
                                    "fields": fields,
                                }
                                return payload

                            return _handler

                        def make_value_change_handler(index: int):
                            def _handler(new_value: str, state: Dict[str, Any]):
                                fields = state.get("fields", []) if isinstance(state, dict) else []
                                fields = update_field_value(fields, index, new_value)
                                payload = {
                                    "filename": state.get("filename") if isinstance(state, dict) else None,
                                    "fields": fields,
                                }
                                return payload

                            return _handler

                        def make_placeholder_change_handler(index: int):
                            def _handler(selection: str, state: Dict[str, Any]):
                                fields = state.get("fields", []) if isinstance(state, dict) else []
                                fields, value, enable_editing = update_field_placeholder(fields, index, selection)
                                payload = {
                                    "filename": state.get("filename") if isinstance(state, dict) else None,
                                    "fields": fields,
                                }
                                return payload, gr.update(value=value, interactive=enable_editing, visible=True)

                            return _handler

                        for idx in range(MAX_WORKFLOW_FIELDS):
                            config_order_inputs[idx].change(
                                make_order_change_handler(idx),
                                inputs=[config_order_inputs[idx], workflow_config_state],
                                outputs=workflow_config_state,
                            )
                            config_values[idx].change(
                                make_value_change_handler(idx),
                                inputs=[config_values[idx], workflow_config_state],
                                outputs=workflow_config_state,
                            )
                            config_placeholders[idx].change(
                                make_placeholder_change_handler(idx),
                                inputs=[config_placeholders[idx], workflow_config_state],
                                outputs=[workflow_config_state, config_values[idx]],
                            )

                        workflow_config_load_btn.click(
                            load_workflow_configuration,
                            inputs=[workflow_config_dropdown, workflow_config_state],
                            outputs=[workflow_config_state, workflow_config_status, *config_component_outputs],
                        )

                        workflow_config_cancel_btn.click(
                            cancel_workflow_configuration,
                            inputs=workflow_config_state,
                            outputs=[workflow_config_state, workflow_config_status, *config_component_outputs],
                        )

                        workflow_config_restore_btn.click(
                            restore_workflow_configuration,
                            inputs=workflow_config_state,
                            outputs=[workflow_config_state, workflow_config_status, *config_component_outputs],
                            js="(state) => ({state, confirmed: confirm('Restore original workflow? This overwrites current edits.')})",
                        )

                        workflow_config_save_btn.click(
                            save_workflow_configuration,
                            inputs=[workflow_config_state, placeholder_form_state],
                            outputs=[
                                workflow_config_state,
                                workflow_config_status,
                                *config_component_outputs,
                                placeholder_form_state,
                                placeholder_status_md,
                                *placeholder_component_outputs,
                            ],
                        )

                        placeholder_add_button.click(
                            add_placeholder_entry,
                            inputs=[
                                placeholder_add_input,
                                placeholder_delete_dropdown,
                                workflow_config_state,
                                placeholder_form_state,
                            ],
                            outputs=[
                                placeholder_status,
                                placeholder_table,
                                placeholder_delete_dropdown,
                                placeholder_add_input,
                                workflow_config_state,
                                workflow_config_status,
                                *config_component_outputs,
                                placeholder_form_state,
                                placeholder_status_md,
                                *placeholder_component_outputs,
                            ],
                        )

                        placeholder_delete_button.click(
                            delete_placeholder_entry,
                            inputs=[
                                placeholder_delete_dropdown,
                                workflow_config_state,
                                placeholder_confirm_state,
                                placeholder_form_state,
                            ],
                            outputs=[
                                placeholder_status,
                                placeholder_table,
                                placeholder_delete_dropdown,
                                placeholder_add_input,
                                workflow_config_state,
                                workflow_config_status,
                                *config_component_outputs,
                                placeholder_form_state,
                                placeholder_status_md,
                                *placeholder_component_outputs,
                            ],
                            js="(name, state, flag, dashboard) => [name, state, confirm(`Delete placeholder '${name}'?`), dashboard]",
                        )

                    with gr.Accordion("Workflow Editor", open=True):
                        workflow_list = gr.Dropdown(
                            choices=workflow_files,
                            value=active_workflow_name,
                            label="Workflow file",
                        )
                        with gr.Row():
                            with gr.Column(scale=3, min_width=400):
                                workflow_editor = gr.Code(language="json")
                            with gr.Column(scale=1, min_width=200):
                                workflow_info = gr.Markdown()
                        workflow_save_btn = gr.Button("Save workflow", variant="primary")
                        workflow_save_status = gr.Markdown()

                        def load_workflow(file_name: str):
                            content, info = load_workflow_content(file_name)
                            return info, content

                        workflow_list.change(
                            load_workflow,
                            inputs=workflow_list,
                            outputs=[workflow_info, workflow_editor],
                        )

                        def persist_workflow(file_name: str, content: str):
                            ok, message = save_workflow_content(file_name, content)
                            info, _ = load_workflow_content(file_name)
                            return message, info

                        workflow_save_btn.click(
                            persist_workflow,
                            inputs=[workflow_list, workflow_editor],
                            outputs=[workflow_save_status, workflow_info],
                        )

                        def _initial_workflow() -> Tuple[str, str]:
                            selected = os.path.basename(config.get("workflow_path", ""))
                            if not selected and workflow_files:
                                selected = workflow_files[0]
                            if not selected:
                                return "No workflow selected", ""
                            info, content = load_workflow(selected)
                            return info, content

                        initial_info, initial_content = _initial_workflow()
                        workflow_info.value = initial_info
                        workflow_editor.value = initial_content

                        def handle_upload(
                            file_data,
                            current_editor_selection,
                            current_dashboard_selection,
                            current_delete_selection,
                            current_config_state,
                            current_placeholder_state,
                        ):
                            existing_state = current_config_state if isinstance(current_config_state, dict) else {
                                "filename": active_workflow_name,
                                "fields": initial_config_fields.copy(),
                            }
                            existing_fields = existing_state.get("fields", [])
                            existing_filename = existing_state.get("filename")

                            placeholder_dashboard_payload = (
                                current_placeholder_state
                                if isinstance(current_placeholder_state, dict)
                                else {"filename": None, "fields": []}
                            )

                            def compose_response(
                                message: str,
                                file_update,
                                editor_dropdown_update,
                                dashboard_dropdown_update,
                                delete_dropdown_update,
                                info_update,
                                editor_update,
                                config_message: str,
                                config_filename: Optional[str],
                                config_fields: List[Dict[str, Any]],
                            ):
                                files_current = get_workflow_files()
                                if config_filename and config_filename not in files_current and files_current:
                                    config_filename_value = files_current[0]
                                else:
                                    config_filename_value = config_filename
                                config_result = _build_config_response(config_filename_value, config_fields, config_message)
                                dashboard_target = getattr(dashboard_dropdown_update, "value", None)
                                if dashboard_target is None and isinstance(dashboard_dropdown_update, dict):
                                    dashboard_target = dashboard_dropdown_update.get("value")
                                if dashboard_target is None:
                                    dashboard_target = placeholder_dashboard_payload.get("filename")
                                placeholder_updates = _refresh_placeholder_dashboard(dashboard_target)
                                return (
                                    message,
                                    file_update,
                                    editor_dropdown_update,
                                    dashboard_dropdown_update,
                                    delete_dropdown_update,
                                    info_update,
                                    editor_update,
                                    config_result[0],
                                    config_result[1],
                                    gr.update(choices=files_current, value=config_filename_value),
                                    *config_result[2:],
                                    *placeholder_updates,
                                )

                            if file_data is None:
                                return compose_response(
                                    "⚠️ Please select a workflow JSON file.",
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    "",
                                    existing_filename,
                                    existing_fields,
                                )

                            filename = os.path.basename(file_data.name)
                            if not filename.lower().endswith(".json"):
                                return compose_response(
                                    "⚠️ Workflow files must be JSON.",
                                    gr.update(value=None),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    "⚠️ Workflow files must be JSON.",
                                    existing_filename,
                                    existing_fields,
                                )

                            try:
                                with open(file_data.name, "r", encoding="utf-8") as fh:
                                    content = fh.read()
                                json.loads(content)
                            except (OSError, json.JSONDecodeError) as exc:
                                return compose_response(
                                    f"❌ Failed to process upload: {exc}",
                                    gr.update(value=None),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    f"❌ Failed to process upload: {exc}",
                                    existing_filename,
                                    existing_fields,
                                )

                            ensure_directory(WORKFLOW_DIR)
                            target_path = os.path.join(WORKFLOW_DIR, filename)
                            try:
                                with open(target_path, "w", encoding="utf-8") as out:
                                    out.write(content)
                            except OSError as exc:
                                return compose_response(
                                    f"❌ Could not save workflow: {exc}",
                                    gr.update(value=None),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    f"❌ Could not save workflow: {exc}",
                                    existing_filename,
                                    existing_fields,
                                )

                            files = get_workflow_files()
                            new_editor_value = filename if filename in files else (files[0] if files else None)
                            new_dashboard_value = current_dashboard_selection if current_dashboard_selection in files else new_editor_value
                            new_delete_value = current_delete_selection if current_delete_selection in files else new_editor_value

                            info_text, editor_content = load_workflow(new_editor_value) if new_editor_value else ("No workflow selected", "")

                            config_fields, _ = parse_workflow_for_configuration(new_editor_value) if new_editor_value else ([], {})

                            settings_message = ""
                            if new_editor_value:
                                try:
                                    workflow_data_for_settings, _ = load_workflow_file(new_editor_value)
                                except FileNotFoundError:
                                    workflow_data_for_settings = None
                                if workflow_data_for_settings is not None:
                                    fields_for_save = [dict(item) for item in config_fields]
                                    existing_placeholders_map, existing_field_map = load_workflow_settings_full(new_editor_value)
                                    (
                                        errors_settings,
                                        new_placeholder_settings,
                                        new_field_map,
                                        _,
                                        active_placeholder_settings,
                                    ) = apply_fields_to_workflow(
                                        new_editor_value,
                                        workflow_data_for_settings,
                                        fields_for_save,
                                    )
                                    if errors_settings:
                                        settings_message = " Settings initialization failed: " + " | ".join(errors_settings)
                                    else:
                                        final_placeholder_settings: Dict[str, Any] = {}
                                        if isinstance(existing_placeholders_map, dict):
                                            for key, value in existing_placeholders_map.items():
                                                if key in active_placeholder_settings:
                                                    final_placeholder_settings[key] = value
                                        for key, value in new_placeholder_settings.items():
                                            final_placeholder_settings[key] = value

                                        existing_field_map_lookup = existing_field_map if isinstance(existing_field_map, dict) else {}
                                        final_field_map: Dict[str, Any] = {}
                                        for idx, field in enumerate(config_fields):
                                            order_value = normalize_field_order(field.get("order"), idx + 1)
                                            node_id = field.get("node_id")
                                            input_name = field.get("input_name")
                                            value_type = field.get("type", "str")
                                            text_value = field.get("text_value", "")
                                            placeholder_name = field.get("placeholder", "")
                                            new_key = workflow_field_key(node_id, input_name, order_value)

                                            if new_key in new_field_map:
                                                value_to_store = new_field_map[new_key]
                                            else:
                                                _, existing_value = find_field_value(
                                                    existing_field_map_lookup, node_id, input_name
                                                )
                                                if existing_value is not None:
                                                    value_to_store = existing_value
                                                elif placeholder_name:
                                                    value_to_store = final_placeholder_settings.get(placeholder_name)
                                                else:
                                                    typed_value, _ = convert_string_to_type(
                                                        str(text_value), value_type
                                                    )
                                                    value_to_store = typed_value

                                            final_field_map[new_key] = value_to_store

                                        save_workflow_settings_full(
                                            new_editor_value,
                                            final_placeholder_settings,
                                            final_field_map,
                                        )
                                        config_fields, _ = parse_workflow_for_configuration(new_editor_value)

                            upload_message = f"✅ Uploaded `{filename}`."
                            if settings_message:
                                upload_message += settings_message

                            return compose_response(
                                upload_message,
                                gr.update(value=None),
                                gr.update(choices=files, value=new_editor_value),
                                gr.update(choices=files, value=new_dashboard_value),
                                gr.update(choices=files, value=new_delete_value),
                                gr.update(value=info_text),
                                gr.update(value=editor_content),
                                f"✅ Uploaded `{filename}`.",
                                new_editor_value,
                                config_fields,
                            )

                        workflow_upload.upload(
                            handle_upload,
                            inputs=[
                                workflow_upload,
                                workflow_list,
                                workflow_dropdown,
                                delete_workflow_dropdown,
                                workflow_config_state,
                                placeholder_form_state,
                            ],
                            outputs=[
                                workflow_manage_status,
                                workflow_upload,
                                workflow_list,
                                workflow_dropdown,
                                delete_workflow_dropdown,
                                workflow_info,
                                workflow_editor,
                                workflow_config_state,
                                workflow_config_status,
                                workflow_config_dropdown,
                                *config_component_outputs,
                                placeholder_form_state,
                                placeholder_status_md,
                                *placeholder_component_outputs,
                            ],
                        )

                        def handle_delete(
                            filename,
                            current_editor_selection,
                            current_dashboard_selection,
                            current_config_state,
                            current_placeholder_state,
                        ):
                            existing_state = current_config_state if isinstance(current_config_state, dict) else {
                                "filename": active_workflow_name,
                                "fields": initial_config_fields.copy(),
                            }
                            existing_fields = existing_state.get("fields", [])
                            existing_filename = existing_state.get("filename")

                            placeholder_dashboard_payload = (
                                current_placeholder_state
                                if isinstance(current_placeholder_state, dict)
                                else {"filename": None, "fields": []}
                            )

                            def compose_response(
                                message: str,
                                editor_dropdown_update,
                                dashboard_dropdown_update,
                                delete_dropdown_update,
                                info_update,
                                editor_update,
                                config_message: str,
                                config_filename: Optional[str],
                                config_fields: List[Dict[str, Any]],
                            ):
                                files_current = get_workflow_files()
                                if config_filename and config_filename not in files_current and files_current:
                                    config_filename_value = files_current[0]
                                else:
                                    config_filename_value = config_filename
                                config_result = _build_config_response(config_filename_value, config_fields, config_message)
                                dashboard_target = getattr(dashboard_dropdown_update, "value", None)
                                if dashboard_target is None and isinstance(dashboard_dropdown_update, dict):
                                    dashboard_target = dashboard_dropdown_update.get("value")
                                if dashboard_target is None:
                                    dashboard_target = placeholder_dashboard_payload.get("filename")
                                placeholder_updates = _refresh_placeholder_dashboard(dashboard_target)
                                return (
                                    message,
                                    editor_dropdown_update,
                                    dashboard_dropdown_update,
                                    delete_dropdown_update,
                                    info_update,
                                    editor_update,
                                    config_result[0],
                                    config_result[1],
                                    gr.update(choices=files_current, value=config_filename_value),
                                    *config_result[2:],
                                    *placeholder_updates,
                                )

                            if not filename:
                                return compose_response(
                                    "Deletion cancelled.",
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    "Deletion cancelled.",
                                    existing_filename,
                                    existing_fields,
                                )

                            removed_paths = remove_associated_workflow_files(filename)
                            if not removed_paths:
                                message = f"⚠️ `{filename}` not found."
                            else:
                                message = f"✅ Deleted `{filename}`"
                                if len(removed_paths) > 1:
                                    message += f" (and {len(removed_paths) - 1} associated files)."

                            files = get_workflow_files()
                            new_editor_value = current_editor_selection if current_editor_selection in files else (files[0] if files else None)
                            new_dashboard_value = current_dashboard_selection if current_dashboard_selection in files else new_editor_value
                            new_delete_value = new_editor_value if new_editor_value in files else None

                            info_text, editor_content = load_workflow(new_editor_value) if new_editor_value else ("No workflow selected", "")

                            config_fields, _ = parse_workflow_for_configuration(new_editor_value) if new_editor_value else ([], {})

                            return compose_response(
                                message,
                                gr.update(choices=files, value=new_editor_value),
                                gr.update(choices=files, value=new_dashboard_value),
                                gr.update(choices=files, value=new_delete_value),
                                gr.update(value=info_text),
                                gr.update(value=editor_content),
                                message,
                                new_editor_value,
                                config_fields,
                            )

                        delete_workflow_button.click(
                            handle_delete,
                            inputs=[
                                delete_workflow_dropdown,
                                workflow_list,
                                workflow_dropdown,
                                workflow_config_state,
                                placeholder_form_state,
                            ],
                            outputs=[
                                workflow_manage_status,
                                workflow_list,
                                workflow_dropdown,
                                delete_workflow_dropdown,
                                workflow_info,
                                workflow_editor,
                                workflow_config_state,
                                workflow_config_status,
                                workflow_config_dropdown,
                                *config_component_outputs,
                                placeholder_form_state,
                                placeholder_status_md,
                                *placeholder_component_outputs,
                            ],
                            js="(selected, editorValue, dashboardValue, configState, dashboardForm) => { if (!selected) { alert('Select a workflow to delete.'); return [null, editorValue, dashboardValue, configState, dashboardForm]; } const ok = confirm(`Delete workflow '${selected}'? This will remove associated files.`); return ok ? [selected, editorValue, dashboardValue, configState, dashboardForm] : [null, editorValue, dashboardValue, configState, dashboardForm]; }",
                        )

                with gr.Accordion("Style management", open=False):
                    styles_table = gr.Dataframe(
                        value=styles_as_table(),
                        headers=["Style", "Pre", "Post"],
                        interactive=False,
                        wrap=True,
                    )
                    style_name_input = gr.Textbox(label="Style name")
                    pre_input = gr.Textbox(label="Pre-prompt", lines=2)
                    post_input = gr.Textbox(label="Post-prompt", lines=2)
                    with gr.Row():
                        add_btn = gr.Button("Add", variant="primary")
                        update_btn = gr.Button("Update")
                        delete_btn = gr.Button("Delete", variant="stop")
                    style_manage_status = gr.Markdown()
                    style_manage_preview = gr.Markdown()

                    def refresh_styles_table() -> List[List[str]]:
                        return styles_as_table()

                    add_btn.click(
                        add_style,
                        inputs=[style_name_input, pre_input, post_input],
                        outputs=[
                            style_manage_status,
                            style_manage_preview,
                            style_preview,
                            style_dropdown,
                            gen_style_dropdown,
                        ],
                    ).then(
                        refresh_styles_table,
                        inputs=None,
                        outputs=styles_table,
                    )

                    update_btn.click(
                        update_style,
                        inputs=[style_name_input, pre_input, post_input],
                        outputs=[
                            style_manage_status,
                            style_manage_preview,
                            style_preview,
                            style_dropdown,
                            gen_style_dropdown,
                        ],
                    ).then(
                        refresh_styles_table,
                        inputs=None,
                        outputs=styles_table,
                    )

                    delete_btn.click(
                        delete_style,
                        inputs=style_name_input,
                        outputs=[
                            style_manage_status,
                            style_manage_preview,
                            style_preview,
                            style_dropdown,
                            gen_style_dropdown,
                        ],
                    ).then(
                        refresh_styles_table,
                        inputs=None,
                        outputs=styles_table,
                    )

                    style_name_input.change(
                        reset_styles_preview,
                        inputs=style_name_input,
                        outputs=style_manage_preview,
                    )

                with gr.Accordion("System tools", open=False):
                    restart_button = gr.Button("Restart Flask API")
                    clear_cache_button = gr.Button("Clear cache")
                    tools_status = gr.Markdown()

                    restart_button.click(lambda: restart_flask(), outputs=tools_status)
                    clear_cache_button.click(lambda: clear_cache(), outputs=tools_status)

        return demo


def main() -> None:
    interface = build_interface()
    cfg = load_config()
    host = "0.0.0.0" if cfg.get("network_access") else "127.0.0.1"
    port = int(cfg.get("web_port", 8501))
    interface.queue().launch(server_name=host, server_port=port, inbrowser=False, show_error=True)


if __name__ == "__main__":
    main()
