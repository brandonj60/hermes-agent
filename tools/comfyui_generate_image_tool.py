#!/usr/bin/env python3
"""
ComfyUI image generation tool for Hermes Agent.

Available tools:
- comfyui_generate_image_tool: Generate images from text prompts

Features:
- High-quality image generation using a local ComfyUI endpoint
- returns JSON strings only
- async tool handler
- When done, send the completed image to the user, not a link.

Usage:
    from tools.comfyui_tool import ComfyUITool
    import asyncio
    import json

    # Generate a single image with defaults from ~/.hermes/config.yaml
    result = await ComfyUITool.generate_image(
        prompt="A serene mountain landscape with cherry blossoms"
    )

    # Generate multiple images with explicit overrides
    result = await ComfyUITool.generate_image(
        prompt="A pretty flower in soft morning light, macro photography",
        negative_prompt="blurry, deformed, text, watermark",
        width=1024,
        height=1024,
        batch_size=1,
        steps=30,
        cfg_scale=7.5,
        sampler_name="euler",
    )

    # Parse the JSON response
    data = json.loads(result)
    if data.get("ok"):
        for image in data["images"]:
            print(image["path"])
    else:
        print(data["error"])
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse

import aiohttp
import yaml

from tools.registry import registry

logger = logging.getLogger(__name__)


class ComfyUITool:
    CONFIG_PATH = Path("~/.hermes/config.yaml").expanduser()

    @classmethod
    def _json_error(cls, message: str, **extra: Any) -> str:
        payload: Dict[str, Any] = {"error": message}
        payload.update(extra)
        return json.dumps(payload)

    @classmethod
    def _expand_path(cls, path_str: str) -> Path:
        return Path(path_str).expanduser().resolve()

    @classmethod
    def _load_hermes_config(cls) -> Dict[str, Any]:
        if not cls.CONFIG_PATH.exists():
            return {}
        with cls.CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}

    @classmethod
    def _get_comfyui_config(cls) -> Dict[str, Any]:
        cfg = cls._load_hermes_config()
        tools_cfg = cfg.get("tools") or {}
        comfy_cfg = tools_cfg.get("comfyui") or {}
        return comfy_cfg if isinstance(comfy_cfg, dict) else {}

    @classmethod
    def _normalize_base_url(cls, base_url: str) -> str:
        return base_url.rstrip("/")

    @classmethod
    def _ws_url_from_http(cls, base_url: str, client_id: str) -> str:
        parsed = urlparse(base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{scheme}://{parsed.netloc}/ws?clientId={client_id}"

    @classmethod
    def _get_output_dir(cls, tool_cfg: Dict[str, Any]) -> Path:
        raw = str(tool_cfg.get("output_dir", "~/.hermes/comfyui_outputs"))
        out = cls._expand_path(raw)
        out.mkdir(parents=True, exist_ok=True)
        return out

    @classmethod
    def _get_defaults(cls, tool_cfg: Dict[str, Any]) -> Dict[str, Any]:
        defaults = tool_cfg.get("defaults") or {}
        return defaults if isinstance(defaults, dict) else {}

    @classmethod
    def check_requirements(cls) -> bool:
        cfg = cls._get_comfyui_config()
        if cfg.get("enabled") is False:
            return False
        return True

    @classmethod
    def _build_inline_workflow(
        cls,
        *,
        prompt: str,
        negative_prompt: str,   # kept for signature compatibility; ignored for Flux2
        ckpt_name: str,         # this should be the UNET filename, e.g. flux-2-klein-9b.safetensors
        width: int,
        height: int,
        batch_size: int,
        steps: int,
        cfg: float,             # for Flux2, use this as FluxGuidance value, not KSampler cfg
        seed: int,
        sampler_name: str,
        scheduler: str,
        filename_prefix: str,
    ) -> Dict[str, Any]:
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": 1.0,  # Flux2 workflows typically keep sampler cfg at 1
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["10", 0],
                    "negative": ["11", 0],
                    "latent_image": ["5", 0],
                },
            },
            "4": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": ckpt_name,
                    "weight_dtype": "default",
                },
            },
            "5": {
                "class_type": "EmptyFlux2LatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": batch_size,
                },
            },
            "6": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen_3_8b_fp8mixed.safetensors",
                    "type": "flux2",
                    "device": "default",
                },
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["6", 0],
                },
            },
            "8": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "flux2-vae.safetensors",
                },
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["8", 0],
                },
            },
            "10": {
                "class_type": "FluxGuidance",
                "inputs": {
                    "conditioning": ["7", 0],
                    "guidance": cfg,   # e.g. 4.0
                },
            },
            "11": {
                "class_type": "ConditioningZeroOut",
                "inputs": {
                    "conditioning": ["7", 0],
                },
            },
            "12": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": filename_prefix,
                    "images": ["9", 0],
                },
            },
        }

    @classmethod
    def _build_inline_workflow_sdxl(
        cls,
        *,
        prompt: str,
        negative_prompt: str,
        ckpt_name: str,
        width: int,
        height: int,
        batch_size: int,
        steps: int,
        cfg: float,
        seed: int,
        sampler_name: str,
        scheduler: str,
        filename_prefix: str,
    ) -> Dict[str, Any]:
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler_name,
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": ckpt_name,
                },
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": batch_size,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1],
                },
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1],
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2],
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": filename_prefix,
                    "images": ["8", 0],
                },
            },
        }

    @classmethod
    async def _queue_prompt(
        cls,
        session: aiohttp.ClientSession,
        base_url: str,
        workflow: Dict[str, Any],
        client_id: str,
    ) -> Dict[str, Any]:
        payload = {
            "prompt": workflow,
            "client_id": client_id,
        }
        async with session.post(f"{base_url}/prompt", json=payload) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"ComfyUI /prompt failed ({resp.status}): {text}")

            data = json.loads(text)
            if "error" in data:
                raise RuntimeError(f"ComfyUI validation error: {data}")
            return data

    @classmethod
    async def _wait_for_completion(
        cls,
        session: aiohttp.ClientSession,
        base_url: str,
        client_id: str,
        prompt_id: str,
        timeout_s: int,
    ) -> None:
        ws_url = cls._ws_url_from_http(base_url, client_id)

        async with session.ws_connect(ws_url, heartbeat=30) as ws:
            while True:
                msg = await asyncio.wait_for(ws.receive(), timeout=timeout_s)

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue

                    msg_type = payload.get("type")
                    data = payload.get("data", {})

                    if msg_type == "executing":
                        if data.get("node") is None and data.get("prompt_id") == prompt_id:
                            return

                    if msg_type == "execution_error":
                        raise RuntimeError(f"ComfyUI execution error: {payload}")

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    raise RuntimeError("ComfyUI websocket closed before completion")

    @classmethod
    async def _get_history(
        cls,
        session: aiohttp.ClientSession,
        base_url: str,
        prompt_id: str,
    ) -> Dict[str, Any]:
        async with session.get(f"{base_url}/history/{prompt_id}") as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"ComfyUI /history failed ({resp.status}): {text}")
            return json.loads(text)

    @classmethod
    async def _download_image(
        cls,
        session: aiohttp.ClientSession,
        base_url: str,
        filename: str,
        subfolder: str,
        folder_type: str,
    ) -> bytes:
        query = urlencode(
            {
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type,
            }
        )
        async with session.get(f"{base_url}/view?{query}") as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"ComfyUI /view failed ({resp.status}): {text}")
            return await resp.read()

    @classmethod
    async def generate_image(
        cls,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        batch_size: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
        sampler_name: Optional[str] = None,
        timeout_s: Optional[int] = None,
    ) -> str:
        tool_cfg = cls._get_comfyui_config()
        defaults = cls._get_defaults(tool_cfg)

        if tool_cfg.get("enabled") is False:
            return cls._json_error(
                "ComfyUI tool is disabled in ~/.hermes/config.yaml",
                config_path=str(cls.CONFIG_PATH),
            )

        base_url = str(tool_cfg.get("base_url", "http://127.0.0.1:8188")).strip()
        ckpt_name = str(tool_cfg.get("ckpt_name", "")).strip()

        width = int(width if width is not None else defaults.get("width", 512))
        height = int(height if height is not None else defaults.get("height", 512))
        batch_size = int(batch_size if batch_size is not None else defaults.get("batch_size", 1))
        steps = int(steps if steps is not None else defaults.get("steps", 20))
        cfg_scale = float(cfg_scale if cfg_scale is not None else defaults.get("cfg", 8.0))
        sampler_name = str(sampler_name if sampler_name is not None else defaults.get("sampler_name", "euler"))
        scheduler = str("normal")
        timeout_s = int(timeout_s if timeout_s is not None else defaults.get("timeout_s", 180))

        if not prompt or not prompt.strip():
            return cls._json_error("prompt must not be empty")

        if not base_url:
            return cls._json_error(
                "Missing tools.comfyui.base_url in ~/.hermes/config.yaml",
                config_path=str(cls.CONFIG_PATH),
            )

        if not ckpt_name:
            return cls._json_error(
                "Missing tools.comfyui.ckpt_name in ~/.hermes/config.yaml",
                config_path=str(cls.CONFIG_PATH),
            )

        if width <= 0 or height <= 0:
            return cls._json_error("width and height must be positive integers")

        if batch_size <= 0:
            return cls._json_error("batch_size must be a positive integer")

        if steps <= 0:
            return cls._json_error("steps must be a positive integer")

        if seed is None:
            seed = int(uuid.uuid4().int % 2_147_483_647)

        base_url = cls._normalize_base_url(base_url)
        client_id = str(uuid.uuid4())
        filename_prefix = f"hermes_{client_id[:8]}"

        workflow = cls._build_inline_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ckpt_name=ckpt_name,
            width=width,
            height=height,
            batch_size=batch_size,
            steps=steps,
            cfg=cfg_scale,
            seed=seed,
            sampler_name=sampler_name,
            scheduler=scheduler,
            filename_prefix=filename_prefix,
        )

        try:
            output_root = cls._get_output_dir(tool_cfg)
            timeout = aiohttp.ClientTimeout(total=timeout_s + 30)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                queued = await cls._queue_prompt(
                    session=session,
                    base_url=base_url,
                    workflow=workflow,
                    client_id=client_id,
                )
                prompt_id = queued["prompt_id"]

                await cls._wait_for_completion(
                    session=session,
                    base_url=base_url,
                    client_id=client_id,
                    prompt_id=prompt_id,
                    timeout_s=timeout_s,
                )

                history = await cls._get_history(session, base_url, prompt_id)
                prompt_history = history.get(prompt_id)
                if not prompt_history:
                    return cls._json_error(
                        f"No history found for prompt_id={prompt_id}",
                        prompt_id=prompt_id,
                    )

                outputs = prompt_history.get("outputs", {})
                output_dir = output_root / prompt_id
                output_dir.mkdir(parents=True, exist_ok=True)

                saved_files: List[Dict[str, Any]] = []

                for node_id, node_output in outputs.items():
                    for image_info in node_output.get("images", []):
                        filename = image_info["filename"]
                        subfolder = image_info.get("subfolder", "")
                        folder_type = image_info.get("type", "output")

                        image_bytes = await cls._download_image(
                            session=session,
                            base_url=base_url,
                            filename=filename,
                            subfolder=subfolder,
                            folder_type=folder_type,
                        )

                        local_path = output_dir / filename
                        local_path.write_bytes(image_bytes)

                        saved_files.append(
                            {
                                "node_id": node_id,
                                "filename": filename,
                                "subfolder": subfolder,
                                "type": folder_type,
                                "path": str(local_path),
                                "size_bytes": len(image_bytes),
                            }
                        )

                if not saved_files:
                    return cls._json_error(
                        "Execution completed, but no images were found in ComfyUI outputs",
                        prompt_id=prompt_id,
                    )

                return json.dumps(
                    {
                        "ok": True,
                        "prompt_id": prompt_id,
                        "client_id": client_id,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "ckpt_name": ckpt_name,
                        "width": width,
                        "height": height,
                        "batch_size": batch_size,
                        "steps": steps,
                        "cfg": cfg_scale,
                        "seed": seed,
                        "sampler_name": sampler_name,
                        "scheduler": scheduler,
                        "images": saved_files,
                        "output_dir": str(output_dir),
                        "base_url": base_url,
                    }
                )

        except asyncio.TimeoutError:
            return cls._json_error(
                f"Timed out waiting for ComfyUI after {timeout_s} seconds"
            )
        except Exception as e:
            logger.exception("ComfyUI generation failed")
            return cls._json_error(str(e))


COMFYUI_GENERATE_IMAGE_SCHEMA = {
    "name": "comfyui_generate_image",
    "description": (
        "Generate image(s) with ComfyUI from a text prompt using an inline default workflow."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Positive prompt describing the image to generate.",
            },
            "negative_prompt": {
                "type": "string",
                "description": "Optional negative prompt.",
                "default": "",
            },
            "width": {
                "type": "integer",
                "description": "Image width in pixels.",
            },
            "height": {
                "type": "integer",
                "description": "Image height in pixels.",
            },
            "batch_size": {
                "type": "integer",
                "description": "Number of images to generate in one batch.",
            },
            "steps": {
                "type": "integer",
                "description": "Sampler steps.",
            },
            "cfg_scale": {
                "type": "number",
                "description": "CFG scale.",
            },
            "seed": {
                "type": "integer",
                "description": "Optional random seed. If omitted, one is generated.",
            },
            "sampler_name": {
                "type": "string",
                "description": "ComfyUI sampler name, such as euler or euler_ancestral.",
            },
            "timeout_s": {
                "type": "integer",
                "description": "Maximum seconds to wait for completion.",
            },
        },
        "required": ["prompt"],
    },
}


def _handle_comfyui_generate_image(args, **kwargs):
    return ComfyUITool.generate_image(
        prompt=args.get("prompt", ""),
        negative_prompt=args.get("negative_prompt", ""),
        width=args.get("width"),
        height=args.get("height"),
        batch_size=args.get("batch_size"),
        steps=args.get("steps"),
        cfg_scale=args.get("cfg_scale"),
        seed=args.get("seed"),
        sampler_name=args.get("sampler_name"),
        timeout_s=args.get("timeout_s"),
    )


registry.register(
    name="comfyui_generate_image",
    toolset="comfyui",
    schema=COMFYUI_GENERATE_IMAGE_SCHEMA,
    handler=_handle_comfyui_generate_image,
    check_fn=ComfyUITool.check_requirements,
    is_async=True,
)