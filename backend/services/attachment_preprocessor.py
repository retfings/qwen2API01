from __future__ import annotations

import base64
import copy
from dataclasses import dataclass, field
from typing import Any

from backend.runtime.attachment_types import NormalizedAttachment


@dataclass(slots=True)
class PreprocessedAttachments:
    payload: dict[str, Any]
    attachments: list[NormalizedAttachment] = field(default_factory=list)
    uploaded_file_ids: list[str] = field(default_factory=list)


async def preprocess_attachments(payload: dict[str, Any], file_store) -> PreprocessedAttachments:
    messages = payload.get("messages", [])
    has_inline_image = False
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") != "image_url":
                continue
            image_url = block.get("image_url") or {}
            if isinstance(image_url, dict) and str(image_url.get("url") or "").strip().startswith("data:"):
                has_inline_image = True
                break
        if has_inline_image:
            break

    if not has_inline_image:
        return PreprocessedAttachments(payload=payload)

    rewritten = copy.deepcopy(payload)
    attachments: list[NormalizedAttachment] = []
    uploaded_file_ids: list[str] = []

    for message in rewritten.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, block in enumerate(content):
            if block.get("type") != "image_url":
                continue
            image_url = block.get("image_url") or {}
            if not isinstance(image_url, dict):
                continue
            url = str(image_url.get("url") or "").strip()
            if not url.startswith("data:"):
                continue
            header, encoded = url.split(",", 1)
            content_type = header.split(";", 1)[0][5:] or "application/octet-stream"
            raw = base64.b64decode(encoded)
            result = await file_store.save_bytes("inline-image", content_type, raw, "vision")
            uploaded_file_ids.append(result["id"])
            attachments.append(
                NormalizedAttachment(
                    file_id=result["id"],
                    filename=result["filename"],
                    content_type=content_type,
                    source="inline",
                )
            )
            content[index] = {"type": "input_image", "file_id": result["id"], "mime_type": content_type}

    return PreprocessedAttachments(
        payload=rewritten,
        attachments=attachments,
        uploaded_file_ids=uploaded_file_ids,
    )
