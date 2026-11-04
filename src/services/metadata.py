"""Read AI declarations from XMP fields and C2PA assertion containers.

Declarations are unverified: C2PA signatures and trust chains are not validated.
Ordinary comments and image pixels are never searched for AI-label strings.
"""
import io
import struct
import xml.etree.ElementTree as ET

from PIL import Image

SOURCE_URI = "http://cv.iptc.org/newscodes/digitalsourcetype/"
SOURCE_TYPES = {"trainedAlgorithmicMedia": "generated", "compositeWithTrainedAlgorithmicMedia": "edited"}
IPTC_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
MAX_METADATA_BYTES = 4 * 1024 * 1024


def _source_type(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    for prefix in (SOURCE_URI, SOURCE_URI.replace("http:", "https:", 1)):
        if value.startswith(prefix):
            return SOURCE_TYPES.get(value[len(prefix):])
    return None


def _xmp_labels(packet):
    if isinstance(packet, str):
        packet = packet.encode("utf-8")
    if not packet or len(packet) > MAX_METADATA_BYTES or b"<!DOCTYPE" in packet or b"<!ENTITY" in packet:
        return set()
    try:
        tree = ET.fromstring(packet)
    except (ET.ParseError, ValueError):
        return set()
    labels = set()
    field = "{" + IPTC_NS + "}DigitalSourceType"
    for element in tree.iter():
        for value in ([element.text] if element.tag == field else []) + [element.attrib.get(field)]:
            label = _source_type(value)
            if label:
                labels.add(label)
    return labels


def _containers(data):
    """Only recognized image metadata containers, never arbitrary byte matches."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        pos = 8
        while pos + 12 <= len(data):
            size = int.from_bytes(data[pos:pos + 4], "big")
            end = pos + 12 + size
            if end > len(data):
                break
            if data[pos + 4:pos + 8] == b"caBX" and size <= MAX_METADATA_BYTES:
                yield data[pos + 8:pos + 8 + size]
            pos = end
    elif data.startswith(b"\xff\xd8"):
        pos, chunks = 2, {}
        while pos + 4 <= len(data):
            if data[pos] != 255:
                break
            marker = data[pos + 1]
            if marker in (0xDA, 0xD9):
                break
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                pos += 2
                continue
            size = int.from_bytes(data[pos + 2:pos + 4], "big")
            if size < 2 or pos + 2 + size > len(data):
                break
            payload = data[pos + 4:pos + 2 + size]
            if marker == 0xEB and payload.startswith(b"JP") and len(payload) >= 8:
                instance, sequence = payload[2:4], int.from_bytes(payload[4:8], "big")
                parts = chunks.setdefault(instance, {})
                parts[sequence] = b"" if sequence in parts else payload[8:]  # reject duplicate packets
            pos += 2 + size
        for parts in chunks.values():
            order = sorted(parts)
            start = order[0]
            if start not in (0, 1) or order != list(range(start, start + len(parts))) or sum(map(len, parts.values())) > MAX_METADATA_BYTES:
                continue
            first = parts[start]  # Some older Microsoft files number their first packet zero.
            if len(first) < 8 or any(part[:8] != first[:8] for part in parts.values()):
                continue
            # JPEG-XT repeats LBox/TBox in every continuation packet. Retain the
            # initial header only; see contentauth/c2pa-rs asset_handlers/jpeg_io.rs.
            box = first + b"".join(parts[k][8:] for k in order[1:])
            size = int.from_bytes(box[:4], "big")
            if size == 1:
                size = int.from_bytes(box[8:16], "big")
            if size in (0, len(box)):
                yield box
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        pos = 12
        while pos + 8 <= len(data):
            size = int.from_bytes(data[pos + 4:pos + 8], "little")
            end = pos + 8 + size
            if end > len(data):
                break
            if data[pos:pos + 4] == b"C2PA" and size <= MAX_METADATA_BYTES:
                yield data[pos + 8:end]
            pos = end + size % 2


def _boxes(data, depth=0):
    if depth > 16:
        return
    pos = 0
    while pos + 8 <= len(data):
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            if pos + 16 > len(data):
                return
            size, header = int.from_bytes(data[pos + 8:pos + 16], "big"), 16
        elif size == 0:
            size = len(data) - pos
        if size < header or pos + size > len(data):
            return
        payload = data[pos + header:pos + size]
        yield kind, payload
        if kind == b"jumb":
            yield from _boxes(payload, depth + 1)
        pos += size


def _cbor_labels(data):
    """Bounded CBOR reader: extract actual map fields named digitalSourceType."""
    labels, remaining = set(), [20000]

    def read(pos, depth=0):
        remaining[0] -= 1
        if depth > 32 or remaining[0] < 0 or pos >= len(data):
            raise ValueError("CBOR limit")
        first = data[pos]
        major, extra, pos = first >> 5, first & 31, pos + 1
        if extra < 24:
            size = extra
        elif extra in (24, 25, 26, 27):
            width = 1 << (extra - 24)
            if pos + width > len(data):
                raise ValueError("Truncated CBOR")
            size = int.from_bytes(data[pos:pos + width], "big")
            pos += width
        elif extra == 31 and major in (2, 3, 4, 5):
            size = None
        else:
            raise ValueError("Unsupported CBOR")
        if major in (0, 1, 7):
            return None, pos
        if major in (2, 3) and size is not None:
            if pos + size > len(data):
                raise ValueError("Truncated CBOR string")
            value = data[pos:pos + size].decode("utf-8", errors="replace") if major == 3 and size <= 4096 else None
            return value, pos + size
        if major == 6:
            return read(pos, depth + 1)
        if major not in (2, 3, 4, 5) or (size is not None and size > 20000):
            raise ValueError("CBOR limit")
        index = 0
        while size is None or index < size:
            if size is None and pos < len(data) and data[pos] == 255:
                return None, pos + 1
            key, pos = read(pos, depth + 1)
            if major == 5:
                value, pos = read(pos, depth + 1)
                label = _source_type(value) if key == "digitalSourceType" else None
                if label:
                    labels.add(label)
            index += 1
        return None, pos

    try:
        _, end = read(0)
        return labels if end == len(data) else set()
    except (ValueError, IndexError, RecursionError):
        return set()


def check_file(data):
    labels, c2pa = set(), False
    try:
        with Image.open(io.BytesIO(data)) as image:
            for field in ("xmp", "XML:com.adobe.xmp"):
                labels.update(_xmp_labels(image.info.get(field)))
    except (OSError, ValueError):
        pass
    for container in _containers(data):
        boxes = list(_boxes(container))
        declared = any(kind == b"jumd" and b"c2pa\x00" in payload for kind, payload in boxes)
        if declared:
            c2pa = True
            for kind, payload in boxes:
                if kind == b"cbor":
                    labels.update(_cbor_labels(payload))
    label = "edited" if "edited" in labels else "generated" if "generated" in labels else None
    return {"ai_label": label, "c2pa": c2pa, "verified": False}


def describe(result):
    if result["ai_label"] == "generated":
        return "The file declares it was made by AI (unverified source declaration)."
    if result["ai_label"] == "edited":
        return "The file declares it was partly made or changed by AI (unverified source declaration)."
    if result["c2pa"]:
        return "A content-credentials container is present; its signature has not been verified."
    return None
