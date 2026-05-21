"""Generate app icons for Tauri using PIL."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def _create_png(width: int, height: int) -> bytes:
    """Create a simple solid-blue PNG icon."""
    # RGBA pixels (blue gradient)
    pixels = []
    for y in range(height):
        row = []
        for x in range(width):
            r = int(40 + (x / width) * 40)
            g = int(100 + (y / height) * 60)
            b = int(200 + (x / width) * 55)
            a = 255
            row.extend([r, g, b, a])
        # Filter byte (0 = None) + pixel data
        pixels.append(b"\x00" + bytes(row))

    raw = b"".join(pixels)

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)

    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA
    # IDAT
    compressed = zlib.compress(raw)
    return sig + make_chunk(b"IHDR", ihdr) + make_chunk(b"IDAT", compressed) + make_chunk(b"IEND", b"")


def main() -> None:
    icons_dir = Path(__file__).parent.parent / "src-tauri" / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)

    sizes = {
        "32x32.png": 32,
        "128x128.png": 128,
        "128x128@2x.png": 256,
    }

    for name, size in sizes.items():
        path = icons_dir / name
        if not path.exists():
            path.write_bytes(_create_png(size, size))
            print(f"Created: {path} ({size}x{size})")

    # Create macOS .icns placeholder (minimal)
    # For production, use `npx @tauri-apps/cli icon` with a proper source image
    icns_path = icons_dir / "icon.icns"
    if not icns_path.exists():
        # Minimal valid icns with a single ic07 (128x128 PNG) entry
        png_128 = _create_png(128, 128)
        icon_entry = b"ic07" + struct.pack(">I", len(png_128) + 8) + png_128
        icns_data = b"icns" + struct.pack(">I", len(icon_entry) + 8) + icon_entry
        icns_path.write_bytes(icns_data)
        print(f"Created: {icns_path} (placeholder)")

    # Create Windows .ico placeholder
    # Minimal valid ICO with a single 32x32 entry
    ico_path = icons_dir / "icon.ico"
    if not ico_path.exists():
        png_32 = _create_png(32, 32)
        ico_data = (
            struct.pack("<HHH", 0, 1, 1)  # ICO header
            + struct.pack("<BBBBHHIH", 32, 32, 0, 0, 1, 32, len(png_32), 22)
            + png_32
        )
        ico_path.write_bytes(ico_data)
        print(f"Created: {ico_path} (placeholder)")

    print("All icons created.")


if __name__ == "__main__":
    main()
