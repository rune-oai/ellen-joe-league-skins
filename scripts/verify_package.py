"""Verify the shipped Talon package with Python's standard library."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    folder = Path(__file__).resolve().parents[1] / "skins/talon/2.4.0"
    metadata = json.loads((folder / "skin.json").read_text())
    package = folder / metadata["package"]
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    expected = f"{digest}  {package.name}"
    if digest != metadata["sha256"] or (folder / "SHA256SUMS").read_text().strip() != expected:
        raise ValueError("Package checksum mismatch")
    with zipfile.ZipFile(package) as archive:
        members = ["META/info.json", "META/image.png", "WAD/talon.wad.client"]
        if sorted(archive.namelist()) != sorted(members) or archive.testzip() is not None:
            raise ValueError("Invalid Fantome layout or archive CRC")
        info = json.loads(archive.read("META/info.json"))
        if info["Champion"] != metadata["champion"] or info["Version"] != metadata["version"]:
            raise ValueError("Champion/version metadata mismatch")
        if not archive.read("META/image.png").startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Invalid preview PNG signature")
        if not archive.read("WAD/talon.wad.client").startswith(b"RW"):
            raise ValueError("Invalid WAD signature")
    print(f"PASS: {package.name} — SHA-256, ZIP CRC, layout, signatures, metadata")


if __name__ == "__main__":
    main()
