# hrv_recorder/ble.py
from typing import List, Optional, Tuple, Union

from bleak import BleakScanner

HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HR_CHAR_UUID    = "00002a37-0000-1000-8000-00805f9b34fb"

def parse_rr_intervals(data: bytes) -> Tuple[List[float], Optional[int]]:
    """Parse BLE Heart Rate Measurement (0x2A37) -> ([RR_ms...], HR_bpm or None)."""
    if not data:
        return [], None
    flags = data[0]
    hr_16bit   = bool(flags & 0x01)
    rr_present = bool(flags & 0x10)

    idx = 1
    if hr_16bit:
        if len(data) < 3: return [], None
        hr = int.from_bytes(data[idx:idx+2], "little"); idx += 2
    else:
        if len(data) < 2: return [], None
        hr = data[idx]; idx += 1

    rrs: List[float] = []
    if rr_present:
        while idx + 1 < len(data):
            rr_1_1024 = int.from_bytes(data[idx:idx+2], "little")
            idx += 2
            rr_ms = rr_1_1024 * 1000.0 / 1024.0
            rrs.append(rr_ms)
    return rrs, int(hr) if hr is not None else None


async def find_device(name_hint: Optional[str] = "polar", timeout: float = 12.0) -> Optional[Union[str, "BLEDevice"]]:
    """
    Return the first BLE device whose name contains name_hint or advertises the HR service.
    Works across OSes with Bleak.
    """
    nh = (name_hint or "").lower()
    devices = await BleakScanner.discover(timeout=timeout)
    hits = []
    for d in devices:
        name = (getattr(d, "name", "") or "")
        if nh and nh in name.lower():
            hits.append(d); continue
        uuids = [u.lower() for u in (getattr(d, "metadata", {}).get("uuids") or [])]
        if HR_SERVICE_UUID.lower() in uuids:
            hits.append(d)
    return hits[0] if hits else None