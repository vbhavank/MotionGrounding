"""Extract selected files from a split zip (name.z01 ... name.zNN + name.zip) on the
Hugging Face Hub without downloading the archive.

The central directory (at the end of name.zip) gives each member's disk and
offset; the member's bytes are then read with HTTP range requests
(huggingface_hub.HfFileSystem), decompressed and CRC-checked. Only the needed
members are transferred.

VideoEspresso (206 GB, 31 parts) for the 832 videos used by STGR-RL:

    python scripts/fetch_split_zip_members.py \\
        --repo hshjerry0315/VideoEspresso_train_video --base VideoEspresso_train_video \\
        --json $J/STGR-RL.json --source videoespresso_train_video \\
        --dest $OO3/videos --plan          # sizes only, nothing written
    (same without --plan to extract; re-runs skip files already present)

Each member is written to DEST/<video_path> (the path the trainer joins with
videos/). ``--local_parts DIR`` reads already-downloaded parts instead of the Hub.
"""

import argparse
import json
import os
import re
import struct
import sys
import zlib

CHUNK = 16 << 20


# ----------------------------------------------------------------------------
# Byte access to the concatenated parts
# ----------------------------------------------------------------------------
class Parts:
    """Disk i of the split archive = parts[i]; offsets are relative to each disk."""

    def __init__(self, names, sizes, opener):
        self.names, self.sizes, self.opener = names, sizes, opener
        self.prefix = [0]
        for s in sizes:
            self.prefix.append(self.prefix[-1] + s)
        self._fh = {}

    def _f(self, i):
        if i not in self._fh:
            if len(self._fh) > 4:
                for k in list(self._fh)[:-2]:
                    self._fh.pop(k).close()
            self._fh[i] = self.opener(self.names[i])
        return self._fh[i]

    def read(self, pos, n):
        """n bytes from global position pos (may span parts)."""
        out = bytearray()
        while n > 0:
            i = max(k for k in range(len(self.sizes)) if self.prefix[k] <= pos)
            off = pos - self.prefix[i]
            take = min(n, self.sizes[i] - off)
            if take <= 0:
                raise EOFError(f"read past end of {self.names[i]}")
            f = self._f(i)
            f.seek(off)
            chunk = f.read(take)
            if len(chunk) != take:
                raise EOFError(f"short read from {self.names[i]}")
            out += chunk
            pos += take
            n -= take
        return bytes(out)

    def glob(self, disk, off):
        return self.prefix[disk] + off


# ----------------------------------------------------------------------------
# Zip structures
# ----------------------------------------------------------------------------
def central_directory(parts):
    last = len(parts.sizes) - 1
    tail_len = min(parts.sizes[last], 65557 + 20)
    tail_pos = parts.prefix[last] + parts.sizes[last] - tail_len
    tail = parts.read(tail_pos, tail_len)
    e = tail.rfind(b"PK\x05\x06")
    if e < 0:
        sys.exit("end of central directory not found in the last part")
    (_, disk_no, cd_disk, _, n_total, cd_size, cd_off, _) = struct.unpack("<IHHHHIIH", tail[e:e + 22])
    if 0xFFFF in (disk_no, cd_disk, n_total) or 0xFFFFFFFF in (cd_size, cd_off):
        loc = tail.rfind(b"PK\x06\x07", 0, e)
        _, z64_disk, z64_off, _ = struct.unpack("<IIQI", tail[loc:loc + 20])
        rec = parts.read(parts.glob(z64_disk, z64_off), 56)
        (sig, _, _, _, disk_no, cd_disk, _, n_total, cd_size, cd_off) = struct.unpack("<IQHHIIQQQQ", rec)
        assert sig == 0x06064B50, "bad zip64 end record"
    cd = parts.read(parts.glob(cd_disk, cd_off), cd_size)
    entries, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (_, _, _, flags, method, _, _, crc, csize, usize, nlen, xlen, clen, disk, _, _, loff) = \
            struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8" if flags & 0x800 else "cp437")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        q = 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 0x0001:  # zip64: only the fields that overflowed, in this order
                vals, r = extra[q + 4:q + 4 + hlen], 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", vals[r:r + 8])[0]; r += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", vals[r:r + 8])[0]; r += 8
                if loff == 0xFFFFFFFF:
                    loff = struct.unpack("<Q", vals[r:r + 8])[0]; r += 8
                if disk == 0xFFFF:
                    disk = struct.unpack("<I", vals[r:r + 4])[0]
            q += 4 + hlen
        entries.append({"name": name, "method": method, "crc": crc, "csize": csize, "usize": usize,
                        "disk": disk, "off": loff, "flags": flags})
        p += 46 + nlen + xlen + clen
    return entries


def extract(parts, ent, dst):
    start = parts.glob(ent["disk"], ent["off"])
    hdr = parts.read(start, 30)
    if hdr[:4] != b"PK\x03\x04":
        raise ValueError(f"no local header for {ent['name']}")
    nlen, xlen = struct.unpack("<HH", hdr[26:30])
    pos, left = start + 30 + nlen + xlen, ent["csize"]
    if ent["method"] not in (0, 8):
        raise ValueError(f"unsupported compression method {ent['method']} for {ent['name']}")
    if ent["flags"] & 1:
        raise ValueError(f"{ent['name']} is encrypted")
    dec = zlib.decompressobj(-15) if ent["method"] == 8 else None
    crc = 0
    tmp = dst + ".part"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(tmp, "wb") as out:
        while left > 0:
            buf = parts.read(pos, min(CHUNK, left))
            pos += len(buf)
            left -= len(buf)
            data = dec.decompress(buf) if dec else buf
            crc = zlib.crc32(data, crc)
            out.write(data)
        if dec:
            data = dec.flush()
            crc = zlib.crc32(data, crc)
            out.write(data)
    if crc & 0xFFFFFFFF != ent["crc"]:
        os.remove(tmp)
        raise ValueError(f"CRC mismatch for {ent['name']}")
    os.replace(tmp, dst)


# ----------------------------------------------------------------------------
def part_names(listing, base):
    zs = sorted((n for n in listing if re.fullmatch(re.escape(base) + r"\.z\d+", os.path.basename(n))),
                key=lambda n: int(n.rsplit(".z", 1)[1]))
    last = [n for n in listing if os.path.basename(n) == base + ".zip"]
    if not last:
        sys.exit(f"{base}.zip (last part) not found")
    return zs + last


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="HF dataset repo id")
    ap.add_argument("--base", required=True, help="archive base name (without .zNN/.zip)")
    ap.add_argument("--local_parts", default=None, help="directory with already-downloaded parts")
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--source", default=None, help="only rows with this source")
    ap.add_argument("--field", default="video_path")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--plan", action="store_true", help="print what would be fetched and stop")
    args = ap.parse_args()

    needed = set()
    for f in args.json:
        for r in json.load(open(f)):
            if (args.source is None or r.get("source") == args.source) and r.get(args.field):
                needed.add(r[args.field].strip().lstrip("/"))
    todo = {p for p in needed if not os.path.isfile(os.path.join(args.dest, p))}
    print(f"{len(needed)} files referenced, {len(needed) - len(todo)} already in {args.dest}")

    if args.local_parts:
        listing = {os.path.join(args.local_parts, n): os.path.getsize(os.path.join(args.local_parts, n))
                   for n in os.listdir(args.local_parts)}
        opener = lambda n: open(n, "rb")  # noqa: E731
    else:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        root = f"datasets/{args.repo}"
        listing = {i["name"]: i["size"] for i in fs.ls(root, detail=True) if i["type"] == "file"}
        opener = lambda n: fs.open(n, "rb", block_size=CHUNK)  # noqa: E731
    names = part_names(listing, args.base)
    parts = Parts(names, [listing[n] for n in names], opener)
    print(f"{len(names)} parts, {sum(parts.sizes) / 1e9:.1f} GB in total; reading the central directory ...")
    entries = central_directory(parts)
    print(f"{len(entries)} members in the archive")

    by_suffix = {}
    for e in entries:
        if e["name"].endswith("/"):
            continue
        segs = e["name"].split("/")
        for k in range(len(segs)):
            by_suffix.setdefault("/".join(segs[k:]), e)
    plan = {p: by_suffix.get(p) for p in sorted(todo)}
    missing = [p for p, e in plan.items() if e is None]
    found = {p: e for p, e in plan.items() if e is not None}
    gb = sum(e["csize"] for e in found.values()) / 1e9
    print(f"to fetch: {len(found)} files, {gb:.1f} GB; not in the archive: {len(missing)}")
    for p in missing[:10]:
        print(f"  missing: {p}")
    if args.plan or not found:
        return
    for i, (p, e) in enumerate(sorted(found.items(), key=lambda kv: parts.glob(kv[1]["disk"], kv[1]["off"])), 1):
        try:
            extract(parts, e, os.path.join(args.dest, p))
        except Exception as ex:  # keep going; a re-run retries what is still missing
            print(f"  failed {p}: {ex}")
            continue
        if i % 25 == 0 or i == len(found):
            print(f"  {i}/{len(found)} extracted", flush=True)


if __name__ == "__main__":
    main()
