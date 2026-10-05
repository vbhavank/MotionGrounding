#!/bin/bash
#SBATCH --job-name=sav_subset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/sav_subset_%j.out
#SBATCH --error=logs/sav_subset_%j.err
#
# Download only the SA-V (Segment Anything Video) videos referenced by STGR/Motion-o
# PLM rows (video_path = sav_XXXXXX.mp4). Shards are fetched one at a time; the
# needed members are extracted and the tar is deleted, so peak disk use is one
# shard plus the kept videos. Safe to re-run: finished shards are skipped.
#
# Usage:
#   bash scripts/download_sav_subset.sh LINKS_FILE OUT_DIR JSON [JSON ...]
#
#   LINKS_FILE  the "file_name  cdn_link" list from Meta's SA-V download page
#               (tab- or space-separated, header line allowed)
#   OUT_DIR     e.g. $OO3/videos/stgr/plm   -> videos land in OUT_DIR/videos/
#   JSON        STGR-SFT.json, STGR-RL.json, the Motion-o PLM json, ...
#
# Env: KEEP_TARS=1 keeps the shards; SHARDS="000 036 val test" restricts to those shards.
# sav_val.tar / sav_test.tar (JPEG frames, no mp4) are handled too: the needed
# frame folders are extracted and re-encoded to <id>.mp4 at their frame rate.

set -uo pipefail

LINKS=$1; OUT=$2; shift 2
WORK="$OUT/_sav_tmp"
mkdir -p "$OUT/videos" "$OUT/sav_annotations" "$WORK" logs

python3 - "$@" > "$WORK/needed_ids.txt" <<'EOF'
import json, re, sys
ids = set()
for f in sys.argv[1:]:
    for r in json.load(open(f)):
        m = re.search(r"(sav_\d+)\.mp4$", (r.get("video_path") or "").strip())
        if m:
            ids.add(m.group(1))
print("\n".join(sorted(ids)))
EOF
N_NEEDED=$(wc -l < "$WORK/needed_ids.txt")
echo "needed SA-V videos: $N_NEEDED"
[ "$N_NEEDED" -eq 0 ] && { echo "no sav_*.mp4 ids found in the given json files"; exit 1; }

# fixed strings matched against tar member names: train shards hold <id>.mp4,
# sav_val.tar / sav_test.tar hold JPEG frames in .../<id>/NNNNN.jpg
sed -e 's/$/.mp4/' "$WORK/needed_ids.txt" > "$WORK/patterns.txt"
sed -e 's/$/_manual.json/' "$WORK/needed_ids.txt" >> "$WORK/patterns.txt"
sed -e 's|^|/|' -e 's|$|/|' "$WORK/needed_ids.txt" >> "$WORK/patterns.txt"

# frames -> <id>.mp4 at the folder's frame rate (".../JPEGImages_24fps/<id>/"), default 24
frames_to_mp4() {
    python3 - "$1" "$2" <<'PY'
import glob, os, re, sys
import cv2
root, out = sys.argv[1], sys.argv[2]
dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)})
for d in dirs:
    vid = os.path.basename(d)
    if not re.fullmatch(r"sav_\d+", vid) or os.path.exists(os.path.join(out, vid + ".mp4")):
        continue
    m = re.search(r"(\d+)fps", d)
    fps = float(m.group(1)) if m else 24.0
    frames = sorted(glob.glob(os.path.join(d, "*.jpg")))
    h, w = cv2.imread(frames[0]).shape[:2]
    vw = cv2.VideoWriter(os.path.join(out, vid + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.imread(f))
    vw.release()
    print(f"  encoded {vid}: {len(frames)} frames at {fps:g} fps")
PY
}

url_of() { awk -v n="$1" '$1 == n {print $2}' "$LINKS"; }

MD5_URL=$(url_of sav_md5sum.chk)
if [ -n "$MD5_URL" ] && [ ! -s "$WORK/sav_md5sum.chk" ]; then
    curl -sSL --fail --retry 5 -o "$WORK/sav_md5sum.chk" "$MD5_URL" || rm -f "$WORK/sav_md5sum.chk"
fi

have() { ls "$OUT/videos" | grep -c '^sav_.*\.mp4$'; }

for name in $(awk '$1 ~ /^sav_([0-9][0-9][0-9]|val|test)\.tar$/ {print $1}' "$LINKS"); do
    shard=${name#sav_}; shard=${shard%.tar}
    if [ -n "${SHARDS:-}" ] && ! grep -qw "$shard" <<< "$SHARDS"; then continue; fi
    [ -f "$WORK/$name.done" ] && { echo "skip $name (done)"; continue; }
    if [ "$(have)" -ge "$N_NEEDED" ]; then echo "all needed videos present"; break; fi

    echo "=== $name  $(date)"
    if ! curl -L --fail --retry 5 --retry-delay 10 -C - -o "$WORK/$name" "$(url_of "$name")"; then
        echo "download failed: $name (link expired? re-run later)"; continue
    fi
    line=$(grep -E "^[0-9a-f]{32} +\*?$name\$" "$WORK/sav_md5sum.chk" 2>/dev/null || true)
    if [ -n "$line" ]; then
        (cd "$WORK" && echo "$line" | md5sum -c --quiet -) \
            || { echo "md5 mismatch: $name, deleting"; rm -f "$WORK/$name"; continue; }
    else
        echo "  (no md5 line for $name in sav_md5sum.chk; skipping verification)"
    fi

    tar -tf "$WORK/$name" > "$WORK/$name.members"
    grep -F -f "$WORK/patterns.txt" "$WORK/$name.members" > "$WORK/$name.keep" || true
    n_keep=$(wc -l < "$WORK/$name.keep")
    echo "  members $(wc -l < "$WORK/$name.members"), keeping $n_keep"
    if [ "$n_keep" -gt 0 ]; then
        grep -v '\.jpg$' "$WORK/$name.keep" | grep -v '/$' > "$WORK/$name.keep_files" || true
        grep '\.jpg$' "$WORK/$name.keep" > "$WORK/$name.keep_jpg" || true
        # -C must precede -T: tar applies -C only to member names listed after it
        if [ -s "$WORK/$name.keep_files" ]; then
            tar -xf "$WORK/$name" -C "$OUT/videos" --transform='s|.*/||' -T "$WORK/$name.keep_files"
            mv "$OUT"/videos/*_manual.json "$OUT/sav_annotations/" 2>/dev/null || true
        fi
        if [ -s "$WORK/$name.keep_jpg" ]; then
            mkdir -p "$WORK/frames"
            tar -xf "$WORK/$name" -C "$WORK/frames" -T "$WORK/$name.keep_jpg"
            frames_to_mp4 "$WORK/frames" "$OUT/videos"
            rm -rf "$WORK/frames"
        fi
    fi
    [ "${KEEP_TARS:-0}" = "1" ] || rm -f "$WORK/$name"
    touch "$WORK/$name.done"
    echo "  have $(have) / $N_NEEDED"
done

comm -23 "$WORK/needed_ids.txt" <(ls "$OUT/videos" | sed -n 's/\.mp4$//p' | sort) > "$WORK/missing_ids.txt"
echo "done: $(have) / $N_NEEDED videos in $OUT/videos; missing listed in $WORK/missing_ids.txt ($(wc -l < "$WORK/missing_ids.txt"))"
