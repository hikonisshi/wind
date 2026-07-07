#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOAA GFS の地上10m風（UGRD/VGRD）を取得し、leaflet-velocity 形式の wind.json に変換する。
- GFS 0.25度・パブリックドメイン（米国政府作成＝商用可・帰属不要）。
- 日本域（沖縄・南西諸島・小笠原を含む）を切り出し、3時間ごと 0〜72時間先（25コマ）。
- 0.25度（GFS最高解像度）で出力。値は0.1 m/s精度に丸めてファイルを抑える。
- 依存: numpy, xarray, cfgrib（eccodes C ライブラリが必要 → Actions側で apt install）。
出力 wind.json:
{
  "run": "2026070800", "generated": "...Z", "source": "NOAA GFS ...",
  "grid": { "nx","ny","lo1"(西端),"la1"(北端),"lo2"(東端),"la2"(南端),"dx","dy" },
  "frames": [ { "fh":0, "t":"2026-07-08T09:00"(JST), "u":[...], "v":[...] }, ... ]
}
data は北→南・西→東の行優先（leaflet-velocity と同じ並び）。
"""
import os, sys, json, time, tempfile, datetime
import urllib.request
import numpy as np
import xarray as xr

REGION = dict(left=120.0, right=150.0, top=48.0, bottom=22.0)  # 与那国〜小笠原まで
FHRS = list(range(0, 73, 3))                                   # 0,3,...,72
STRIDE = 1                                                     # 0.25度（GFS最高解像度・間引きしない）
BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"


def cycle_candidates():
    now = datetime.datetime.utcnow()
    base = (now - datetime.timedelta(hours=5)).replace(minute=0, second=0, microsecond=0)
    h = (base.hour // 6) * 6
    c = base.replace(hour=h)
    return [c - datetime.timedelta(hours=6 * i) for i in range(0, 5)]   # 最大24h遡って探す


def filter_url(run, fhr):
    d = run.strftime("%Y%m%d"); hh = run.strftime("%H")
    return (BASE + f"?dir=%2Fgfs.{d}%2F{hh}%2Fatmos"
            f"&file=gfs.t{hh}z.pgrb2.0p25.f{fhr:03d}"
            f"&var_UGRD=on&var_VGRD=on&lev_10_m_above_ground=on"
            f"&subregion=&leftlon={REGION['left']}&rightlon={REGION['right']}"
            f"&toplat={REGION['top']}&bottomlat={REGION['bottom']}")


def fetch(url, dest, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hiko-nisshi-wind/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) < 200 or data[:4] != b"GRIB":      # エラーページ/空
                raise IOError(f"not a GRIB ({len(data)} bytes)")
            with open(dest, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            print(f"  retry {i+1}/{tries}: {e}", file=sys.stderr); time.sleep(6)
    return False


def pick_run():
    for run in cycle_candidates():
        tmp = os.path.join(tempfile.gettempdir(), "gfs_probe.grib2")
        if fetch(filter_url(run, 0), tmp, tries=1):
            print(f"Using GFS run {run:%Y-%m-%d %HZ}")
            return run
        print(f"  run {run:%Y-%m-%d %HZ} not ready")
    raise SystemExit("No available GFS run found")


def read_uv(path):
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    names = list(ds.data_vars)
    uname = "u10" if "u10" in ds else ("u" if "u" in ds else names[0])
    vname = "v10" if "v10" in ds else ("v" if "v" in ds else names[1])
    lats = np.asarray(ds["latitude"].values, dtype=float)
    lons = np.asarray(ds["longitude"].values, dtype=float)
    return lats, lons, np.asarray(ds[uname].values, float), np.asarray(ds[vname].values, float)


def main():
    run = pick_run()
    tmpdir = tempfile.mkdtemp()
    grid = None
    frames = []
    for fhr in FHRS:
        dest = os.path.join(tmpdir, f"f{fhr:03d}.grib2")
        if not fetch(filter_url(run, fhr), dest):
            print(f"  skip f{fhr:03d} (download failed)", file=sys.stderr); continue
        lats, lons, u2d, v2d = read_uv(dest)
        # 北→南・西→東にそろえる
        if lats[0] < lats[-1]:
            lats = lats[::-1]; u2d = u2d[::-1, :]; v2d = v2d[::-1, :]
        if lons[0] > lons[-1]:
            lons = lons[::-1]; u2d = u2d[:, ::-1]; v2d = v2d[:, ::-1]
        # 0.5度へ間引き
        lats = lats[::STRIDE]; lons = lons[::STRIDE]
        u2d = u2d[::STRIDE, ::STRIDE]; v2d = v2d[::STRIDE, ::STRIDE]
        if grid is None:
            grid = dict(nx=int(len(lons)), ny=int(len(lats)),
                        lo1=float(lons[0]), la1=float(lats[0]),
                        lo2=float(lons[-1]), la2=float(lats[-1]),
                        dx=abs(float(lons[1] - lons[0])), dy=abs(float(lats[0] - lats[1])))
        valid = run + datetime.timedelta(hours=fhr)
        jst = valid + datetime.timedelta(hours=9)
        frames.append(dict(fh=fhr, t=jst.strftime("%Y-%m-%dT%H:%M"),
                           u=[round(float(x), 1) for x in u2d.ravel()],
                           v=[round(float(x), 1) for x in v2d.ravel()]))
        print(f"  f{fhr:03d} ok  ({grid['nx']}x{grid['ny']})")
    if not frames:
        raise SystemExit("no frames produced")
    out = dict(run=run.strftime("%Y%m%d%H"),
               generated=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
               source="NOAA GFS 0.25deg (public domain), 10 m wind",
               grid=grid, frames=frames)
    with open("wind.json", "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), ensure_ascii=False)
    sz = os.path.getsize("wind.json")
    print(f"wrote wind.json: {len(frames)} frames, {grid['nx']}x{grid['ny']}, {sz//1024} KB")


if __name__ == "__main__":
    main()
