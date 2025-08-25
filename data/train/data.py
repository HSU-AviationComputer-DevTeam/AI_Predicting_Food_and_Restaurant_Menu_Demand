import os, re
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/train")
OUT_DIR = DATA_DIR / "assoc_by_venue"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR/"train.csv")
df["영업일자"] = pd.to_datetime(df["영업일자"])
df["매출수량"] = pd.to_numeric(df["매출수량"], errors="coerce").fillna(0)
df["present"] = (df["매출수량"] > 0).astype(np.uint8)
df["영업장명"] = df["영업장명_메뉴명"].str.split("_").str[0]

min_support_days = 8
min_conf = 0.35
min_lift = 1.15
max_base_prob = 0.98
topk = 300

def safe(name: str) -> str:
    return re.sub(r"[^가-힣A-Za-z0-9._-]+", "_", name)

for venue, g in df.groupby("영업장명"):
    piv = g.pivot_table(index="영업일자", columns="영업장명_메뉴명",
                        values="present", aggfunc="max", fill_value=0)
    if piv.shape[1] < 2:
        continue

    M = piv.to_numpy(dtype=np.uint16)
    D = M.shape[0]
    S = M.sum(axis=0).astype(np.int32)
    menus = np.array(piv.columns)

    C = (M.T @ M).astype(np.int32)
    i, j = np.triu_indices(len(menus), k=1)
    co = C[i, j]; sa = S[i]; sb = S[j]
    Pa, Pb = sa / D, sb / D

    with np.errstate(divide="ignore", invalid="ignore"):
        conf_ab = np.where(sa > 0, co / sa, 0.0)
        conf_ba = np.where(sb > 0, co / sb, 0.0)
        lift = np.where((Pa * Pb) > 0, (co / D) / (Pa * Pb), 0.0)
        jacc = np.where((sa + sb - co) > 0, co / (sa + sb - co), 0.0)
        cosine = np.where((sa * sb) > 0, co / np.sqrt(sa * sb), 0.0)

    mask = (
        (co >= min_support_days) &
        ((conf_ab >= min_conf) | (conf_ba >= min_conf)) &
        (lift >= min_lift) &
        (Pa < max_base_prob) & (Pb < max_base_prob)
    )
    if not np.any(mask):
        continue

    sel = np.where(mask)[0]
    pairs = pd.DataFrame({
        "menu_a": menus[i[sel]],
        "menu_b": menus[j[sel]],
        "support_days": co[sel],
        "conf_a_to_b": conf_ab[sel],
        "conf_b_to_a": conf_ba[sel],
        "lift": lift[sel],
        "jaccard": jacc[sel],
        "cosine": cosine[sel],
    }).sort_values(["lift","support_days","cosine"], ascending=[False,False,False]).head(topk)

    rules = []
    for k in sel:
        if conf_ab[k] >= min_conf:
            rules.append([menus[i[k]], menus[j[k]], int(co[k]), float(conf_ab[k]), float(lift[k])])
        if conf_ba[k] >= min_conf:
            rules.append([menus[j[k]], menus[i[k]], int(co[k]), float(conf_ba[k]), float(lift[k])])
    rules = pd.DataFrame(rules, columns=["antecedent","consequent","support_days","confidence","lift"])\
             .sort_values(["confidence","lift","support_days"], ascending=[False,False,False]).head(topk)

    venue_tag = safe(venue)
    pairs.to_csv(OUT_DIR / f"{venue_tag}_pairs.csv", index=False)
    rules.to_csv(OUT_DIR / f"{venue_tag}_rules.csv", index=False)