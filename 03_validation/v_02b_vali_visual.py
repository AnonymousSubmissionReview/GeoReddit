"""
This script computes state-level correlations between subreddit participation
(users and posts) and external benchmark indicators (e.g., ACS, BLS). It normalizes
subreddit activity by state-level cumulative denominators to ensure comparability,
then evaluates correlations for each type and subreddit combination, generating
CSV outputs, scatterplots, and optional U.S. maps.

How --type_map interacts with normalization:
    --type_map is provided as a string like:
        "FIN=Financial Analysts and Advisors,DEV=Software Developers + Web Developers"
    Example: "FIN" must match a value in the `type` column of p_user_type.csv; the
    string "Financial Analysts and Advisors" must match a column name in
    o_indicators.csv. For each mapping pair KEY=INDICATOR_NAME, the script:
        1) Selects rows from p_user_type.csv where `type` == KEY (e.g., FIN).
        2) Aggregates per-state subreddit activity (distinct users by union and total posts).
        3) Normalizes by denominators from p_state_counts_cumulative.csv using the chosen --year:
               users_norm = users / {year}_user
               posts_norm = posts / {year}_post
        4) Correlates users_norm and posts_norm against the specified indicator column
           INDICATOR_NAME from o_indicators.csv at the state level.

Input files:
    - p_user_type.csv
        Subreddit-level data containing columns: author, subreddit, type, num (posts).
    - o_2005-06to2023-12_filtered_authors.csv
        Author-to-state mapping with columns: author, state. （ratio2 > 1)
    - p_state_counts_cumulative.csv
        State-level denominators by year. Contains columns like:
            state, {year}_user, {year}_post
        {year}_user: total cumulative distinct geolocated users by state up to that year
        {year}_post: total cumulative posts by state up to that year
    - o_indicators.csv
        External benchmark indicators (e.g., ACS, BLS) with state-level values.[state,INDICATOR_NAME]
        Indicator column names must match values provided in --type_map.

Output files:
    - p_cor_{TYPE}.csv
       Correlation results for each subreddit or combination of subreddits
        within a type. Columns include:
            * combination: subreddit(s) combined
            * users: total distinct geolocated users
            * posts: total posts/comments
            * r_user: Pearson correlation coefficient (users_norm vs indicator)
            * p_user: p-value for r_user
            * r_post: Pearson correlation coefficient (posts_norm vs indicator)
            * p_post: p-value for r_post
            * type: short code from --type_map (e.g., FIN, DEV)
            * indicator: external indicator column from o_indicators.csv
    - p_{TYPE}_{RANK}_scatter.png  # top-K scatter panels (users vs posts) per type
                                   # Scatterplots of normalized users and posts vs. indicator.
    - p_{TYPE}_{RANK}_map.{fmt}    # (optional) US map trio per top-K, else an .html fallback if no kaleido
                                   # with subreddit combination and the number of users and posts
                                   # U.S. state-level triptych maps comparing real indicator values with
                                     normalized GeoReddit users and posts.
    - p_correlations_overall.csv          # summary of top-K rows across all types


Example usage:
    python v_02b_vali_visual.py \
        --input_folder /data/inputs \
        --output_folder /data/outputs \
        --year 2023 \
        --unit "Employment per 1,000 jobs" \
        --type_map "FIN=Financial Analysts and Advisors,DEV=Software Developers + Web Developers" \
        --top_k 10 \
        --make_maps \
        --map_out_format png
"""

import itertools
import os, argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re


# find_col: fuzzy finder for a target column name in a DataFrame
def find_col(df: pd.DataFrame, target: str) -> str:
    t = target.strip().lower().replace("_"," ").replace("-"," ")
    for c in df.columns:
        cc = str(c).strip().lower().replace("_"," ").replace("-"," ")
        if cc == t: return c
    for c in df.columns:
        cc = str(c).strip().lower().replace(" ","")
        if cc == t.replace(" ",""): return c
    raise KeyError(f"Column '{target}' not found. Available: {list(df.columns)}")

# parse_type_map: parse "KEY=Indicator,KEY2=Indicator2" into {KEY: Indicator}
def parse_type_map(s: str):
    out = {}
    if not s:
        return out
    matches = re.finditer(r'(\w+)\s*=\s*([^=]+?)(?=(?:,\s*\w+\s*=|$))', s)
    for m in matches:
        k = m.group(1).strip()
        v = m.group(2).strip().rstrip(",")
        out[k] = v
    return out

def safe_state(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

# draw_scatter_panel: scatter, OLS line, state labels, and a stats box
def _apply_academic_rc(font_family="Times New Roman", base_fs=10):
    mpl.rcParams.update({
        "font.family": font_family,
        "font.size": base_fs,
        "axes.titlesize": base_fs+2,
        "axes.labelsize": base_fs+1,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

# draw_scatter_panel: scatter, OLS line, state labels, and a stats box
def draw_scatter_panel(ax, df, xcol, ycol, title, xlabel, ylabel):
    sub = df[["state", xcol, ycol]].dropna()
    if sub.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=':', alpha=0.4)
        return

    x = sub[xcol].astype(float).values
    y = sub[ycol].astype(float).values

    # Pearson + OLS
    r, p = pearsonr(x, y)
    lr = linregress(x, y)
    slope, intercept, r2 = lr.slope, lr.intercept, (lr.rvalue ** 2)

    ax.scatter(x, y, s=28, alpha=0.9, edgecolors='black', linewidths=0.6, facecolors='#4C78A8')
    for _, row in sub.iterrows():
        ax.annotate(row["state"], (row[xcol], row[ycol]),
                    fontsize=6, xytext=(2, 2), textcoords="offset points")

    xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ys = slope * xs + intercept
    ax.plot(xs, ys, linestyle='--', color='#B22222', linewidth=1)

    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    stats = (
        f"$r$={r:.2f}, $R^2$={r2:.2f}\n"
        f"$y$={slope:.2f}$x$+{intercept:.2f}\n"
        f"$p$={p:.1e}"
    )
    ax.text(0.03, 0.97, stats,
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.8))

    ax.grid(True, linestyle=':', alpha=0.4)

# plot_us_map_trio: three-panel choropleths (benchmark, normalized users, normalized posts)
def plot_us_map_trio(indicator_name: str,
                     df_indicator: pd.DataFrame,
                     df_users: pd.DataFrame,
                     df_posts: pd.DataFrame,
                     out_path: str, fmt: str,
                     comb_label=None, users=None, posts=None,
                     unit=None,
                     font_family="Times New Roman"):
    df_indicator = df_indicator.copy()
    df_users = df_users.copy()
    df_posts = df_posts.copy()

    df_indicator["state"] = safe_state(df_indicator["state"])
    df_users["state"] = safe_state(df_users["state"])
    df_posts["state"] = safe_state(df_posts["state"])

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(f"Real: {indicator_name}", "Normalized Users", "Normalized Posts"),
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}, {"type": "choropleth"}]],
        horizontal_spacing=0.02
    )
    for ann in fig.layout.annotations:
        ann.y = 0.9
        ann.font.size = 12
        ann.font.family = font_family

    def add_panel(col, df, val_col, cbar_x, cbar_title):
        vals = pd.to_numeric(df[val_col], errors="coerce")
        locs = df["state"]
        zmin, zmax = np.nanmin(vals.values), np.nanmax(vals.values)
        fig.add_trace(
            go.Choropleth(
                locations=locs,
                z=vals,
                locationmode="USA-states",
                colorscale="Reds",
                zmin=zmin, zmax=zmax,
                marker_line_color="white",
                colorbar=dict(
                    title=dict(text=cbar_title, font=dict(size=9, family=font_family)),
                    orientation="h",
                    x=cbar_x,
                    y=-0.06,
                    xanchor="center",
                    thickness=6,
                    len=0.26,
                    title_side="bottom"
                ),
                showscale=True
            ),
            row=1, col=col
        )

    add_panel(1, df_indicator, df_indicator.columns[1], 0.17, unit)
    add_panel(2, df_users, df_users.columns[1], 0.50, "Users / GeoReddit User Count (%)")
    add_panel(3, df_posts, df_posts.columns[1], 0.83, "Posts / GeoReddit Post Count (%)")

    fig.update_layout(
        template="plotly_white",
        geo=dict(scope="usa", projection=go.layout.geo.Projection(type="albers usa")),
        geo2=dict(scope="usa", projection=go.layout.geo.Projection(type="albers usa")),
        geo3=dict(scope="usa", projection=go.layout.geo.Projection(type="albers usa")),
      #  title={'text':f"{indicator_name} — Real vs GeoReddit", 'x':0.5, 'y':0.98, 'font': {'size': 16, 'family': font_family}},
        font=dict(family=font_family, size=10),
        height=300, width=2000,
        margin=dict(t=40, b=100, l=50, r=50),
    )

    if comb_label is not None:
        text_lines = [f"Subreddits: {comb_label}"]
        if users is not None and posts is not None:
            text_lines.append(f"User Count: {users}, Post Count: {posts}")
        fig.add_annotation(
            text="<br>".join(text_lines),
            x=0.01, y=-0.27, showarrow=False,
            xref="paper", yref="paper",
            align="left", font=dict(size=10, family=font_family)
        )

    root, _ = os.path.splitext(out_path)
    out_img = f"{root}.{fmt.lower()}"
    try:
        fig.write_image(out_img, scale=3, width=1100, height=380)
        print(f"[OK] saved map: {out_img}")
    except Exception as e:
        out_html = f"{root}.html"
        fig.write_html(out_html, include_plotlyjs="cdn")
        print(f"[WARN] write_image failed ({e}). Saved HTML instead: {out_html}")


def main():
    _apply_academic_rc()  # set global rc

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_folder", required=True, help="Input folder containing CSV files")
    ap.add_argument("--output_folder", required=True, help="Output folder for results")
    ap.add_argument("--type_map", default="MR=Market Research Analysts and Marketing Specialists,ADV=Advertising, Promotions, and Marketing Managers",
                    help="Type mapping string: 'KEY=Indicator Column,...'. Keys map to subreddit types.")
    ap.add_argument("--year", type=int, default=2023, help="Year for normalization denominators")
    ap.add_argument("--make_maps", action="store_true", help="Whether to generate U.S. maps")
    ap.add_argument("--map_out_format", default="png", choices=["png","pdf","svg"], help="Map output format")
    ap.add_argument("--unit", required=True, help="Unit label for indicator (e.g., 'Share of Catholics (%)')")
    ap.add_argument("--top_k", type=int, default=10, help="Top K combinations to analyze")
    ap.add_argument("--font_family", default="Times New Roman", help="Font family for plots")
    args = ap.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    _apply_academic_rc(font_family=args.font_family)

    df = pd.read_csv(os.path.join(args.input_folder,"p_user_type.csv"), dtype=str)
    df["num"] = pd.to_numeric(df["num"], errors="coerce").fillna(0)

    authors = pd.read_csv(os.path.join(args.input_folder,"o_2005-06to2023-12_filtered_authors.csv"),
                          dtype=str, usecols=["author","state"])
    authors["state"] = safe_state(authors["state"])
    df = df.merge(authors, on="author", how="inner")

    sc = pd.read_csv(os.path.join(args.input_folder,"p_state_counts_cumulative.csv"), dtype=str)
    sc = sc.rename(columns={find_col(sc,"state"):"state"})
    sc["state"] = safe_state(sc["state"])
    user_col = find_col(sc, f"{args.year}_user")
    post_col = find_col(sc, f"{args.year}_post")
    sc[user_col] = pd.to_numeric(sc[user_col], errors="coerce")
    sc[post_col] = pd.to_numeric(sc[post_col], errors="coerce")

    ind = pd.read_csv(os.path.join(args.input_folder,"o_indicators.csv"), dtype=str)
    ind = ind.rename(columns={find_col(ind,"state"):"state"})
    ind["state"] = safe_state(ind["state"])

    type_map = parse_type_map(args.type_map)
    all_rows = []

    for typ, ind_label in type_map.items():
        try:
            ind_col = find_col(ind, ind_label)
        except KeyError:
            print(f"[WARN] Indicator column not found for type {typ}: {ind_label}")
            continue

        type_dir = os.path.join(args.output_folder, typ)
        os.makedirs(type_dir, exist_ok=True)

        df_t = df[df["type"] == typ].copy()
        subs = sorted(df_t["subreddit"].dropna().unique().tolist())
        if not subs:
            print(f"[INFO] No subreddits for type {typ}")
            continue

        ind_sub = ind[["state", ind_col]].copy()
        ind_sub[ind_col] = pd.to_numeric(ind_sub[ind_col], errors="coerce")
        den = sc[["state", user_col, post_col]].copy()

        authors_all = df_t["author"].dropna().unique()
        author2id = {a: i for i, a in enumerate(authors_all)}

        use_bitmap = False
        try:
            from pyroaring import BitMap
            use_bitmap = True
        except Exception:
            pass

        users_bm = {}   # (sub, state) -> BitMap()/set()
        posts_sum = {}  # (sub, state) -> float

        for sub in subs:
            g = df_t[df_t["subreddit"] == sub]
            for st, s in g.groupby("state")["author"]:
                if use_bitmap:
                    bm = BitMap([author2id[a] for a in s.dropna().unique() if a in author2id])
                else:
                    bm = set(author2id[a] for a in s.dropna().unique() if a in author2id)
                users_bm[(sub, st)] = bm
            ps = g.groupby("state")["num"].sum()
            for st, val in ps.items():
                posts_sum[(sub, st)] = float(val)

        states_all = ind_sub["state"].unique().tolist()


        base_rows = []
        for sub in subs:
            u_total = 0
            for st in states_all:
                bm = users_bm.get((sub, st))
                if bm is None:
                    continue
                u_total += len(bm)
            p_total = int(round(sum(posts_sum.get((sub, st), 0.0) for st in states_all)))
            base_rows.append({
                "combination": sub,
                "users": int(u_total),
                "posts": int(p_total),
                "r_user": "", "p_user": "", "r_post": "", "p_post": "",
                "indicator": "", "type": typ
            })


        per_type_rows = []
        for r in range(1, len(subs)+1):
            for comb in itertools.combinations(subs, r):
                comb_name = "+".join(comb)

                u_counts, p_counts = {}, {}
                for st in states_all:
                    bms = [users_bm.get((sub, st)) for sub in comb]
                    bms = [bm for bm in bms if bm is not None]
                    if not bms:
                        u_counts[st] = 0
                    else:
                        if use_bitmap:
                            u = BitMap()
                            for bm in bms: u |= bm
                        else:
                            u = set()
                            for bm in bms: u |= bm
                        u_counts[st] = len(u)
                    p_counts[st] = float(sum(posts_sum.get((sub, st), 0.0) for sub in comb))

                users_by_state = pd.Series(u_counts, name="users").reset_index().rename(columns={"index": "state"})
                posts_by_state = pd.Series(p_counts, name="posts").reset_index().rename(columns={"index": "state"})

                tmp = ind_sub.merge(den, on="state", how="left")
                tmp = tmp.merge(users_by_state, on="state", how="left")
                tmp = tmp.merge(posts_by_state, on="state", how="left")

                tmp["users"] = pd.to_numeric(tmp["users"], errors="coerce").fillna(0.0)
                tmp["posts"] = pd.to_numeric(tmp["posts"], errors="coerce").fillna(0.0)
                tmp["users_norm"] = np.where(tmp[user_col] > 0, tmp["users"] / tmp[user_col], np.nan)
                tmp["posts_norm"] = np.where(tmp[post_col] > 0, tmp["posts"] / tmp[post_col], np.nan)

                x = tmp[ind_col].astype(float).values
                y_u = tmp["users_norm"].astype(float).values
                y_p = tmp["posts_norm"].astype(float).values

                def _safe_pearson(xx, yy):
                    mask = np.isfinite(xx) & np.isfinite(yy)
                    if mask.sum() < 3: return (np.nan, np.nan)
                    return pearsonr(xx[mask], yy[mask])

                r_user, p_user = _safe_pearson(x, y_u)
                r_post, p_post = _safe_pearson(x, y_p)

                users_total = int(sum(u_counts.values()))
                posts_total = int(round(sum(p_counts.values())))

                per_type_rows.append({
                    "combination": comb_name,
                    "users": users_total,
                    "posts": posts_total,
                    "r_user": r_user, "p_user": p_user,
                    "r_post": r_post, "p_post": p_post,
                    "type": typ,
                    "indicator": ind_label
                })

        full_res = pd.DataFrame(per_type_rows).copy()
        for c in ["r_user", "r_post", "p_user", "p_post", "users", "posts"]:
            if c in full_res.columns:
                full_res[c] = pd.to_numeric(full_res[c], errors="coerce")
        full_res = full_res.sort_values(by="r_user", ascending=False)

        col_order = ["combination", "users", "posts", "r_user", "p_user", "r_post", "p_post", "type", "indicator"]
        base_df = pd.DataFrame(base_rows)

        for c in col_order:
            if c not in base_df.columns:
                base_df[c] = np.nan
            if c not in full_res.columns:
                full_res[c] = np.nan

        base_df = base_df[col_order]
        full_res = full_res[col_order]
        out_df = pd.concat([base_df, full_res], ignore_index=True)

        for col in ["users", "posts"]:
            if col in out_df.columns:
                out_df[col] = pd.to_numeric(out_df[col], errors="coerce").round().astype("Int64")

        full_path = os.path.join(type_dir, f"p_cor_{typ}.csv")
        out_df.to_csv(full_path, index=False)
        print(f"[OK] saved full combos: {full_path}")

        top_k = max(1, int(args.top_k))
        res = full_res.head(top_k).reset_index(drop=True)
        res["rank"] = res.index + 1

        for i in range(len(res)):
            row = res.loc[i].copy()
            comb = row["combination"].split("+")

            u_counts, p_counts = {}, {}
            for st in states_all:
                bms = [users_bm.get((sub, st)) for sub in comb]
                bms = [bm for bm in bms if bm is not None]
                if not bms:
                    u_counts[st] = 0
                else:
                    if use_bitmap:
                        u = BitMap()
                        for bm in bms: u |= bm
                    else:
                        u = set()
                        for bm in bms: u |= bm
                    u_counts[st] = len(u)
                p_counts[st] = float(sum(posts_sum.get((sub, st), 0.0) for sub in comb))

            users_total = int(sum(u_counts.values()))
            posts_total = int(round(sum(p_counts.values())))

            row["type"] = typ
            row["indicator"] = ind_label
            row["users"] = users_total
            row["posts"] = posts_total
            all_rows.append(row.to_dict())

            comb_label = " + ".join([f"r/{s}" for s in comb])
            rank = int(row["rank"])

            base = ind[["state", ind_col]].copy()
            base[ind_col] = pd.to_numeric(base[ind_col], errors="coerce")
            base = base.merge(sc[["state", user_col, post_col]], on="state", how="left")

            base_users = pd.Series(u_counts, name="users").reset_index().rename(columns={"index": "state"})
            base_posts = pd.Series(p_counts, name="posts").reset_index().rename(columns={"index": "state"})
            base = base.merge(base_users, on="state", how="left")
            base = base.merge(base_posts, on="state", how="left")

            base["users"] = pd.to_numeric(base.get("users"), errors="coerce").fillna(0.0)
            base["posts"] = pd.to_numeric(base.get("posts"), errors="coerce").fillna(0.0)
            base["users_norm"] = np.where(base[user_col] > 0, base["users"] / base[user_col], np.nan)
            base["posts_norm"] = np.where(base[post_col] > 0, base["posts"] / base[post_col], np.nan)


            fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
            draw_scatter_panel(
                axes[0], base, "users_norm", ind_col,
                f"Users",
                f'User-norm "{typ}"', ind_label
            )
            draw_scatter_panel(
                axes[1], base, "posts_norm", ind_col,
                f"Posts",
                f'Post-norm "{typ}"', ind_label
            )
          #  plt.suptitle(f'{ind_label} – Users vs. Posts', fontsize=13)

            #anno = f"Subreddits: {comb_label}\nUser Count: {users_total:,}, Post Count: {posts_total:,}"
            #fig = plt.gcf()
            #fig.subplots_adjust(top=0.86, bottom=0.22, wspace=0.25)
            #left_ax = axes[0]
            #pos = left_ax.get_position()
            #fig.text(pos.x0, pos.y0 - 0.10, anno, ha="left", va="top", fontsize=9)

            out_png = os.path.join(type_dir, f"p_{typ}_{rank}_scatter.png")
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] saved scatter: {out_png}")

            if args.make_maps:
                df_real = base[["state", ind_col]].copy().rename(columns={ind_col: "indicator"})
                df_u = base[["state", "users_norm"]].copy()
                df_p = base[["state", "posts_norm"]].copy()
                df_u.loc[:, "users_norm"] = df_u["users_norm"] * 100
                df_p.loc[:, "posts_norm"] = df_p["posts_norm"] * 100
                plot_us_map_trio(
                    ind_label, df_real, df_u, df_p,
                    os.path.join(type_dir, f"p_{typ}_{rank}_map"),
                    args.map_out_format, comb_label,
                    users=users_total, posts=posts_total,
                    unit=args.unit, font_family=args.font_family
                )

    if all_rows:
        overall = pd.DataFrame(all_rows)
        for col in ["users", "posts"]:
            if col in overall.columns:
                overall[col] = pd.to_numeric(overall[col], errors="coerce").round().astype("Int64")
        overall_path = os.path.join(args.output_folder,"p_correlations_overall.csv")
        overall.to_csv(overall_path, index=False)
        print("[OK] overall CSV saved:", overall_path)

if __name__ == "__main__":
    main()