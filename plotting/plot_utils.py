"""Shared manuscript colors, typography, axes, and PDF export."""

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator, FuncFormatter

from .utils import (compact_methods, compact_scenarios, compact_strata,
                    display_phenotype, methods, phenotypes)


method_colors = {
    "Best single-source PRS": "#A6A6A6",
    "Best single-source BLocPRS": "#555555",
    "MultiPRS": "#6BAED6",
    "BLocPRS (fixed composition)": "#3182BD",
    "BLocPRS": "#6A51A3",
}

family_colors = {"continuous": "#2A9D8F", "binary": "#C44E67"}

# Font sizes are in points at the saved figure's physical dimensions.
style = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 6.5,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.bfit": "Liberation Sans:bold:italic",
    "mathtext.sf": "Liberation Sans",
    "mathtext.cal": "DejaVu Sans",
    "axes.linewidth": 0.52,
    "axes.edgecolor": "#5A5A5A",
    "axes.labelcolor": "#303030",
    "axes.labelsize": 6.8,
    "axes.titlesize": 7,
    "text.color": "#303030",
    "xtick.color": "#303030",
    "ytick.color": "#303030",
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.52,
    "ytick.major.width": 0.52,
    "legend.fontsize": 6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


# Use the same axis styling without changing other figures' settings.
def style_axis(ax, grid_axis=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#5A5A5A")
    ax.spines[["left", "bottom"]].set_linewidth(0.52)
    ax.tick_params(direction="out", length=3, width=0.52, pad=1.5)

    if grid_axis is not None:
        ax.set_axisbelow(True)
        ax.grid(axis=grid_axis, color="#D8D8D8", alpha=0.36, linewidth=0.32)


# Keep the specified canvas size; no automatic crop or font rescaling.
def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style):
        fig.savefig(path, format="pdf", dpi=450)

    # Release artists between consecutive multi-panel figures.
    fig.clear()
    plt.close(fig)
    gc.collect()
    return path


contribution_colors = {"EUR": "#228833", "AFR": "#C43C4E", "EAS": "#0072B2", "AMR": "#CC9900",
                       "CSA": "#8064A2", "Female": "#D9776A", "Male": "#72A58B"}
source_order = ["EUR", "AFR", "EAS", "AMR", "CSA", "Female", "Male"]
reference_bins = ["0-50", "50-70", "70-85", "85-90", "90-95", "95-99", "99-100"]


def ordered_sources(labels):
    observed = set(labels)
    return [source for source in source_order if source in observed] + sorted(observed.difference(source_order))


def contribution_palette(labels):
    other = sorted(set(labels).difference(contribution_colors))
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Dark2").colors)
    return {**dict(zip(other, colors)), **contribution_colors}


# Both distributions retain every window and share a phenotype-specific scale.
def draw_signed(ax, rows, column, rng, colors=None):
    import numpy as np
    from matplotlib.colors import to_rgba
    from matplotlib.ticker import MaxNLocator

    colors = contribution_colors if colors is None else colors
    sources = ordered_sources(rows.expert_label)
    for position, source in enumerate(sources):
        group = rows.loc[rows.expert_label.eq(source)].sort_values("window")
        values, color = group[column].to_numpy(float), colors[source]
        face = to_rgba(color, .16)

        # Whiskers show the 5th/95th percentiles; the scatter still shows every window.
        ax.boxplot([values], positions=[position], widths=.42, whis=(5, 95), patch_artist=True,
                   showfliers=False, boxprops={"facecolor": face, "edgecolor": face, "linewidth": .68},
                   whiskerprops={"color": color, "linewidth": .68}, capprops={"color": color, "linewidth": .68},
                   medianprops={"color": "#5A5A5A", "linewidth": .92})
        ax.scatter(position + rng.uniform(-.075, .075, len(values)), values,
                   s=3.5, color=color, alpha=.40, edgecolors="none", zorder=3)
        ax.scatter([position], [values.mean()], marker="o", s=11, facecolors="white",
                   edgecolors="#333333", linewidths=.5, zorder=5)
    both = rows[["reference", "effective"]].to_numpy()
    low, high = min(0, both.min()), max(0, both.max())
    pad = max((high-low)*.10, 1e-6)
    ax.set_ylim(low-pad, high+pad)
    ax.axhline(0, color="#888888", lw=.6)
    ax.set(xticks=np.arange(len(sources)), xticklabels=sources, xlim=(-.5, len(sources)-.5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    style_axis(ax)
    ax.tick_params(labelsize=7)



compact_colors = ["#A6A6A6", "#6BAED6", "#6A51A3", "#9279B8", "#B5A3CE"]
compact_labels = ["Best single-source PRS", "MultiPRS", "BLocPRS (1 Mb)",
                  "BLocPRS (3 Mb)", "BLocPRS (5 Mb)"]
cohort_labels = ["Held-out UKB", r"UKB $\rightarrow$ CaG transfer"]
method_labels = ["BLocPRS - 1-Mb (primary)", "BLocPRS - 3-Mb", "BLocPRS - 5-Mb",
                 "MultiPRS", "Best single-source PRS"]


# Use short phenotype labels in the dense supplementary grids.
def compact_phenotype_label(phenotype):
    labels = {"LDL_MA": "LDL-C", "LDL_ADJ_MA": "LDL-C\n(adj.)", "LOG_HDL_MA": "HDL-C\u2020",
              "HEIGHT_MA": "Height", "TC_ADJ_MA": "TC\n(adj.)", "DBP_ADJ_MA": "DBP\n(adj.)",
              "SBP_ADJ_MA": "SBP\n(adj.)", "ASTHMA_MA": "Asthma", "ASTHMA_MT": "Asthma"}
    return labels.get(phenotype, display_phenotype(phenotype))


# Draw stars from the supplied two-sided paired test, without multiplicity adjustment.
def draw_comparison(ax, a, b, p, y, height):
    from matplotlib.transforms import ScaledTranslation

    transform = ax.transData + ScaledTranslation(0, 2 / 72, ax.figure.dpi_scale_trans)
    ax.plot([a, a, b, b], [y, y + height, y + height, y], transform=transform,
            color="#555555", linewidth=0.4)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
    ax.annotate(stars, ((a + b) / 2, y + height), xycoords=transform,
                xytext=(0, 0.8), textcoords="offset points", ha="center", va="bottom", fontsize=5)


# Keep matching continuous/binary scales across both cohorts.
def compact_limits(data, comparisons):
    finite = data.loc[np.isfinite(data.value)]
    error = finite.err.fillna(0)
    low, high = min(0, float((finite.value - error).min())), float((finite.value + error).max())
    span = max(high - low, 0.02)
    return low - (0.03 * span if low < 0 else 0), high + (0.85 if len(comparisons) else 0.14) * span


# Plot raw estimates and stored uncertainty; missing estimates remain NE.
def draw_benchmark_cell(ax, frame, comparisons, ylim, first, auroc=False):
    indexed = frame.set_index("method").reindex(compact_methods)
    values, errors = indexed.value.to_numpy(copy=True), indexed.err.to_numpy()
    finite = np.isfinite(values)
    x = np.arange(5)
    ax.bar(x[finite], values[finite], width=0.72 if auroc else 0.78,
           color=np.asarray(compact_colors)[finite], edgecolor="#666666", linewidth=0.18, zorder=3)
    uncertain = finite & np.isfinite(errors) & (errors > 0)
    ax.errorbar(x[uncertain], values[uncertain], yerr=errors[uncertain], fmt="none",
                ecolor="#363636", elinewidth=0.48, capsize=0, zorder=4)
    if not finite.any():
        ax.text(0.5, 0.5, "NE", transform=ax.transAxes, ha="center", va="center", fontsize=5.2)
    else:
        for position in x[~finite]:
            ax.text(position, 0.10, "NE", transform=ax.get_xaxis_transform(), ha="center", fontsize=5)
        if np.any(finite & ~np.isfinite(errors)):
            ax.text(0.98, 0.91, "NE", transform=ax.transAxes, ha="right", fontsize=5)
    row = frame.iloc[0]
    tests = comparisons.loc[comparisons.scenario.eq(row.scenario) & comparisons.phenotype.eq(row.phenotype)
                            & comparisons.stratum.eq(row.stratum)]

    # Missing SEs affect bracket placement only; they are not drawn as zero-width error bars.
    top = np.max(values[finite] + np.nan_to_num(errors[finite], nan=0)) if finite.any() else 0
    span, level = ylim[1] - ylim[0], 0
    for baseline in [compact_methods[1], compact_methods[0]]:
        selected = tests.loc[tests.baseline.eq(baseline)]
        if selected.empty or not np.isfinite(selected.iloc[0].p_wald) or selected.iloc[0].p_wald >= 0.05:
            continue
        gap, step, height = (0.04, 0.20, 0.015) if auroc else (0.065, 0.19, 0.025)
        draw_comparison(ax, compact_methods.index(baseline), 2, selected.iloc[0].p_wald,
                        top + (gap + step * level) * span, height * span)
        level += 1
    ax.set(xlim=(-0.6, 4.6), ylim=ylim, xticks=[])
    style_axis(ax)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_visible(first)
    ax.tick_params(length=1.8, width=0.45, labelsize=5.1, left=first, labelleft=first, bottom=False)
    if auroc:
        ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    else:
        ticks = MaxNLocator(nbins=3, min_n_ticks=3).tick_values(*ylim)
        ax.set_yticks([tick for tick in ticks if ylim[0] <= tick <= ylim[1]])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if abs(value) < 1e-10 else f"{value:.2f}"))


# Draw the repeated ancestry grids used by the supplementary benchmark figures.
def draw_compact_benchmarks(data, comparisons, candidates, non_eur=False, scenarios=compact_scenarios, auroc=False):
    height = 152 if non_eur else 148
    figure = plt.figure(figsize=(183 / 25.4, height / 25.4))
    left, right, center = 0.100, 0.988, 0.544
    strata = ("non-EUR",) if non_eur else compact_strata
    blocks = [(scenario, family) for scenario in scenarios for family in candidates]
    gap, header = (4, 15.5) if non_eur else (6, 16 if auroc else 12)
    block_height = (height - 10 - (10 if auroc else 9) - gap * (len(blocks) - 1)) / len(blocks)
    plot_height = block_height - header
    row_height = (plot_height - 3 * (len(strata) - 1)) / len(strata)
    title = ("Non-EUR UKB training - non-EUR evaluation" if non_eur else "Full UKB training" if auroc
             else cohort_labels[compact_scenarios.index(scenarios[0])] + " - Full UKB training")
    figure.text(center, 1 - 3 / height, title, fontsize=7, ha="center", va="top")
    figure.legend(handles=[Patch(facecolor=color, label=label) for color, label in zip(compact_colors, compact_labels)],
                  loc="lower center", bbox_to_anchor=(center, 1.5 / height), ncol=5, frameon=False,
                  fontsize=6.1, columnspacing=1.1, handlelength=1.4, handletextpad=0.4)
    bounds = {family: compact_limits(data.loc[data.family.eq(family)], comparisons) for family in candidates}
    if auroc:
        ymax = 1.0
        for key, group in data.groupby(["scenario", "phenotype", "stratum"]):
            tests = comparisons.loc[comparisons.scenario.eq(key[0]) & comparisons.phenotype.eq(key[1])
                                    & comparisons.stratum.eq(key[2])]
            levels = int(tests.p_wald.lt(0.05).sum())
            headroom = 0.04 + 0.20 * (levels - 1) + 0.015 + 0.18 if levels else 0.01
            top = float((group.value + group.err.fillna(0)).max())
            ymax = max(ymax, np.ceil(top / (1 - headroom) * 10) / 10)
        bounds = {"binary": (0, ymax)}
    for block, (scenario, family) in enumerate(blocks):
        subset = data.loc[data.scenario.eq(scenario) & data.family.eq(family)]
        names = [name for name in candidates[family] if name in set(subset.phenotype)]
        width = (right - left) / len(names)
        top = 10 + block * (block_height + gap)
        figure.text(0.018, 1 - top / height, "ABCD"[block], fontsize=8.5, fontweight="bold", va="top")
        if non_eur or auroc:
            figure.text(center, 1 - top / height, cohort_labels[compact_scenarios.index(scenario)],
                        fontsize=7.5, ha="center", va="top")
        band_top = top + (4 if auroc else 3.5 if non_eur else 0)
        color = "#2f7f69" if family == "continuous" else "#b34860"
        figure.add_artist(Rectangle((left, 1 - (band_top + 3) / height), right - left, 3 / height,
                                    transform=figure.transFigure, facecolor=color, edgecolor="none"))
        figure.text(center, 1 - (band_top + 1.5) / height, family.title(), fontsize=5.7,
                    fontweight="semibold", color="white", ha="center", va="center")
        groups = ["Sex-specific" if name.endswith("_SEX") else "Related-trait" if name.endswith("_MT")
                  else "Multi-ancestry" for name in names]
        begin = 0
        for end in range(1, len(names) + 1):
            if end < len(names) and groups[end] == groups[begin]:
                continue
            x0, x1 = left + begin * width, left + end * width
            colors = {"Multi-ancestry": "#b7d9c7" if family == "continuous" else "#e2b5c6",
                      "Sex-specific": "#a2cdb7", "Related-trait": "#dfa0b8"}
            figure.add_artist(Rectangle((x0, 1 - (band_top + 6.5) / height), x1 - x0 - 0.0015, 3 / height,
                                        transform=figure.transFigure, facecolor=colors[groups[begin]], edgecolor="none"))
            figure.text((x0 + x1) / 2, 1 - (band_top + 5) / height, groups[begin],
                        fontsize=5.7, fontweight="semibold", ha="center", va="center")
            begin = end
        for column, phenotype in enumerate(names):
            figure.text(left + (column + 0.5) * width, 1 - (band_top + 7.5) / height,
                        compact_phenotype_label(phenotype), fontsize=5.2 if auroc else 5.8,
                        ha="center", va="top", linespacing=1.05)
        for row, stratum in enumerate(strata):
            axis_top = 1 - (top + header + row * (row_height + 3)) / height
            for column, phenotype in enumerate(names):
                ax = figure.add_axes([left + column * width, axis_top - row_height / height,
                                      width * 0.91, row_height / height])
                frame = subset.loc[subset.stratum.eq(stratum) & subset.phenotype.eq(phenotype)]
                draw_benchmark_cell(ax, frame, comparisons, bounds[family], column == 0, auroc)
            if not non_eur:
                figure.text(0.051, axis_top - row_height / (2 * height),
                            {"All": "Full", "EUR": "EUR", "non-EUR": "Non-EUR"}[stratum],
                            fontsize=5.4 if auroc else 6, ha="center", va="center", rotation=90)
        ylabel = "AUROC" if auroc else r"Incremental $R^2$" if family == "continuous" else r"Nagelkerke $R^2$"
        figure.text(0.018, 1 - (top + header + plot_height / 2) / height, ylabel,
                    rotation=90, fontsize=6.4, ha="center", va="center")
    return figure


def score_color(family, value):

    # The exponent changes color contrast, not the underlying normalized score.
    shade = float(np.clip(value, 0, 1)) ** 1.65
    if family == "overall":
        gray = 0.96 - 0.84 * shade
        return gray, gray, gray
    low, high = {"continuous": ("#F4FBF7", "#005442"),
                 "binary": ("#FFF3EC", "#871522")}[family]
    low, high = np.array(to_rgb(low)), np.array(to_rgb(high))
    return tuple(low + (high - low) * shade)


def score_size(value):

    # Matplotlib scatter sizes are areas; the positive minimum keeps small scores visible.
    return 0.96 * (18 + 58 * float(np.clip(value, 0, 1)))


def score_bar(ax, x, y, value, family, width=0.92):
    ax.add_patch(Rectangle((x - width / 2, y - 0.35), width, 0.70,
                           facecolor="#EEEEEE", edgecolor="#686868", linewidth=0.42))
    if np.isfinite(value):
        ax.add_patch(Rectangle((x - width / 2, y - 0.35), width * value, 0.70,
                               facecolor=score_color(family, value), edgecolor="none"))
    else:
        ax.text(x, y, "-", ha="center", va="center", color="#8F8F8F", fontsize=7)


def header(ax, left, width, y, height, label, color, fontsize=5.6):
    ax.add_patch(Rectangle((left, y), width, height, facecolor=color,
                           edgecolor="none", clip_on=False))
    ax.text(left + width / 2, y + height / 2, label, ha="center", va="center",
            color="white", fontsize=fontsize, fontweight="semibold", linespacing=0.85)


def draw_matrix(figure, grid, data, label, title):
    axes = [figure.add_subplot(grid[column]) for column in (0, 1, 3)]
    meta, overall_axis, matrix_axis = axes
    columns, groups = [], {}
    for family, names in phenotypes.items():
        family_columns = [family] + names
        columns.extend(family_columns)
        groups.update(dict.fromkeys(family_columns, family))
    values = data.pivot(index="Method", columns="Phenotype", values="Value")
    names = phenotypes["continuous"] + phenotypes["binary"]
    values = values.reindex(index=methods, columns=names)

    # Only display scores clip negative estimates; all comparisons use raw values.
    scores = values.clip(lower=0).divide(values.max(axis=0), axis=1)
    for family, names in phenotypes.items():
        scores[family] = scores[names].mean(axis=1)

    # Give each outcome family equal weight, irrespective of its number of phenotypes.
    overall = scores[["continuous", "binary"]].mean(axis=1)
    limits = [(0, 1), (-0.5, 0.5), (-0.5, len(columns) - 0.5)]
    for ax, (left, right) in zip(axes, limits):
        ax.set_xlim(left, right)
        ax.set_ylim(4.5, -1.94)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines[:].set_visible(False)
        for row in range(5):
            ax.axhspan(row - 0.5, row + 0.5, color="#F5F5F5" if row % 2 == 0 else "white", zorder=0)
        ax.axhline(2.5, color="#777777", linewidth=0.55)
    meta.text(0, 1.25, label, transform=meta.transAxes, fontsize=8.5, fontweight="bold")
    meta.text(0.15, 1.25, title, transform=meta.transAxes, fontsize=7.4, fontweight="semibold")
    header(meta, 0, 1, -1.80, 0.58, "Method", "#263238")
    header(overall_axis, -0.49, 0.98, -1.80, 0.58, "Overall", "#387DA0")
    for row, (method, method_label) in enumerate(zip(methods, method_labels)):
        meta.text(0.03, row, method_label, va="center", fontsize=5.8)
        score_bar(overall_axis, 0, row, overall.loc[method], "overall")
        for column, name in enumerate(columns):
            value = scores.loc[method, name]
            family = groups[name]
            if name in phenotypes:
                score_bar(matrix_axis, column, row, value, family, width=1.10)
            elif np.isfinite(value):
                matrix_axis.scatter(column, row, s=score_size(value), marker="o",
                                    color=score_color(family, value), edgecolor="#3F3F3F",
                                    linewidth=0.35, zorder=3)
            else:
                matrix_axis.text(column, row, "-", fontsize=7, color="#BBBBBB",
                                 va="center", ha="center")
    labels = [("Cont.\nscore" if name == "continuous" else "Binary\nscore")
              if name in phenotypes else display_phenotype(name) for name in columns]
    matrix_axis.set_xticks(range(len(columns)), labels, rotation=55, ha="left",
                          va="bottom", rotation_mode="anchor", fontsize=5.4)
    matrix_axis.xaxis.tick_top()
    matrix_axis.tick_params(axis="x", length=2, width=0.5, pad=1.8)
    for tick, name in zip(matrix_axis.get_xticklabels(), columns):
        if name in phenotypes:
            tick.set_rotation(0)
            tick.set_ha("center")
    for family in phenotypes:
        positions = [index for index, name in enumerate(columns) if groups[name] == family]
        header(matrix_axis, min(positions) - 0.44, len(positions) - 0.12, -1.80, 0.58,
               family.title() + " phenotypes", family_colors[family])
    subgroup_specs = [
        (phenotypes["continuous"][:9], "Multi-ancestry", "#6F9E99"),
        (phenotypes["continuous"][9:], "Sex-specific", "#427D75"),
        (phenotypes["binary"][:2], "Multi\nancestry", "#CA8194"),
        (phenotypes["binary"][2:], "Related-trait", "#AE536B"),
    ]
    for names, subgroup, color in subgroup_specs:
        positions = [columns.index(name) for name in names]
        header(matrix_axis, min(positions) - 0.5, len(positions) - 0.06, -1.18, 0.70,
               subgroup, color, fontsize=5.3)



def save_pages(figures, path):
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style), PdfPages(path) as pdf:
        for figure in figures:
            pdf.savefig(figure, dpi=450)
            figure.clear()
            plt.close(figure)
            gc.collect()
    return path


# Compare methods on a common absolute scale within each outcome family.
def grouped_estimates(data, methods, colors, title, order_by=None, markers=None,
                      open_first=False, sex_labels=False):
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from .utils import metric_labels, phenotype_label

    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 150/25.4))
    fig.subplots_adjust(left=.095, right=.985, bottom=.115, top=.91, wspace=.30)
    fig.suptitle(title, fontsize=7, y=.987)
    for index, (ax, family) in enumerate(zip(axes, ("continuous", "binary"))):
        selected = data.loc[data.family.eq(family)]
        ordering = selected.loc[selected.method.eq(order_by or methods[-1])]
        order = ordering.sort_values(["value", "phenotype"], kind="stable").phenotype.tolist()
        y = np.arange(len(order))
        step = (.95 if markers else .78) / len(methods)
        for j, (method, color) in enumerate(zip(methods, colors)):
            values = selected.loc[selected.method.eq(method)].set_index("phenotype").reindex(order)
            positions = y + (j-(len(methods)-1)/2)*step
            errors = values.bootstrap_se if "bootstrap_se" in values else None
            if markers is None:
                ax.barh(positions, values.value, height=step*.90, color=color, edgecolor="none",
                        xerr=errors, error_kw={"ecolor": "#414141", "elinewidth": .55, "capsize": 0})
            else:
                ax.errorbar(values.value, positions, xerr=errors, fmt=markers[j], linestyle="none",
                            markersize=2.5 if markers[j] == "D" else 3.0,
                            markerfacecolor="white" if open_first and j == 0 else color,
                            markeredgecolor=color, markeredgewidth=.55,
                            ecolor=color, elinewidth=.55, capsize=0, zorder=3)
        style_axis(ax)
        labels = [phenotype_label(p) + (" (SEX)" if sex_labels and p.endswith("_SEX") else "") for p in order]
        ax.set_yticks(y, labels, fontsize=6.6)
        ax.invert_yaxis()
        se = selected.bootstrap_se if "bootstrap_se" in selected else 0
        low, high = min(0, (selected.value-se).min()), max(0, (selected.value+se).max())
        ax.set_xlim(low*1.1, high*1.05)
        ticks = MaxNLocator(nbins=4).tick_values(*ax.get_xlim())
        ax.set_xticks([tick for tick in ticks if ax.get_xlim()[0] <= tick <= ax.get_xlim()[1]])
        ax.set_xlabel(metric_labels[family], fontsize=7)
        ax.tick_params(axis="x", labelsize=6.4)
        ax.set_title(family.capitalize(), fontsize=7, pad=5)
        ax.text(0, 1.045, chr(65+index), transform=ax.transAxes, fontsize=8, fontweight="bold")
        ax.set_axisbelow(True)
        for row in range(0, len(order), 2):
            ax.axhspan(row-.49, row+.49, color="#F6F7F7" if markers else "#F2F3F4", zorder=0, linewidth=0)
    handles = [Patch(facecolor=color, edgecolor="none") for color in colors]
    if markers:
        handles = [Line2D([], [], linestyle="none", marker=marker, markersize=3.8,
                          markerfacecolor="white" if open_first and i == 0 else color,
                          markeredgecolor=color, markeredgewidth=.55)
                   for i, (color, marker) in enumerate(zip(colors, markers))]
    fig.legend(handles, methods, loc="lower center", bbox_to_anchor=(.5, .012), ncol=len(methods),
               frameon=False, fontsize=6.1, handlelength=1.1, handletextpad=.4, columnspacing=.9)
    return fig


# These colors identify PRS sources consistently across chromosome maps.
source_colors = {
    "EUR": "#228833", "AFR": "#C43C4E", "EAS": "#0072B2", "AMR": "#CC9900",
    "CSA": "#8064A2", "Female": "#C44E67", "Male": "#4C78A8",
}


# Round the largest absolute value upward without clipping heatmap cells.
def nice_limit(value):
    import numpy as np

    value = max(float(value), 1e-12)
    scale = 10 ** np.floor(np.log10(value))
    steps = (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10)
    return next(step for step in steps if step >= value / scale) * scale


# Hatching distinguishes absent genomic windows from effects near zero.
def mark_window_gaps(ax, starts, ends):
    with plt.rc_context({"hatch.linewidth": .4}):
        for left, right in zip(ends[:-1], starts[1:]):
            if right > left:
                ax.axvspan(left, right, facecolor="none", edgecolor="#D6D6D6",
                           hatch="//", linewidth=0, zorder=4)


# Draw each modeled interval at its actual genomic bounds.
def regional_heatmap(ax, panel, values, cmap, norm, fontsize=6):
    import numpy as np
    from matplotlib.collections import PolyCollection

    vertices = [[(lo, row-.5), (hi, row-.5), (hi, row+.5), (lo, row+.5)]
                for row in range(len(values)) for lo, hi in zip(panel["starts"], panel["ends"])]

    # Row-major flattening matches the source-by-window order of the rectangles above.
    mesh = PolyCollection(vertices, array=values.ravel(), cmap=cmap, norm=norm, edgecolors="none")
    ax.add_collection(mesh)
    ax.set_ylim(len(values) - .5, -.5)
    ax.set_yticks(range(len(values)), panel["sources"], fontsize=fontsize)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.hlines(np.arange(.5, len(values) - .5), 0, panel["ends"][-1],
              colors="white", linewidth=.35, alpha=.7)
    mark_window_gaps(ax, panel["starts"], panel["ends"])
    return mesh


# Plot every retained SNP and emphasize genome-wide significance.
def regional_gwas_track(ax, panel, groups, label, detailed=False):
    import numpy as np
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    threshold = -np.log10(5e-8)
    peak = threshold
    for group in groups:
        points = panel["gwas"][group]
        significant = points.LOGP.ge(threshold)
        size = 3.4 if group == "EUR" else 5.8
        sizes = (.95, 3.) if detailed else (max(size*.28, .85), max(size*.82, 2.8))
        for mask, marker_size, alpha in ((~significant, sizes[0], .22), (significant, sizes[1], .90)):
            ax.scatter(points.loc[mask, "POS"] / 1e6, points.loc[mask, "LOGP"], s=marker_size,
                       alpha=alpha, color=source_colors[group], edgecolors="none", rasterized=True)
        peak = max(peak, float(points.LOGP.max()))
    ax.axhline(threshold, color="#9E2F38", linestyle=(0, (3, 2)), linewidth=.65, alpha=.88)
    ymax = 1.12 * peak if detailed else max(threshold * 1.15, peak * 1.12)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.text(.012, .92, label, transform=ax.transAxes, va="top", fontsize=7 if detailed else 6,
            fontweight="semibold", color="#303030",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=.7))
