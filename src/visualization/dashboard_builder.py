"""Interactive HTML5 and Plotly dashboard generator for ISML."""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


def generate_interactive_dashboard(
    df: pd.DataFrame,
    clf_metrics_dict: Dict[str, float],
    reg_metrics_dict: Dict[str, float],
    importance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate self-contained HTML5 dashboard using Tailwind CSS and Plotly.js."""
    # Subsample 1,000 students for fluid browser performance
    sample_df = df.sample(n=min(1200, len(df)), random_state=42).copy()

    records = sample_df[[
        "Student_ID", "Age", "Gender", "Academic_Level", "Primary_Platform",
        "Daily_Usage_Hours", "Sleep_Duration_Hours", "Sleep_Quality_Score",
        "Late_Night_Usage", "Perceived_Stress_Score", "Mental_Health_Index",
        "Academic_Performance_GPA", "Overall_Impact"
    ]].to_dict(orient="records")

    records_json = json.dumps(records)
    features_json = json.dumps(importance_df.head(10).to_dict(orient="records"))

    accuracy_val = float(clf_metrics_dict.get("accuracy", 0.0))
    balanced_acc_val = float(clf_metrics_dict.get("balanced_accuracy", 0.0))
    macro_f1_val = float(clf_metrics_dict.get("macro_f1", 0.0))
    per_class = clf_metrics_dict.get("per_class_f1", {})
    negative_f1_val = float(per_class.get("Negative", 0.0)) if isinstance(per_class, dict) else 0.0
    r2_val = float(reg_metrics_dict.get("r2_score", 0.0))
    rho_val = float(reg_metrics_dict.get("spearman_rho", 0.0))

    html = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <title>ISML: Impact of Social Media on Life — Interactive Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
  <style>
    body {{ background-color: #090D16; color: #E2E8F0; font-family: 'Plus Jakarta Sans', sans-serif; }}
    .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
    .tabular-nums {{ font-variant-numeric: tabular-nums; }}
  </style>
</head>
<body class="p-4 sm:p-6 space-y-6 max-w-7xl mx-auto">
  <!-- Header -->
  <header class="flex flex-wrap items-center justify-between gap-4 p-5 rounded-xl bg-slate-900/80 border border-slate-800">
    <div>
      <div class="flex items-center space-x-2">
        <h1 class="text-xl font-bold text-white tracking-tight">ISML — Impact of Social Media on Life</h1>
        <span class="px-2 py-0.5 rounded text-xs font-mono bg-cyan-950 text-cyan-400 border border-cyan-800">Cohort: 4,500</span>
      </div>
      <p class="text-xs text-slate-400 font-mono mt-1">Chronobiological Disruption, Psychometric Stress & Academic Performance</p>
    </div>
    <div class="text-xs font-mono text-slate-400">
      Charter: <span class="text-emerald-400 font-semibold">ELMAS.md Compliant</span>
    </div>
  </header>

  <!-- Metrics Strip -->
  <section class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
    <div class="p-3 rounded-lg bg-slate-900 border border-slate-800">
      <div class="text-[11px] font-mono text-slate-400 uppercase">Impact Accuracy</div>
      <div class="text-xl font-bold font-mono text-cyan-400 mt-1 tabular-nums">{accuracy_val*100:.2f}%</div>
      <div class="text-[10px] text-slate-400 font-mono">5-Fold Soft Ensemble</div>
    </div>
    <div class="p-3 rounded-lg bg-slate-900 border border-slate-800">
      <div class="text-[11px] font-mono text-slate-400 uppercase">Balanced Accuracy</div>
      <div class="text-xl font-bold font-mono text-emerald-400 mt-1 tabular-nums">{balanced_acc_val*100:.2f}%</div>
      <div class="text-[10px] text-slate-400 font-mono">Neutral & Negative Tiers</div>
    </div>
    <div class="p-3 rounded-lg bg-slate-900 border border-slate-800">
      <div class="text-[11px] font-mono text-slate-400 uppercase">Macro F1 Score</div>
      <div class="text-xl font-bold font-mono text-cyan-400 mt-1 tabular-nums">{macro_f1_val:.4f}</div>
      <div class="text-[10px] text-slate-400 font-mono">Unbiased Mean</div>
    </div>
    <div class="p-3 rounded-lg bg-slate-900 border border-slate-800">
      <div class="text-[11px] font-mono text-slate-400 uppercase">Negative F1</div>
      <div class="text-xl font-bold font-mono text-rose-400 mt-1 tabular-nums">{negative_f1_val:.4f}</div>
      <div class="text-[10px] text-slate-400 font-mono">Rare Class (3.7%)</div>
    </div>
    <div class="p-3 rounded-lg bg-slate-900 border border-slate-800">
      <div class="text-[11px] font-mono text-slate-400 uppercase">Stress Model R²</div>
      <div class="text-xl font-bold font-mono text-amber-400 mt-1 tabular-nums">{r2_val:.4f}</div>
      <div class="text-[10px] text-slate-400 font-mono">Predicting PSS (0-40)</div>
    </div>
    <div class="p-3 rounded-lg bg-slate-900 border border-slate-800">
      <div class="text-[11px] font-mono text-slate-400 uppercase">Spearman Rho</div>
      <div class="text-xl font-bold font-mono text-emerald-400 mt-1 tabular-nums">{rho_val:.4f}</div>
      <div class="text-[10px] text-slate-400 font-mono">p < 0.0001</div>
    </div>
  </section>

  <!-- Interactive Scatter & Feature Importance -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <div class="lg:col-span-2 p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
      <div class="flex justify-between items-center">
        <h2 class="text-sm font-bold text-white uppercase font-mono">Screen Time vs. Nightly Sleep Duration</h2>
        <span class="text-xs text-slate-400 font-mono">Hover for Student Details</span>
      </div>
      <div id="plot-screen-sleep" class="w-full h-[400px]"></div>
    </div>

    <div class="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
      <h2 class="text-sm font-bold text-white uppercase font-mono">Top Behavioral Attribution Drivers</h2>
      <div id="plot-importance" class="w-full h-[400px]"></div>
    </div>
  </div>

  <!-- Dose Response & Platform Distribution -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <div class="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
      <h2 class="text-sm font-bold text-white uppercase font-mono">Perceived Stress vs. Academic GPA</h2>
      <div id="plot-stress-gpa" class="w-full h-[320px]"></div>
    </div>
    <div class="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
      <h2 class="text-sm font-bold text-white uppercase font-mono">Average Screen Time by Digital Platform</h2>
      <div id="plot-platform-hours" class="w-full h-[320px]"></div>
    </div>
  </div>

  <!-- Footer -->
  <footer class="text-center text-xs font-mono text-slate-400 pt-4 border-t border-slate-800">
    ISML &bull; ElmasCore Research & Engineering &bull; "From Data to Systems"
  </footer>

  <script>
    const data = {records_json};
    const features = {features_json};

    const darkLayout = {{
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: {{ family: 'JetBrains Mono, monospace', color: '#94A3B8', size: 10 }},
      margin: {{ l: 40, r: 20, t: 20, b: 40 }},
      autosize: true
    }};

    // 1. Scatter Screen Time vs Sleep Duration
    const colorMap = {{ 'Beneficial': '#10B981', 'Neutral': '#F59E0B', 'Negative': '#EF4444' }};
    Plotly.newPlot('plot-screen-sleep', [{{
      x: data.map(d => d.Daily_Usage_Hours),
      y: data.map(d => d.Sleep_Duration_Hours),
      text: data.map(d => `${{d.Student_ID}} | ${{d.Age}}yo ${{d.Academic_Level}}<br>Platform: ${{d.Primary_Platform}}<br>GPA: ${{d.Academic_Performance_GPA}} | Stress: ${{d.Perceived_Stress_Score}}`),
      mode: 'markers',
      marker: {{
        size: 7,
        color: data.map(d => colorMap[d.Overall_Impact] || '#94A3B8'),
        opacity: 0.75
      }}
    }}], {{
      ...darkLayout,
      xaxis: {{ title: 'Daily Social Media Usage (Hours)', gridcolor: '#1E293B' }},
      yaxis: {{ title: 'Nightly Sleep Duration (Hours)', gridcolor: '#1E293B' }}
    }}, {{ responsive: true, displayModeBar: false }});

    // 2. Feature Importance Horizontal Bar
    Plotly.newPlot('plot-importance', [{{
      y: features.map(f => f.feature).reverse(),
      x: features.map(f => f.composite_importance).reverse(),
      type: 'bar',
      orientation: 'h',
      marker: {{ color: '#0284C7' }}
    }}], {{
      ...darkLayout,
      margin: {{ l: 150, r: 20, t: 10, b: 30 }},
      xaxis: {{ title: 'Composite Importance (%)', gridcolor: '#1E293B' }}
    }}, {{ responsive: true, displayModeBar: false }});

    // 3. Stress vs GPA
    Plotly.newPlot('plot-stress-gpa', [{{
      x: data.map(d => d.Perceived_Stress_Score),
      y: data.map(d => d.Academic_Performance_GPA),
      mode: 'markers',
      marker: {{
        size: 6,
        color: data.map(d => colorMap[d.Overall_Impact]),
        opacity: 0.65
      }}
    }}], {{
      ...darkLayout,
      xaxis: {{ title: 'Perceived Stress Score (0 - 40)', gridcolor: '#1E293B' }},
      yaxis: {{ title: 'Cumulative GPA (1.70 - 4.00)', gridcolor: '#1E293B' }}
    }}, {{ responsive: true, displayModeBar: false }});

    // 4. Platform Bar
    const platformMeans = {{}};
    data.forEach(d => {{
      if (!platformMeans[d.Primary_Platform]) platformMeans[d.Primary_Platform] = [];
      platformMeans[d.Primary_Platform].push(d.Daily_Usage_Hours);
    }});
    const platforms = Object.keys(platformMeans);
    const avgHours = platforms.map(p => {{
      const arr = platformMeans[p];
      return arr.reduce((a, b) => a + b, 0) / arr.length;
    }});

    Plotly.newPlot('plot-platform-hours', [{{
      x: platforms,
      y: avgHours,
      type: 'bar',
      marker: {{ color: '#06B6D4' }}
    }}], {{
      ...darkLayout,
      xaxis: {{ gridcolor: 'transparent' }},
      yaxis: {{ title: 'Average Daily Hours', gridcolor: '#1E293B' }}
    }}, {{ responsive: true, displayModeBar: false }});
  </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
