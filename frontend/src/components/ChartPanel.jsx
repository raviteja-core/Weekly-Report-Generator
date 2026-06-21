import { lazy, Suspense } from "react";
import { motion } from "framer-motion";

const Plot = lazy(() => import("react-plotly.js"));

function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(Number(value || 0));
}

function chartInsight(chart) {
  const points = chart.data_points || [];
  if (!points.length) return chart.description || "Interactive chart generated from the cleaned dataset.";

  if (chart.type === "line") {
    const latest = points[points.length - 1];
    const prior = points[points.length - 2];
    if (latest && prior && Number(prior.revenue)) {
      const delta = (((Number(latest.revenue) - Number(prior.revenue)) / Number(prior.revenue)) * 100).toFixed(1);
      return `Latest period revenue is ${formatCurrency(latest.revenue)}, a ${delta}% change versus the previous point.`;
    }
    return `Latest period revenue is ${formatCurrency(latest?.revenue)}.`;
  }

  const leader = points[0];
  const dimension = Object.keys(leader).find((key) => key !== "revenue");
  if (dimension) {
    return `${leader[dimension]} leads this view at ${formatCurrency(leader.revenue)}.`;
  }

  return chart.description || "Interactive chart generated from the cleaned dataset.";
}

export default function ChartPanel({ chart, active }) {
  return (
    <motion.section
      layout
      className={`group min-w-0 overflow-hidden rounded-2xl border p-6 shadow-md transition-all duration-500 ${
        active ? "border-white bg-black/22 backdrop-blur-[2px]" : "border-white/12 bg-white/[0.012] backdrop-blur-[2px] hover:border-white/28 hover:bg-black/16"
      }`}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.15 }}
      whileHover={{ y: -6 }}
    >
      <div className="mb-4 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="ui-eyebrow ui-text-soft font-mono-ui uppercase tracking-[0.3em]">Chart</p>
          <h3 className="ui-title-card mt-2 break-words font-display tracking-[-0.04em] text-white">{chart.title}</h3>
        </div>
        <span className="ui-meta ui-text-soft rounded-full border border-white/12 px-3 py-1 font-mono-ui uppercase tracking-[0.22em] transition duration-300 group-hover:border-white/30">
          Interactive
        </span>
      </div>
      <p className="ui-body-sm ui-text-soft mb-4 max-w-2xl break-words">{chartInsight(chart)}</p>
      <div className="overflow-hidden rounded-[1.5rem] border border-white/10 bg-black/8 backdrop-blur-[2px] transition duration-300 group-hover:border-white/24 group-hover:bg-black/18">
        <Suspense
          fallback={<div className="ui-body-sm ui-text-soft flex h-[340px] items-center justify-center">Loading interactive chart...</div>}
        >
          <Plot
            data={chart.figure.data}
            layout={{
              ...chart.figure.layout,
              autosize: true,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              font: {
                family: "IBM Plex Mono, monospace",
                color: "rgba(255,255,255,0.8)",
              },
              margin: { l: 40, r: 20, t: 54, b: 40 },
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
            className="h-[340px] min-w-0 w-full"
          />
        </Suspense>
      </div>
    </motion.section>
  );
}
