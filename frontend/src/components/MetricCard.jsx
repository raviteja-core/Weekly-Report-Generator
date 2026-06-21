import { motion } from "framer-motion";
import { Minus, TrendingDown, TrendingUp } from "lucide-react";

const icons = {
  upward: TrendingUp,
  downward: TrendingDown,
  stable: Minus,
};

function formatValue(value, kind = "number") {
  if (value === null || value === undefined) return "N/A";
  if (kind === "currency") {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(Number(value));
  }
  if (kind === "percent") {
    return `${Number(value).toFixed(1)}%`;
  }
  if (kind === "count") {
    return new Intl.NumberFormat("en-US").format(Number(value));
  }
  return String(value);
}

export default function MetricCard({ metric, index }) {
  const Icon = icons[metric.trendIcon] || Minus;
  const accentStyles =
    metric.trendIcon === "upward"
      ? {
          card: "border-emerald-400/18",
          badge: "bg-emerald-400/12 text-emerald-200",
          icon: "text-emerald-300",
        }
      : metric.trendIcon === "downward"
        ? {
            card: "border-rose-400/18",
            badge: "bg-rose-400/12 text-rose-200",
            icon: "text-rose-300",
          }
        : {
            card: "border-sky-400/16",
            badge: "bg-sky-400/12 text-sky-200",
            icon: "text-sky-300",
          };

  return (
    <motion.article
      className={`group min-w-0 rounded-2xl border bg-white/[0.012] p-5 shadow-md backdrop-blur-[2px] transition duration-300 hover:bg-black/18 ${accentStyles.card}`}
      initial={{ opacity: 0, y: 28 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={{ delay: index * 0.06, duration: 0.45 }}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="ui-eyebrow ui-text-soft font-mono-ui uppercase tracking-[0.3em]">{metric.label}</p>
          <p className="ui-title-metric mt-4 font-display tracking-[-0.05em] text-white">
            {formatValue(metric.value, metric.kind)}
          </p>
          <p className="ui-body-sm ui-text-soft mt-4 max-w-[16rem] break-words">{metric.caption}</p>
        </div>
        <div className={`rounded-full border border-white/10 p-3 transition duration-300 ${accentStyles.badge}`}>
          <Icon size={18} className={accentStyles.icon} />
        </div>
      </div>
      <div className="mt-5">
        <span className={`ui-meta inline-flex items-center gap-1 rounded-full px-3 py-1 font-mono-ui uppercase tracking-[0.18em] ${accentStyles.badge}`}>
          {metric.trendIcon === "upward" ? "↑ Positive" : metric.trendIcon === "downward" ? "↓ Watch" : "→ Stable"}
        </span>
      </div>
    </motion.article>
  );
}
