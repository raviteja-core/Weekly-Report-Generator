import { AlertTriangle, CheckCircle2, ShieldAlert } from "lucide-react";

function qualityState(score, warningsCount) {
  if (warningsCount > 0 && (score ?? 1) < 0.8) return "risk";
  if (warningsCount > 0 || (score ?? 1) < 0.95) return "warn";
  return "good";
}

const styles = {
  good: {
    icon: CheckCircle2,
    container: "border-emerald-400/20 bg-emerald-400/10 text-emerald-50",
    badge: "bg-emerald-300/14 text-emerald-100",
    label: "Data Quality: Good",
  },
  warn: {
    icon: AlertTriangle,
    container: "border-amber-400/20 bg-amber-400/10 text-amber-50",
    badge: "bg-amber-300/14 text-amber-100",
    label: "Data Quality: Review Needed",
  },
  risk: {
    icon: ShieldAlert,
    container: "border-rose-400/20 bg-rose-400/10 text-rose-50",
    badge: "bg-rose-300/14 text-rose-100",
    label: "Data Quality: Risk Detected",
  },
};

export default function DataQualityBanner({ qualityReport, warnings }) {
  const score = qualityReport?.quality_score;
  const actionableWarnings = (warnings || []).filter((warning) => {
    const normalized = String(warning || "").toLowerCase();
    return normalized && !normalized.includes("dataset quality score");
  });
  const warningsCount = actionableWarnings.length;
  const state = qualityState(score, warningsCount);
  const config = styles[state];
  const Icon = config.icon;
  const detail =
    actionableWarnings[0] ||
    (score !== undefined && score !== null
      ? `Validated dataset with score ${score}.`
      : "Validated dataset is ready for KPI and chart review.");

  return (
    <section className={`rounded-2xl border p-5 shadow-md backdrop-blur-[3px] ${config.container}`}>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="rounded-full border border-white/10 bg-white/10 p-2.5">
            <Icon size={18} />
          </div>
          <div>
            <p className="ui-body font-semibold">{config.label}</p>
            <p className="ui-body-sm opacity-90">{detail}</p>
          </div>
        </div>
        <span className={`ui-body-sm rounded-full px-4 py-2 font-medium ${config.badge}`}>
          Score: {score ?? "N/A"}
        </span>
      </div>
    </section>
  );
}
