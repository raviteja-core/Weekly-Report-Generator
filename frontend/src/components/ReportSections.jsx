import { motion } from "framer-motion";
import { AlertTriangle, ArrowUpRight, CheckCircle2, Sparkles } from "lucide-react";

const sectionConfig = {
  overview: {
    icon: CheckCircle2,
    title: "Overview",
  },
  insights: {
    icon: Sparkles,
    title: "Key Insights",
  },
  risks: {
    icon: AlertTriangle,
    title: "Risks / Warnings",
  },
  opportunities: {
    icon: ArrowUpRight,
    title: "Opportunities",
  },
};

function firstSentences(text, limit = 1) {
  if (!text) return [];
  return text
    .split(/(?<=[.!?])\s+/)
    .map((part) => part.trim())
    .filter(Boolean)
    .slice(0, limit);
}

function dedupe(items) {
  return [...new Set(items.filter(Boolean))].slice(0, 3);
}

function buildSections(report) {
  const overview = dedupe([
    ...firstSentences(report.summary, 2),
    report.story?.[0]?.headline,
  ]).slice(0, 3);

  const insights = dedupe(
    (report.insights || [])
      .filter((item) => item.tone !== "caution")
      .flatMap((item) => [`${item.title}: ${firstSentences(item.body, 1)[0] || item.body}`]),
  ).slice(0, 3);

  const risks = dedupe([
    ...(report.warnings || []),
    ...(report.insights || [])
      .filter((item) => item.tone === "caution")
      .filter((item) => !String(item.body || "").toLowerCase().includes("dataset quality score is 1"))
      .map((item) => `${item.title}: ${firstSentences(item.body, 1)[0] || item.body}`),
  ]).slice(0, 3);

  const opportunities = dedupe([
    report.story?.[1]?.headline,
    ...(report.suggested_questions || []).map((item) => item.replace(/\?$/, "")),
  ]).slice(0, 3);

  return { overview, insights, risks, opportunities };
}

function SectionCard({ kind, items, index }) {
  if (!items.length) return null;
  const config = sectionConfig[kind];
  const Icon = config.icon;

  return (
    <motion.section
      className="rounded-[1.75rem] border border-white/12 bg-white/[0.014] p-6 shadow-md backdrop-blur-[3px]"
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={{ delay: index * 0.06, duration: 0.4 }}
    >
      <div className="flex items-center gap-3">
        <div className="rounded-full border border-white/10 bg-white/8 p-2.5 text-white">
          <Icon size={16} />
        </div>
        <h3 className="ui-title-card font-display tracking-[-0.04em] text-white">{config.title}</h3>
      </div>
      <ul className="mt-5 space-y-3">
        {items.map((item) => (
          <li key={item} className="ui-body-sm flex gap-3 text-white/82">
            <span className="mt-1 h-1.5 w-1.5 flex-none rounded-full bg-white/70" />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </motion.section>
  );
}

export default function ReportSections({ report }) {
  if (!report) return null;
  const sections = buildSections(report);

  return (
    <section className="grid gap-6 lg:grid-cols-2">
      <SectionCard kind="overview" items={sections.overview} index={0} />
      <SectionCard kind="insights" items={sections.insights} index={1} />
      <SectionCard kind="risks" items={sections.risks} index={2} />
      <SectionCard kind="opportunities" items={sections.opportunities} index={3} />
    </section>
  );
}
