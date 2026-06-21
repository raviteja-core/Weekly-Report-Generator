import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Download, Link2 } from "lucide-react";

import ChartPanel from "./components/ChartPanel";
import ChatPanel from "./components/ChatPanel";
import DataQualityBanner from "./components/DataQualityBanner";
import DataDropzone from "./components/DataDropzone";
import HeroSection from "./components/HeroSection";
import KPIGrid from "./components/KPIGrid";
import LoadingState from "./components/LoadingState";
import ReportSections from "./components/ReportSections";
import SchemaSummary from "./components/SchemaSummary";
import ShaderBackground from "./components/ShaderBackground";

const defaultPrompt =
  "Generate a premium analytics narrative with KPI highlights, anomalies, trend commentary, and next-step recommendations.";

const starterMessages = [
  {
    role: "assistant",
    content:
      "Upload a CSV and I will turn it into KPI calculations, charts, executive commentary, and a structured weekly report.",
  },
];

function CursorAura() {
  const [cursor, setCursor] = useState({
    x: typeof window !== "undefined" ? window.innerWidth / 2 : 0,
    y: typeof window !== "undefined" ? window.innerHeight / 2 : 0,
    pressed: false,
  });

  useEffect(() => {
    function handleMove(event) {
      setCursor((current) => ({
        ...current,
        x: event.clientX,
        y: event.clientY,
      }));
    }

    function handleDown() {
      setCursor((current) => ({ ...current, pressed: true }));
    }

    function handleUp() {
      setCursor((current) => ({ ...current, pressed: false }));
    }

    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerdown", handleDown);
    window.addEventListener("pointerup", handleUp);

    return () => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerdown", handleDown);
      window.removeEventListener("pointerup", handleUp);
    };
  }, []);

  return (
    <>
      <motion.div
        className="pointer-events-none fixed z-[40] hidden h-10 w-10 rounded-full border border-white/65 mix-blend-difference md:block"
        animate={{
          x: cursor.x - 20,
          y: cursor.y - 20,
          scale: cursor.pressed ? 0.78 : 1,
          opacity: 0.95,
        }}
        transition={{ type: "spring", stiffness: 240, damping: 22, mass: 0.45 }}
      />
      <motion.div
        className="pointer-events-none fixed z-[41] hidden h-2.5 w-2.5 rounded-full bg-white mix-blend-difference md:block"
        animate={{
          x: cursor.x - 5,
          y: cursor.y - 5,
          scale: cursor.pressed ? 1.5 : 1,
        }}
        transition={{ type: "spring", stiffness: 400, damping: 30, mass: 0.2 }}
      />
    </>
  );
}

function mapMetrics(report) {
  if (!report) return [];
  const headline = report.kpis?.headline || report.kpis || {};
  const comparisons = report.kpis?.comparisons || {};
  const anomalies = report.kpis?.anomalies || { count: 0, periods: [], dates: [] };
  const anomalyPeriods = anomalies.periods || anomalies.dates || [];
  return [
    {
      label: "Total Revenue",
      value: headline.total_revenue,
      kind: "currency",
      caption: "Commercial output captured across the uploaded dataset.",
      trendIcon: headline.trend_direction,
    },
    {
      label: "Growth Rate",
      value: headline.growth_rate ?? headline.latest_period_growth_rate ?? 0,
      kind: "percent",
      caption: `Measured from ${comparisons.previous_period || "the prior period"} to ${
        comparisons.latest_period || "the latest period"
      }.`,
      trendIcon: (headline.growth_rate ?? headline.latest_period_growth_rate ?? 0) >= 0 ? "upward" : "downward",
    },
    {
      label: "Average Revenue",
      value: headline.average_revenue,
      kind: "currency",
      caption: "Average revenue per normalized row in the source file.",
      trendIcon: "stable",
    },
    {
      label: "Anomaly Periods",
      value: anomalies.count,
      kind: "count",
      caption: anomalyPeriods.length ? anomalyPeriods.join(" / ") : "No anomaly periods were detected.",
      trendIcon: anomalies.count ? "downward" : "upward",
    },
  ];
}

function confidenceLabel(value) {
  if (value >= 0.9) return "High";
  if (value >= 0.8) return "Good";
  if (value >= 0.65) return "Moderate";
  return "Low";
}

function filterDisplayWarnings(warnings) {
  return (warnings || []).filter((warning) => {
    const normalized = String(warning || "").toLowerCase();
    if (!normalized) return false;
    if (normalized.includes("unable to reach ollama")) return false;
    if (normalized.includes("dataset quality score")) return false;
    return true;
  });
}

function normalizeReport(rawReport) {
  if (!rawReport) return null;

  const isFailed = rawReport.status === "failed";
  const qualityReport = rawReport.quality_report || rawReport.details || {};
  const warnings = rawReport.warnings || [];
  const displayWarnings = filterDisplayWarnings(warnings);
  const schema = rawReport.schema || {};
  const schemaRoles = schema.roles || {
    date: schema.date_col || null,
    revenue: schema.revenue_col || null,
    category: schema.category_col || null,
    region: schema.region_col || null,
  };
  const roleDiagnostics =
    schema.role_diagnostics ||
    Object.fromEntries(
      Object.entries(schema.column_confidence || {}).map(([role, confidence]) => [
        role,
        {
          method: "strict-schema",
          confidence,
        },
      ]),
    );

  const kpis =
    rawReport.kpis?.headline
      ? rawReport.kpis
      : {
          headline: {
            total_revenue: rawReport.kpis?.total_revenue ?? null,
            growth_rate: rawReport.kpis?.growth_rate ?? null,
            average_revenue: rawReport.kpis?.average_revenue ?? null,
            trend_direction:
              (rawReport.kpis?.growth_rate ?? 0) > 0 ? "upward" : (rawReport.kpis?.growth_rate ?? 0) < 0 ? "downward" : "stable",
          },
          comparisons: {},
          anomalies: {
            count: rawReport.kpis?.anomalies?.count ?? 0,
            periods: rawReport.kpis?.anomalies?.dates || [],
            dates: rawReport.kpis?.anomalies?.dates || [],
          },
        };

  return {
    ...rawReport,
    title: rawReport.title || (isFailed ? "Data Quality Gate Failed" : "Trust-First KPI Report"),
    summary:
      rawReport.summary ||
      (isFailed
        ? `${rawReport.reason || "Report generation was blocked."} Review the quality warnings before retrying.`
        : "Report generated successfully."),
    preview: rawReport.preview || [],
    charts: rawReport.charts || [],
    insights:
      rawReport.insights ||
      displayWarnings.map((warning) => ({
        title: "Warning",
        body: warning,
        tone: "caution",
      })),
    story:
      rawReport.story ||
      [
        {
          id: "story-quality",
          eyebrow: isFailed ? "Pipeline Blocked" : "Method",
          headline: isFailed ? "The pipeline stopped before KPI generation." : "Results were generated from one cleaned and validated dataset.",
          copy: isFailed
            ? rawReport.reason || "The uploaded data did not pass the quality gate."
            : "Validation, schema detection, KPI calculation, and charts all used the same cleaned dataframe.",
          chart_id: rawReport.charts?.[0]?.id || null,
        },
      ],
    suggested_questions: rawReport.suggested_questions || [
      "Which fields failed validation?",
      "What anomalies were detected?",
      "Should I inspect the cleaned preview?",
    ],
    warnings: displayWarnings,
    systemWarnings: rawReport.meta?.warnings || [],
    meta: rawReport.meta || { generation_mode: isFailed ? "trust-gate-block" : "trust-first-pipeline" },
    schema: {
      ...schema,
      row_count: schema.row_count ?? qualityReport.row_count_after_cleaning ?? 0,
      column_count: schema.column_count ?? schema.columns?.length ?? 0,
      roles: schemaRoles,
      role_diagnostics: roleDiagnostics,
      schema_assist: schema.schema_assist || { used: false },
    },
    kpis,
  };
}

function downloadInsights(report) {
  if (!report) return;
  const payload = {
    title: report.title,
    summary: report.summary,
    insights: report.insights,
    story: report.story,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `insightforge-${report.report_id}-insights.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function SectionTitle({ eyebrow, title, copy, align = "left" }) {
  return (
    <motion.div
      className={align === "center" ? "mx-auto max-w-3xl text-center" : "max-w-3xl"}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.35 }}
      transition={{ duration: 0.6 }}
    >
      <p className="ui-eyebrow ui-text-soft font-mono-ui uppercase tracking-[0.34em]">{eyebrow}</p>
      <h2 className="ui-title-section mt-4 font-display tracking-[-0.05em] text-white">{title}</h2>
      {copy ? <p className="ui-body ui-text-muted mt-5 max-w-2xl">{copy}</p> : null}
    </motion.div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [report, setReport] = useState(null);
  const [status, setStatus] = useState("Ready to analyze a dataset.");
  const [messages, setMessages] = useState(starterMessages);
  const [activeStory, setActiveStory] = useState(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const reportId = params.get("report");
    if (!reportId) return;

    (async () => {
      setLoading(true);
      setStatus("Loading shared report...");
      try {
        const response = await fetch(`/api/reports/${reportId}`);
        if (!response.ok) {
          throw new Error("Unable to load the shared report.");
        }
        const sharedReport = normalizeReport(await response.json());
        setReport(sharedReport);
        setActiveStory(sharedReport.story?.[0] || null);
        setStatus("Shared report loaded.");
      } catch (error) {
        setStatus(error.message);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const metrics = useMemo(() => mapMetrics(report), [report]);
  const activeChartId = activeStory?.chart_id || report?.charts?.[0]?.id;

  async function runAnalysis(nextFile = file) {
    if (!nextFile) {
      setStatus("Choose a CSV file to begin.");
      return;
    }

    const formData = new FormData();
    formData.append("file", nextFile);
    formData.append("prompt", prompt.trim() || defaultPrompt);

    setLoading(true);
      setStatus("Analyzing data and building the executive dashboard...");

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Analysis failed.");
      }
      const normalized = normalizeReport(data);
      setReport(normalized);
      setActiveStory(normalized.story?.[0] || null);
      setMessages([
        starterMessages[0],
        {
          role: "assistant",
          content: normalized.summary,
        },
      ]);
      if (normalized.share_url) {
        window.history.replaceState({}, "", normalized.share_url);
      }
      setStatus(
        normalized.status === "failed"
          ? "Analysis stopped by the trust gate. Review the data quality section below."
          : "Analysis complete. Review the dashboard, charts, and executive report sections below.",
      );
    } catch (error) {
      setStatus(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadSample() {
    setStatus("Loading bundled sample data...");
    try {
      const sampleMeta = await fetch("/api/sample").then((response) => response.json());
      const fileResponse = await fetch(sampleMeta.sample_file);
      if (!fileResponse.ok) {
        throw new Error("Could not load sample data.");
      }
      const blob = await fileResponse.blob();
      const sampleFile = new File([blob], "sales_sample.csv", { type: "text/csv" });
      setFile(sampleFile);
      await runAnalysis(sampleFile);
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function sendChatMessage(content) {
    if (!report?.report_id) return;

    setMessages((current) => [...current, { role: "user", content }]);
    setChatLoading(true);
    try {
      const formData = new FormData();
      formData.append("report_id", report.report_id);
      formData.append("message", content);
      const response = await fetch("/api/chat", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Chat request failed.");
      }
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: data.answer,
          toolsUsed: data.tools_used,
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: error.message,
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  }

  return (
    <div className="relative min-h-screen bg-app text-white">
      <ShaderBackground />
      <CursorAura />
      <div className="pointer-events-none fixed inset-0 z-[1] bg-[linear-gradient(180deg,rgba(0,0,0,0.01),rgba(0,0,0,0.04),rgba(0,0,0,0.08))]" />
      <div className="pointer-events-none fixed inset-0 z-[2] opacity-[0.018] [background-image:linear-gradient(rgba(255,255,255,0.22)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.22)_1px,transparent_1px)] [background-size:72px_72px]" />

      <div className="relative z-10 mx-auto max-w-7xl px-4 py-6 md:px-6 lg:px-8">
        <motion.header
          className="flex flex-col gap-5 border-b border-white/12 pb-6 md:flex-row md:items-center md:justify-between"
          initial={{ opacity: 0, y: -18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div>
            <p className="ui-eyebrow ui-text-soft font-mono-ui uppercase tracking-[0.38em]">ReportGenie AI</p>
            <p className="ui-body-sm ui-text-soft mt-3 max-w-xl">Executive analytics dashboards from uploaded CSV data.</p>
          </div>
          <div className="ui-body-sm ui-text-muted flex flex-wrap items-center gap-3">
            <a className="border-b border-white/30 pb-1 transition hover:border-white" href="#overview">
              Home
            </a>
            <a className="border-b border-white/10 pb-1 transition hover:border-white" href="#report">
              Dashboard
            </a>
            {report ? (
              <>
                <a
                  href={report.share_url}
                  className="inline-flex items-center gap-2 rounded-full border border-white/16 px-4 py-2 transition hover:bg-white hover:text-black"
                >
                  <Link2 size={15} />
                  Share
                </a>
                <button
                  type="button"
                  onClick={() => downloadInsights(report)}
                  className="inline-flex items-center gap-2 rounded-full border border-white/16 px-4 py-2 transition hover:bg-white hover:text-black"
                >
                  <Download size={15} />
                  Export
                </button>
              </>
            ) : null}
          </div>
        </motion.header>

        <main className="pb-16">
          <HeroSection />

          <section id="report" className="py-18">
            <div className="grid gap-8 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
              <div className="min-w-0">
                <SectionTitle
                  eyebrow="Live Demo"
                  title="Executive dashboard"
                  copy="Upload a CSV and review KPIs, data quality, schema, charts, and follow-up answers in one place."
                />
              </div>
              <motion.div
                className="min-w-0 rounded-[1.8rem] border border-white/12 bg-white/[0.014] p-5 shadow-md backdrop-blur-[2px] transition duration-300 hover:border-white/26 hover:bg-black/18"
                initial={{ opacity: 0, y: 28 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.3 }}
              >
                <p className="ui-eyebrow ui-text-soft font-mono-ui uppercase tracking-[0.32em]">System Status</p>
                <p className="ui-body-sm ui-text-muted mt-4">{status}</p>
              </motion.div>
            </div>

            <div className="mt-10">
              <DataDropzone
                file={file}
                dragging={dragging}
                prompt={prompt}
                onPromptChange={setPrompt}
                onFileSelect={setFile}
                onDragState={setDragging}
                onAnalyze={() => runAnalysis()}
                onLoadSample={loadSample}
                loading={loading}
              />
            </div>

            {loading ? <div className="mt-10"><LoadingState /></div> : null}

            <AnimatePresence>
              {report ? (
                <motion.div
                  className="mt-14 space-y-8"
                  initial={{ opacity: 0, y: 24 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  <DataQualityBanner qualityReport={report.quality_report} warnings={report.warnings} />

                  <section className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
                    <motion.article
                      className="rounded-[1.8rem] border border-white/12 bg-white/[0.014] p-6 shadow-md backdrop-blur-[3px]"
                      initial={{ opacity: 0, y: 24 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <p className="ui-eyebrow ui-text-soft font-mono-ui uppercase tracking-[0.34em]">Report</p>
                      <h2 className="ui-title-display mt-4 font-display tracking-[-0.05em] text-white">{report.title}</h2>
                      <p className="ui-body-sm ui-text-muted mt-4 max-w-3xl">{report.summary}</p>
                      <div className="mt-6 flex flex-wrap gap-3">
                        <span className="ui-meta ui-text-soft rounded-full border border-white/12 px-4 py-2 font-mono-ui uppercase tracking-[0.24em]">
                          {report.schema.row_count} rows
                        </span>
                        <span className="ui-meta ui-text-soft rounded-full border border-white/12 px-4 py-2 font-mono-ui uppercase tracking-[0.24em]">
                          {report.schema.column_count} columns
                        </span>
                        <span className="ui-meta ui-text-soft rounded-full border border-white/12 px-4 py-2 font-mono-ui uppercase tracking-[0.24em]">
                          {confidenceLabel(Number(report.confidence ?? report.schema.confidence ?? 0))} confidence
                        </span>
                      </div>
                    </motion.article>

                    <SchemaSummary schema={report.schema} overallConfidence={report.confidence ?? report.schema.confidence} />
                  </section>

                  <KPIGrid metrics={metrics} />

                  <ReportSections report={report} />

                  <section>
                    <SectionTitle
                      eyebrow="Charts"
                      title="Validated chart views"
                      copy="Each chart is wrapped as a report card with a single readout for faster review."
                    />
                    <div className="mt-8 grid gap-5 xl:grid-cols-2">
                      {report.charts.map((chart, index) => {
                        const isLastOdd = report.charts.length % 2 === 1 && index === report.charts.length - 1;
                        return (
                          <div key={chart.id} className={isLastOdd ? "xl:col-span-2" : ""}>
                            <ChartPanel chart={chart} active={chart.id === activeChartId} />
                          </div>
                        );
                      })}
                    </div>
                  </section>

                  {report.status !== "failed" ? (
                    <div className="grid gap-6 xl:grid-cols-[1fr_0.9fr]">
                      <motion.section
                        className="rounded-[1.8rem] border border-white/12 bg-white/[0.014] p-6 shadow-md backdrop-blur-[3px]"
                        initial={{ opacity: 0, y: 24 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true, amount: 0.2 }}
                      >
                        <p className="ui-eyebrow ui-text-soft font-mono-ui uppercase tracking-[0.34em]">Preview</p>
                        <div className="mt-5 overflow-x-auto">
                          <table className="ui-body-sm ui-text-muted min-w-full border-collapse text-left">
                            <thead>
                              <tr className="ui-text-soft border-b border-white/12">
                                {Object.keys(report.preview[0] || {}).map((column) => (
                                  <th key={column} className="ui-meta px-3 py-3 font-mono-ui uppercase tracking-[0.24em]">
                                    {column}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {report.preview.map((row, index) => (
                                <tr key={index} className="border-b border-white/8 last:border-b-0">
                                  {Object.entries(row).map(([column, value]) => (
                                    <td key={`${column}-${index}`} className="px-3 py-4">
                                      {String(value ?? "")}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </motion.section>

                      <ChatPanel
                        reportId={report.report_id}
                        suggestions={report.suggested_questions || []}
                        onSend={sendChatMessage}
                        messages={messages}
                        loading={chatLoading}
                      />
                    </div>
                  ) : null}

                  <motion.section
                    className="min-w-0 overflow-hidden rounded-[2rem] border border-white/12 bg-white/[0.014] p-7 shadow-md backdrop-blur-[3px] transition duration-300 hover:border-white/26 hover:bg-black/16"
                    initial={{ opacity: 0, y: 24 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, amount: 0.25 }}
                  >
                    <div className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
                      <div className="max-w-2xl">
                        <p className="ui-eyebrow ui-text-soft font-mono-ui uppercase tracking-[0.34em]">Shareable Output</p>
                        <h3 className="ui-title-section mt-4 font-display tracking-[-0.04em] text-white">Send the generated report, not a screenshot of your spreadsheet.</h3>
                        <p className="ui-body-sm ui-text-muted mt-4">
                          The final output can be reopened as a live report so stakeholders get the same structured view, charts, and written summary.
                        </p>
                      </div>
                      {report.share_url ? (
                        <a
                          href={report.share_url}
                          className="inline-flex items-center gap-2 rounded-full border border-white/16 px-5 py-3 text-sm transition hover:bg-white hover:text-black"
                        >
                          Open live report
                          <ArrowRight size={16} />
                        </a>
                      ) : (
                        <span className="ui-body-sm ui-text-soft inline-flex items-center gap-2 rounded-full border border-white/10 px-5 py-3">
                          Share disabled for blocked reports
                        </span>
                      )}
                    </div>
                    <p className="ui-body-sm ui-text-strong mt-6 overflow-x-auto rounded-[1.4rem] border border-white/10 bg-black/10 px-4 py-4 font-mono-ui backdrop-blur-[2px]">
                      {report.share_url ? `${window.location.origin}${report.share_url}` : "No share URL was created because the report did not complete successfully."}
                    </p>
                  </motion.section>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </section>
        </main>
      </div>
    </div>
  );
}
