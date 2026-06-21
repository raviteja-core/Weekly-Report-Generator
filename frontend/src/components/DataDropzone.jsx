import { motion } from "framer-motion";
import { ArrowRight, FileSpreadsheet, Upload } from "lucide-react";

export default function DataDropzone({
  file,
  dragging,
  prompt,
  onPromptChange,
  onFileSelect,
  onDragState,
  onAnalyze,
  onLoadSample,
  loading,
}) {
  return (
    <motion.section
      className="relative overflow-hidden rounded-[2rem] border border-white/12 bg-white/[0.012] p-6 shadow-md backdrop-blur-[2px] transition duration-300 hover:border-white/26 hover:bg-black/14 md:p-8"
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
    >
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(255,255,255,0.08),transparent_24%),linear-gradient(135deg,rgba(255,255,255,0.02),transparent_46%)]" />
      <div className="relative grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
        <div>
          <p className="ui-eyebrow ui-text-soft font-mono-ui uppercase tracking-[0.36em]">Workspace</p>
          <h2 className="ui-title-section mt-4 max-w-3xl font-display tracking-[-0.05em] text-white">
            Upload once, then review a clean dashboard instead of a raw output log.
          </h2>
          <p className="ui-body-sm ui-text-muted mt-5 max-w-xl">
            The workspace validates your data, calculates KPIs, and prepares charts before the report is shown.
          </p>
          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            {[
              ["01", "Validate", "Check quality, schema, and cleaned row consistency first."],
              ["02", "Measure", "Compute revenue, growth, averages, and anomaly signals."],
              ["03", "Present", "Turn outputs into an executive-ready dashboard and report."],
            ].map(([num, title, copy]) => (
              <div key={title} className="rounded-2xl border border-white/10 bg-black/10 p-4">
                <p className="ui-eyebrow ui-text-soft font-mono-ui uppercase tracking-[0.3em]">{num}</p>
                <p className="ui-body-sm mt-3 text-white">{title}</p>
                <p className="ui-body-sm ui-text-soft mt-2">{copy}</p>
              </div>
            ))}
          </div>
        </div>

        <div id="workspace" className="min-w-0 rounded-[1.8rem] border border-white/12 bg-black/8 p-5 backdrop-blur-[3px] transition duration-300 hover:border-white/26 hover:bg-black/16 md:p-6">
          <div
            onDragEnter={() => onDragState(true)}
            onDragOver={(event) => {
              event.preventDefault();
              onDragState(true);
            }}
            onDragLeave={() => onDragState(false)}
            onDrop={(event) => {
              event.preventDefault();
              onDragState(false);
              onFileSelect(event.dataTransfer.files?.[0] || null);
            }}
            className={`rounded-[1.6rem] border border-dashed p-6 transition duration-300 ${
              dragging ? "border-white bg-white/[0.04]" : "border-white/18 bg-white/[0.01]"
            }`}
          >
            <div className="flex items-center gap-4">
              <div className="rounded-2xl border border-white/12 bg-white/[0.05] p-3 text-white">
                <Upload size={20} />
              </div>
              <div>
                <p className="ui-body text-white">Drop a CSV here</p>
                <p className="ui-body-sm ui-text-soft">Upload the latest weekly dataset and let the report rebuild itself.</p>
              </div>
            </div>
            <label className="ui-body-sm mt-6 inline-flex cursor-pointer items-center gap-3 rounded-full border border-white/16 px-4 py-3 text-white transition hover:bg-white hover:text-black">
              <FileSpreadsheet size={16} />
              Choose file
              <input
                type="file"
                accept=".csv"
                className="hidden"
                onChange={(event) => onFileSelect(event.target.files?.[0] || null)}
              />
            </label>
            <p className="ui-meta ui-text-soft mt-4 font-mono-ui uppercase tracking-[0.2em]">
              {file ? `${file.name} selected` : "No file selected"}
            </p>
          </div>

          <label className="mt-5 block">
            <span className="ui-eyebrow ui-text-soft font-mono-ui uppercase tracking-[0.3em]">AI prompt</span>
            <textarea
              value={prompt}
              onChange={(event) => onPromptChange(event.target.value)}
              rows={5}
              className="ui-body-sm mt-3 w-full rounded-[1.4rem] border border-white/12 bg-white/[0.012] px-4 py-4 text-white outline-none transition placeholder:text-white/28 focus:border-white"
              placeholder="Generate a weekly KPI report with metric highlights, charts, anomaly commentary, and executive recommendations."
            />
          </label>

          <div className="mt-5 flex flex-col gap-3 sm:flex-row">
            <button
              type="button"
              onClick={onAnalyze}
              disabled={loading}
              className="ui-body-sm group inline-flex min-h-14 flex-1 items-center justify-center gap-2 rounded-[1.2rem] bg-white px-5 font-medium text-black transition hover:gap-3 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Analyzing..." : "Generate report"}
              <ArrowRight size={16} className="transition group-hover:translate-x-1" />
            </button>
            <button
              type="button"
              onClick={onLoadSample}
              disabled={loading}
              className="ui-body-sm inline-flex min-h-14 items-center justify-center rounded-[1.2rem] border border-white/16 px-5 text-white transition hover:bg-white hover:text-black disabled:cursor-not-allowed disabled:opacity-60"
            >
              Use sample data
            </button>
          </div>
        </div>
      </div>
    </motion.section>
  );
}
