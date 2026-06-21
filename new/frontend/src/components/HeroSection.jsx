import { motion } from "framer-motion";
import { ArrowRight, PlayCircle } from "lucide-react";

export default function HeroSection() {
  return (
    <section id="overview" className="border-b border-white/10 py-16 sm:py-20">
      <motion.div
        className="mx-auto flex max-w-4xl flex-col items-center text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <p className="ui-eyebrow ui-text-soft font-mono-ui uppercase tracking-[0.38em]">ReportGenie AI</p>
        <h1 className="ui-title-hero mt-6 max-w-5xl font-display tracking-[-0.08em] text-white">
          Turn CSV Data into Executive Reports in Seconds
        </h1>
        <p className="ui-body-lg ui-text-muted mt-5 max-w-2xl">
          Upload your data → get KPIs, charts, and insights instantly
        </p>
        <div className="mt-10 flex flex-col gap-3 sm:flex-row">
          <a
            href="#workspace"
            className="ui-body-sm inline-flex items-center justify-center gap-2 rounded-full bg-white px-6 py-3 font-medium text-black transition hover:translate-y-[-1px]"
          >
            Upload CSV
            <ArrowRight size={16} />
          </a>
          <a
            href="#report"
            className="ui-body-sm inline-flex items-center justify-center gap-2 rounded-full border border-white/16 px-6 py-3 text-white transition hover:bg-white hover:text-black"
          >
            <PlayCircle size={16} />
            View Demo
          </a>
        </div>
      </motion.div>
    </section>
  );
}
