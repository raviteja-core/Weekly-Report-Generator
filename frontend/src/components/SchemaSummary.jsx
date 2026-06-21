import { motion } from "framer-motion";
import { CheckCircle2 } from "lucide-react";

function formatRole(role) {
  return role.charAt(0).toUpperCase() + role.slice(1);
}

function percent(value) {
  const numeric = Number(value || 0);
  return Math.max(0, Math.min(100, Math.round(numeric * 100)));
}

export default function SchemaSummary({ schema, overallConfidence }) {
  const roles = Object.entries(schema?.roles || {});

  return (
    <motion.section
      className="rounded-[1.75rem] border border-white/12 bg-white/[0.014] p-6 shadow-md backdrop-blur-[3px]"
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
    >
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="ui-eyebrow ui-text-soft font-mono-ui uppercase tracking-[0.32em]">Schema</p>
          <h3 className="ui-title-card mt-3 font-display tracking-[-0.04em] text-white">Detected fields</h3>
        </div>
        <span className="ui-body-sm rounded-full border border-white/10 px-4 py-2 text-white/82">
          Confidence {percent(overallConfidence)}%
        </span>
      </div>

      <div className="mt-6 space-y-4">
        {roles.map(([role, value]) => {
          const confidence = percent(schema?.role_diagnostics?.[role]?.confidence);
          return (
            <div key={role} className="rounded-2xl border border-white/10 bg-black/10 p-4">
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <CheckCircle2 size={16} className="text-emerald-300" />
                  <p className="ui-body-sm text-white">
                    {formatRole(role)} <span className="ui-text-soft">-&gt;</span> {value || "Not detected"}
                  </p>
                </div>
                <span className="ui-meta ui-text-soft font-mono-ui">{confidence}%</span>
              </div>
              <div className="mt-3 h-2 rounded-full bg-white/8">
                <div className="h-2 rounded-full bg-white/70 transition-all" style={{ width: `${confidence}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </motion.section>
  );
}
