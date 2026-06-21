import { useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight, Bot } from "lucide-react";

export default function ChatPanel({ reportId, suggestions, onSend, messages, loading }) {
  const [input, setInput] = useState("");
  const promptSuggestions = [...new Set(["Why is growth 0%?", "Which category performed best?", "Show regional trends", ...(suggestions || [])])].slice(0, 6);

  function toolLabel(tool) {
    if (typeof tool === "string") return tool;
    return tool?.name || "tool";
  }

  async function submitMessage(nextMessage) {
    if (!reportId || !nextMessage.trim()) return;
    setInput("");
    await onSend(nextMessage.trim());
  }

  return (
    <section className="sticky bottom-4 rounded-[2rem] border border-white/12 bg-black/30 p-6 shadow-md backdrop-blur-[10px] md:p-7">
      <div className="flex items-center gap-3">
        <div className="rounded-full border border-white/12 p-3 text-white">
          <Bot size={18} />
        </div>
        <div>
          <p className="ui-eyebrow ui-text-soft font-mono-ui uppercase tracking-[0.3em]">Report QA</p>
          <h2 className="ui-title-section mt-1 font-display tracking-[-0.04em] text-white">Interrogate the generated report.</h2>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        {promptSuggestions.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            onClick={() => submitMessage(suggestion)}
            className="ui-body-sm ui-text-muted rounded-full border border-white/12 px-4 py-2 transition hover:bg-white hover:text-black"
          >
            {suggestion}
          </button>
        ))}
      </div>

      <div className="mt-6 max-h-[22rem] space-y-4 overflow-y-auto pr-1">
        {messages.map((message, index) => (
          <motion.div
            key={`${message.role}-${index}-${message.content.slice(0, 12)}`}
            className={`rounded-[1.5rem] border px-4 py-4 ${
              message.role === "user"
                ? "ml-auto max-w-[85%] border-white bg-white text-black"
                : "max-w-[92%] border-white/10 bg-black/8 backdrop-blur-[2px] text-white"
            }`}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <p className="ui-body-sm">{message.content}</p>
            {message.toolsUsed?.length ? (
              <p className={`ui-meta mt-3 font-mono-ui uppercase tracking-[0.24em] ${message.role === "user" ? "text-black/60" : "ui-text-soft"}`}>
                Tools: {message.toolsUsed.map(toolLabel).join(", ")}
              </p>
            ) : null}
          </motion.div>
        ))}
      </div>

      <form
        className="mt-6 flex flex-col gap-3 md:flex-row"
        onSubmit={async (event) => {
          event.preventDefault();
          await submitMessage(input);
        }}
      >
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Why did revenue drop in the latest period?"
          className="ui-body-sm min-h-14 flex-1 rounded-[1.25rem] border border-white/12 bg-white/[0.012] px-4 text-white outline-none transition placeholder:text-white/28 focus:border-white"
        />
        <button
          type="submit"
          disabled={loading}
          className="ui-body-sm group inline-flex min-h-14 items-center justify-center gap-2 rounded-[1.25rem] bg-white px-5 font-medium text-black transition hover:gap-3 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? "Thinking..." : "Send"}
          <ArrowRight size={16} className="transition group-hover:translate-x-1" />
        </button>
      </form>
    </section>
  );
}
