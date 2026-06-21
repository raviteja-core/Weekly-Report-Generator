import { useEffect, useState } from "react";
import { LoaderCircle } from "lucide-react";

const steps = ["Analyzing data...", "Calculating KPIs...", "Generating charts..."];

export default function LoadingState() {
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    const interval = window.setInterval(() => {
      setStepIndex((current) => (current + 1) % steps.length);
    }, 1200);

    return () => window.clearInterval(interval);
  }, []);

  return (
    <section className="rounded-[1.9rem] border border-white/12 bg-white/[0.014] p-6 shadow-md backdrop-blur-[3px]">
      <div className="flex items-center gap-3">
        <LoaderCircle size={18} className="animate-spin text-white" />
        <div>
          <p className="ui-body font-semibold text-white">{steps[stepIndex]}</p>
          <p className="ui-body-sm ui-text-soft mt-1">ReportGenie is validating the dataset and assembling the dashboard.</p>
        </div>
      </div>
      <div className="mt-6 grid gap-4 md:grid-cols-3 xl:grid-cols-4">
        {[0, 1, 2, 3].map((item) => (
          <div key={item} className="h-32 animate-pulse rounded-[1.5rem] border border-white/8 bg-white/[0.03]" />
        ))}
      </div>
    </section>
  );
}
