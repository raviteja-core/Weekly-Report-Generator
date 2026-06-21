import MetricCard from "./MetricCard";

export default function KPIGrid({ metrics }) {
  if (!metrics?.length) return null;

  return (
    <section className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">
      {metrics.map((metric, index) => (
        <MetricCard key={metric.label} metric={metric} index={index} />
      ))}
    </section>
  );
}
