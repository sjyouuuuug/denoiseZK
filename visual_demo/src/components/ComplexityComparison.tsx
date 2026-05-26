import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ComparisonSeries } from "../types";

export default function ComplexityComparison({ comparisons }: { comparisons: ComparisonSeries[] }) {
  return (
    <section className="comparison-grid">
      {comparisons.map((series) => (
        <div className="panel chart-panel" key={series.name}>
          <div className="panel-title">{series.name}</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={series.metrics} margin={{ top: 8, right: 12, left: 8, bottom: 16 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 12 }} interval={0} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip formatter={(value, _name, item) => [`${value} ${item.payload.unit}`, "value"]} />
              <Bar dataKey="value" fill="#2563eb" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ))}
    </section>
  );
}
