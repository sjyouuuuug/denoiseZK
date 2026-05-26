import { formatFixed } from "../format";

interface HeatmapGridProps {
  title: string;
  matrix: number[][];
  divisor?: number;
  digits?: number;
}

function cellColor(value: number, min: number, max: number): string {
  if (min === max) return "#f1f5f9";
  const abs = Math.max(Math.abs(min), Math.abs(max), 1);
  const intensity = Math.min(Math.abs(value) / abs, 1);
  if (value > 0) {
    return `rgba(220, 38, 38, ${0.15 + intensity * 0.75})`;
  }
  if (value < 0) {
    return `rgba(37, 99, 235, ${0.15 + intensity * 0.75})`;
  }
  return "#f8fafc";
}

export default function HeatmapGrid({ title, matrix, divisor = 1, digits = 3 }: HeatmapGridProps) {
  const flat = matrix.flat().map((value) => value / divisor);
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const columns = matrix[0]?.length ?? 1;

  return (
    <section className="panel heatmap-panel">
      <div className="panel-title">{title}</div>
      <div className="heatmap" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
        {matrix.flatMap((row, rowIndex) =>
          row.map((value, colIndex) => {
            const displayValue = value / divisor;
            const formatted = formatFixed(value, divisor, digits);
            return (
              <div
                className="heatmap-cell"
                style={{ background: cellColor(displayValue, min, max) }}
                key={`${rowIndex}-${colIndex}`}
                title={`(${rowIndex}, ${colIndex}) = ${formatted} (raw ${value})`}
              >
                {formatted}
              </div>
            );
          })
        )}
      </div>
    </section>
  );
}
