import type { VisualStep } from "../types";
import { formatFixed, formatFixedVector } from "../format";
import HeatmapGrid from "./HeatmapGrid";

export default function SingleStepView({ step, scale }: { step: VisualStep; scale: number }) {
  const isFused = step.update.mode === "FusedFloor";
  const rawDivisor = scale * scale;
  return (
    <section className="single-step">
      <div className="panel formula-panel">
        <div className="panel-title">Step {step.t} Update</div>
        <div className="formula">
          {isFused
            ? "x_next[j] = floor((alpha*x[j] + beta*epsilon[j]) / S)"
            : "x_next[j] = floor(alpha*x[j]/S) + floor(beta*epsilon[j]/S)"}
        </div>
        <div className="scalar-grid">
          <div>
            <span>alpha</span>
            <strong>{formatFixed(step.alpha, scale)}</strong>
          </div>
          <div>
            <span>beta</span>
            <strong>{formatFixed(step.beta, scale)}</strong>
          </div>
          <div>
            <span>time embedding</span>
            <strong>[{formatFixedVector(step.time_embedding, scale)}]</strong>
          </div>
        </div>
      </div>

      <div className="heatmap-row">
        <HeatmapGrid
          title={isFused ? "alpha*x raw" : "alpha term"}
          matrix={step.update.alpha_term}
          divisor={isFused ? rawDivisor : scale}
        />
        <HeatmapGrid
          title={isFused ? "beta*epsilon raw" : "beta term"}
          matrix={step.update.beta_term}
          divisor={isFused ? rawDivisor : scale}
        />
        {step.update.fused_raw && (
          <HeatmapGrid title="fused raw" matrix={step.update.fused_raw} divisor={rawDivisor} />
        )}
      </div>

      <div className="heatmap-row">
        {step.predictor.conv_raw && (
          <HeatmapGrid title="conv raw" matrix={step.predictor.conv_raw} divisor={rawDivisor} />
        )}
        {step.predictor.conv_pre_activation && (
          <HeatmapGrid
            title="conv pre-activation"
            matrix={step.predictor.conv_pre_activation}
            divisor={scale}
          />
        )}
        {step.predictor.relu_output && (
          <HeatmapGrid title="clipped ReLU output" matrix={step.predictor.relu_output} divisor={scale} />
        )}
      </div>
    </section>
  );
}
