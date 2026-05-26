import { useEffect, useMemo, useState } from "react";
import type { VisualDemo } from "./types";
import HeatmapGrid from "./components/HeatmapGrid";
import StepTimeline from "./components/StepTimeline";
import ProofSummary from "./components/ProofSummary";
import SingleStepView from "./components/SingleStepView";
import ComplexityComparison from "./components/ComplexityComparison";

export default function App() {
  const [demo, setDemo] = useState<VisualDemo | null>(null);
  const [selectedStep, setSelectedStep] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/denoise_demo.json")
      .then((res) => {
        if (!res.ok) throw new Error(`failed to load JSON: ${res.status}`);
        return res.json();
      })
      .then((data: VisualDemo) => {
        setDemo(data);
        setSelectedStep(data.trajectory[0]?.t ?? 0);
      })
      .catch((err) => setError(String(err)));
  }, []);

  const step = useMemo(
    () => demo?.trajectory.find((item) => item.t === selectedStep) ?? demo?.trajectory[0],
    [demo, selectedStep]
  );

  if (error) {
    return <main className="app-shell"><div className="panel">Error: {error}</div></main>;
  }

  if (!demo || !step) {
    return <main className="app-shell"><div className="panel">Loading denoise demo...</div></main>;
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <h1>{demo.metadata.title}</h1>
          <p>{demo.metadata.description}</p>
        </div>
        <div className="config-strip">
          <span>{demo.metadata.backend}</span>
          <span>{demo.metadata.update_mode}</span>
          <span>scale {demo.config.scale}</span>
          <span>{demo.config.total_iters} iters</span>
        </div>
      </header>

      <section className="layout">
        <div className="main-column">
          <StepTimeline
            trajectory={demo.trajectory}
            novaSteps={demo.nova_steps}
            selectedStep={selectedStep}
            onSelectStep={setSelectedStep}
          />
          <div className="nav-row">
            <button onClick={() => setSelectedStep(Math.max(0, selectedStep - 1))}>Prev</button>
            <strong>Iteration {selectedStep}</strong>
            <button
              onClick={() =>
                setSelectedStep(Math.min(demo.trajectory.length - 1, selectedStep + 1))
              }
            >
              Next
            </button>
          </div>
          <div className="heatmap-row primary-heatmaps">
            <HeatmapGrid title={`x_${step.t}`} matrix={step.x} divisor={demo.config.scale} />
            <HeatmapGrid
              title={`epsilon_${step.t}`}
              matrix={step.epsilon}
              divisor={demo.config.scale}
            />
            <HeatmapGrid title={`x_${step.t + 1}`} matrix={step.x_next} divisor={demo.config.scale} />
          </div>
          <SingleStepView step={step} scale={demo.config.scale} />
          <ComplexityComparison comparisons={demo.comparisons} />
        </div>

        <div className="side-column">
          <section className="panel config-panel">
            <div className="panel-title">Configuration</div>
            <dl>
              <dt>backend</dt><dd>{demo.metadata.backend}</dd>
              <dt>update mode</dt><dd>{demo.metadata.update_mode}</dd>
              <dt>scale</dt><dd>{demo.config.scale}</dd>
              <dt>state shape</dt><dd>{demo.config.state_height}x{demo.config.state_width}</dd>
              <dt>time dim</dt><dd>{demo.config.time_embedding_dim}</dd>
              <dt>Nova chunks</dt><dd>{demo.config.num_steps} x {demo.config.num_iters_per_step}</dd>
              <dt>lookup range</dt><dd>{demo.config.lookup_range}</dd>
              <dt>clip max</dt><dd>{demo.config.clip_max}</dd>
            </dl>
          </section>
          <ProofSummary proof={demo.proof} />
        </div>
      </section>
    </main>
  );
}
