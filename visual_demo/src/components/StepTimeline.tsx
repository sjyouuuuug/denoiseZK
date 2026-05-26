import type { NovaStepView, VisualStep } from "../types";

interface StepTimelineProps {
  trajectory: VisualStep[];
  novaSteps: NovaStepView[];
  selectedStep: number;
  onSelectStep: (step: number) => void;
}

export default function StepTimeline({
  trajectory,
  novaSteps,
  selectedStep,
  onSelectStep,
}: StepTimelineProps) {
  return (
    <section className="panel timeline-panel">
      <div className="panel-title">Denoise Trajectory and Nova Chunks</div>
      <div className="timeline">
        {trajectory.map((step) => (
          <button
            className={`timeline-dot ${step.t === selectedStep ? "active" : ""}`}
            key={step.t}
            onClick={() => onSelectStep(step.t)}
            title={`iteration ${step.t}`}
          >
            {step.t}
          </button>
        ))}
      </div>
      <div className="nova-chunks">
        {novaSteps.map((step) => (
          <div className="nova-chip" key={step.nova_step_index}>
            <strong>Nova {step.nova_step_index}</strong>
            <span>iter {step.iter_start}-{step.iter_end}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
