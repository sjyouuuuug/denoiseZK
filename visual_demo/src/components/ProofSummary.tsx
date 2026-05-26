import { CheckCircle2, XCircle } from "lucide-react";
import type { ProofSummary as ProofSummaryType } from "../types";

function Status({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className={`status ${ok ? "ok" : "bad"}`}>
      {ok ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
      <span>{label}</span>
    </div>
  );
}

export default function ProofSummary({ proof }: { proof: ProofSummaryType }) {
  const rows = [
    ["Primary constraints", proof.primary_constraints.toLocaleString()],
    ["Secondary constraints", proof.secondary_constraints.toLocaleString()],
    ["Primary variables", proof.primary_variables.toLocaleString()],
    ["Secondary variables", proof.secondary_variables.toLocaleString()],
    ["Recursive prove", `${proof.recursive_prove_ms.toFixed(2)} ms`],
    ["Recursive verify", `${proof.recursive_verify_ms.toFixed(2)} ms`],
    ["Compressed prove", `${proof.compressed_prove_ms.toFixed(2)} ms`],
    ["Compressed verify", `${proof.compressed_verify_ms.toFixed(2)} ms`],
    ["Proof size", `${proof.proof_size_bytes.toLocaleString()} bytes`],
  ];

  return (
    <aside className="panel proof-panel">
      <div className="panel-title">Proof Summary</div>
      <div className="status-row">
        <Status ok={proof.recursive_verified} label="Recursive" />
        <Status ok={proof.compressed_verified} label="Compressed" />
      </div>
      <div className="metric-table">
        {rows.map(([label, value]) => (
          <div className="metric-row" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
    </aside>
  );
}
