import React, { useState } from "react";
import { PredictionItem } from "../types";
import { ChevronDown, ChevronUp, Layers } from "lucide-react";

interface PredictionCardProps {
  predictions: Record<string, PredictionItem>;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({ predictions }) => {
  const [showDetails, setShowDetails] = useState<boolean>(false);
  const entries = Object.entries(predictions);
  if (entries.length === 0) return null;

  return (
    <div className="my-3 rounded-2xl border border-white/10 bg-[#171717] overflow-hidden text-xs">
      {/* Clean compact summary badges */}
      <div className="p-3.5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          {entries.map(([phenotype, pred]) => {
            const isCultivar = phenotype === "Cultivar";
            const valStr =
              pred.value === null
                ? "—"
                : isCultivar
                ? String(pred.value)
                : typeof pred.value === "number"
                ? pred.value.toFixed(4)
                : String(pred.value);

            const stdStr =
              pred.ensemble_std !== null && typeof pred.ensemble_std === "number"
                ? ` ± ${pred.ensemble_std.toFixed(4)}`
                : isCultivar && pred.vote_confidence
                ? ` (${(pred.vote_confidence * 100).toFixed(0)}%)`
                : "";

            return (
              <div
                key={phenotype}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-[#212121] border border-white/5 shadow-sm"
              >
                <span className="text-zinc-400 font-medium">{phenotype}:</span>
                <span className="font-semibold text-emerald-400 font-mono">
                  {valStr}
                  {stdStr && <span className="text-zinc-400 font-normal text-[11px]">{stdStr}</span>}
                </span>
              </div>
            );
          })}
        </div>

        <button
          onClick={() => setShowDetails(!showDetails)}
          className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-200 transition py-1 px-2 rounded-lg hover:bg-white/5 ml-auto"
        >
          <Layers className="w-3.5 h-3.5" />
          <span>{showDetails ? "Hide model details" : "Model output details"}</span>
          {showDetails ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>
      </div>

      {/* Expandable aggregate model output table */}
      {showDetails && (
        <div className="border-t border-white/10 bg-[#121212] p-3 overflow-x-auto">
          <table className="w-full text-left font-mono text-[11px]">
            <thead className="text-zinc-500 text-[10px] uppercase tracking-wider border-b border-white/5">
              <tr>
                <th className="pb-2 font-sans">Target</th>
                <th className="pb-2">Prediction (Mean)</th>
                <th className="pb-2">Ensemble Std</th>
                <th className="pb-2">Calibrated Interval</th>
                <th className="pb-2">Live Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-zinc-300">
              {entries.map(([phenotype, pred]) => (
                <tr key={phenotype}>
                  <td className="py-2 font-sans font-medium text-zinc-200">{phenotype}</td>
                  <td className="py-2 text-emerald-400 font-semibold">
                    {pred.value !== null && typeof pred.value === "number"
                      ? pred.value.toFixed(4)
                      : String(pred.value ?? "—")}
                  </td>
                  <td className="py-2 text-zinc-400">
                    {pred.ensemble_std !== null ? `± ${pred.ensemble_std.toFixed(4)}` : "—"}
                  </td>
                  <td className="py-2 text-zinc-500 italic">
                    {pred.interval_low !== null && pred.interval_high !== null
                      ? `[${pred.interval_low.toFixed(4)}, ${pred.interval_high.toFixed(4)}]`
                      : "null (unasserted)"}
                  </td>
                  <td className="py-2">
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">
                      {pred.calibration_status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
