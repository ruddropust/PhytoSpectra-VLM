import React from "react";
import { ModelStatus } from "../types";
import { X, Cpu, Layers } from "lucide-react";

interface ModelStatusModalProps {
  status: ModelStatus | null;
  isOpen: boolean;
  onClose: () => void;
}

export const ModelStatusModal: React.FC<ModelStatusModalProps> = ({
  status,
  isOpen,
  onClose,
}) => {
  if (!isOpen || !status) return null;

  const idcv = status.idcv_development_vlm;
  const idcvActive = Boolean(idcv?.enabled);
  const frozenStage40 = Boolean(idcv?.deployment_status?.startsWith("FROZEN_STAGE40S"));
  const frozenStage42 = Boolean(idcv?.deployment_status?.startsWith("FROZEN_STAGE42"));
  const researchV3 = Boolean(idcv?.deployment_status?.startsWith("RESEARCH_V3_STAGE42D"));
  const parameterEntries = Object.entries(status.learned_parameters || {});
  const formatLabel = (value: string) =>
    value
      .replace(/_/g, " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-xl w-full p-6 shadow-2xl space-y-6">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-2.5">
            <Cpu className="w-5 h-5 text-emerald-400" />
            <h3 className="text-base font-semibold text-white">
              PhytoSpectra-VLM System Registry
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Device Info */}
        <div className="grid grid-cols-2 gap-3 text-xs font-mono">
          <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800 space-y-1">
            <span className="text-slate-400 block font-sans">Execution Device</span>
            <span className="text-emerald-400 font-bold uppercase">{status.device}</span>
          </div>
          <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800 space-y-1">
            <span className="text-slate-400 block font-sans">GPU Acceleration</span>
            <span className="text-slate-200">{status.gpu_name || "CPU Standard Mode"}</span>
          </div>
        </div>

        {/* Parameter Count Breakdown */}
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Learned Scientific Architecture Parameters
          </h4>
          <div className="bg-slate-950/70 rounded-xl border border-slate-800 divide-y divide-slate-800/60 text-xs font-mono">
            {parameterEntries.map(([name, count]) => (
              <div className="flex justify-between p-3" key={name}>
                <span className="text-slate-300 font-sans">{formatLabel(name)}</span>
                <span className="text-slate-200">{count.toLocaleString()}</span>
              </div>
            ))}
            <div className="flex justify-between p-3 bg-emerald-950/30 font-bold">
              <span className="text-emerald-400 font-sans">Total Learned Parameter Count</span>
              <span className="text-emerald-400">{status.total_learned_parameters.toLocaleString()}</span>
            </div>
          </div>
        </div>

        <div className={`p-3.5 rounded-xl border space-y-2 text-xs ${
          idcvActive
            ? idcv.verified
              ? "bg-emerald-950/20 border-emerald-500/20"
              : "bg-red-950/20 border-red-500/20"
            : "bg-slate-950/70 border-slate-800"
        }`}>
          <div className="flex items-center justify-between gap-3">
            <span className="font-semibold text-slate-200">
              {researchV3
                ? "research_v3 Stage-42D VLM"
                : frozenStage42
                ? "Verified Stage-42 Contract VLM"
                : frozenStage40
                  ? "Frozen Stage-40 VLM Runtime"
                  : "ID-CV Vision-Language Runtime"}
            </span>
            <span className={`font-mono text-[10px] ${
              idcvActive && idcv.verified
                ? "text-emerald-400"
                : idcvActive
                  ? "text-red-300"
                  : "text-slate-500"
            }`}>
              {idcv?.deployment_status || "DISABLED"}
            </span>
          </div>
          <p className="text-slate-400 text-[11px] leading-relaxed">
            {idcv?.architecture || "Frozen PhytoCore + visual bridge + Qwen-LoRA grounded answer verifier"}
          </p>
          {idcvActive && (
            <div className="grid grid-cols-2 gap-2 font-mono text-[10px] text-slate-400">
              <span>Fold: {idcv.outer_fold ?? "—"}</span>
              <span>Seed: {idcv.model_seed ?? "—"}</span>
              <span>EMA: {idcv.ema_weights ? "active" : "not reported"}</span>
              <span>Vision: {idcv.vision_frozen ? "frozen" : "not reported"}</span>
              {(frozenStage40 || frozenStage42) && (
                <>
                  <span>Architecture lock: {idcv.architecture_freeze_verified ? "verified" : "failed"}</span>
                  <span>Protocol lock: {idcv.protocol_freeze_verified ? "verified" : "failed"}</span>
                </>
              )}
              {frozenStage42 && (
                <>
                  <span>Recipe lock: {idcv.recipe_freeze_verified ? "verified" : "failed"}</span>
                  <span>Outer evaluation: {idcv.outer_evaluation_verified ? "verified" : "failed"}</span>
                  <span>Images: {idcv.evaluated_images ?? "—"}</span>
                  <span>Questions: {idcv.evaluated_questions ?? "—"}</span>
                </>
              )}
              {researchV3 && (
                <>
                  <span>Campaign: 15/15 PASS</span>
                  <span>Evaluation: internal image-wise CV</span>
                </>
              )}
            </div>
          )}
          {idcv?.scope_warning && (
            <p className="text-amber-300 text-[10px] leading-relaxed">
              {idcv.scope_warning}
            </p>
          )}
          {idcv?.verification_error && (
            <p className="text-red-300 text-[10px] font-mono break-words">
              {idcv.verification_error}
            </p>
          )}
        </div>

        {/* Checkpoints & Inference Strategy */}
        <div className="p-3.5 bg-slate-950/70 rounded-xl border border-slate-800 space-y-1.5 text-xs">
          <div className="flex items-center gap-2 text-slate-200 font-semibold">
            <Layers className="w-4 h-4 text-emerald-400" />
            <span>
              Runtime Inference Strategy: {researchV3 ? "RESEARCH_V3 STAGE42D · SEED 2026 · FOLD 0" : frozenStage42 ? "STAGE-42 CONTRACT N-BEST · SEED 2026 · FOLD 0" : frozenStage40 ? "FROZEN STAGE-40S FOLD-0" : idcvActive ? "ID-CV DEVELOPMENT VLM" : status.scientific_ensemble_mode.toUpperCase()}
            </span>
          </div>
          <p className="text-slate-400 text-[11px] leading-relaxed">
            {researchV3 ? "One selected PhytoCore fold model is used for live inference." : `Loaded ${status.phytocore_checkpoints_loaded} legacy PhytoCore checkpoint(s).`}
            {idcvActive
              ? ` Image-derived answers require the separately verified Fold-${idcv.outer_fold ?? 0} ID-CV runtime.`
              : " The ID-CV development runtime is intentionally disabled."}
          </p>
        </div>

        <div className="flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-semibold transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
