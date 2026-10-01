import React from "react";
import { VerificationDetails } from "../types";
import { ShieldCheck, Info } from "lucide-react";

interface VerificationTabProps {
  verification: VerificationDetails;
  evidence: Record<string, any>;
  warnings: string[];
}

export const VerificationTab: React.FC<VerificationTabProps> = ({
  verification,
  evidence,
  warnings,
}) => {
  const phenoEntries = Object.entries(evidence);
  const policyLabel =
    verification.response_policy === "VALIDATED"
      ? "DUAL-EVIDENCE PASSED"
      : verification.response_policy === "EVIDENCE_BOUNDED"
        ? "EVIDENCE-LIMITED"
        : "UNAVAILABLE";

  return (
    <div className="p-4 space-y-3.5 text-xs text-zinc-300 font-sans">
      <div className="flex flex-wrap items-center gap-2 pb-2 border-b border-white/5">
        <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#1f1f1f] border border-white/5 text-zinc-300">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span>Internal check: <strong className="text-white">{policyLabel}</strong></span>
        </span>

        <span className="px-2.5 py-1 rounded-lg bg-[#1f1f1f] border border-white/5 text-zinc-300">
          Dual Evidence: <strong className={verification.all_dual_verified ? "text-emerald-400" : "text-amber-400"}>
            {verification.evidence_status}
          </strong>
        </span>
      </div>

      {/* Per-Phenotype Verification Breakdown */}
      {phenoEntries.length > 0 && (
        <div className="space-y-2">
          {phenoEntries.map(([pheno, ev]) => {
            const sp = ev.spatial || {};
            const lp = ev.spectral || {};
            
            // For Stage-30 IDCV development mode, spatial and spectral tokens might be directly on ev
            const spTokens = sp.selected_token_indices?.length ?? ev.spatial_indices?.length ?? 0;
            const lpTokens = lp.selected_token_indices?.length ?? ev.spectral_indices?.length ?? 0;
            const isIdcvMode = ev.spatial_indices !== undefined;

            return (
              <div
                key={pheno}
                className="p-3 rounded-xl bg-[#1a1a1a] border border-white/5 space-y-1.5 font-mono text-[11px]"
              >
                <div className="flex justify-between items-center text-zinc-200 font-sans font-medium text-xs">
                  <span>{pheno} Counterfactual Grounding</span>
                  <span className="text-zinc-500 text-[10px]">
                    Spatial k={spTokens} | Spectral k={lpTokens}
                  </span>
                </div>

                {!isIdcvMode && (
                  <div className="grid grid-cols-2 gap-3 text-zinc-400 pt-1">
                    <div className="space-y-0.5">
                      <div className="text-zinc-500 font-sans text-[10px]">Spatial Patch Tokens</div>
                      <div>Comp Gain: <span className="text-zinc-200">{sp.comprehensiveness_gain?.toFixed(4) ?? "—"}</span></div>
                      <div>Suff Adv: <span className="text-zinc-200">{sp.sufficiency_advantage?.toFixed(4) ?? "—"}</span></div>
                    </div>
                    <div className="space-y-0.5">
                      <div className="text-zinc-500 font-sans text-[10px]">Spectral Context Tokens</div>
                      <div>Comp Gain: <span className="text-zinc-200">{lp.comprehensiveness_gain?.toFixed(4) ?? "—"}</span></div>
                      <div>Suff Adv: <span className="text-zinc-200">{lp.sufficiency_advantage?.toFixed(4) ?? "—"}</span></div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* IDCV Stage 33 Grounded Verifier Results */}
      {verification.raw_answer_accepted !== undefined && (
        <div className="p-3 rounded-xl bg-[#1a1a1a] border border-emerald-500/10 space-y-2">
          <div className="flex justify-between items-center pb-2 border-b border-white/5">
            <span className="text-emerald-400 font-medium">Stage-33 Grounded Verifier</span>
            <span className="text-zinc-500 text-[10px] font-mono tracking-wider">
              {verification.grounding_source || "QWEN_VLM"}
            </span>
          </div>
          
          <div className="grid grid-cols-2 gap-3 text-zinc-400 font-mono text-[11px]">
            <div className="space-y-1">
              <div className="flex justify-between">
                <span>Raw output accepted:</span>
                <strong className={verification.raw_answer_accepted ? "text-emerald-400" : "text-amber-400"}>
                  {verification.raw_answer_accepted ? "YES" : "NO"}
                </strong>
              </div>
              <div className="flex justify-between">
                <span>Fallback triggered:</span>
                <strong className={verification.fallback_used ? "text-amber-400" : "text-zinc-500"}>
                  {verification.fallback_used ? "YES" : "NO"}
                </strong>
              </div>
            </div>
            <div className="space-y-1">
              <div className="flex justify-between">
                <span>EMA Weights:</span>
                <strong className={verification.ema_weights ? "text-emerald-400" : "text-zinc-500"}>
                  {verification.ema_weights ? "ACTIVE" : "NONE"}
                </strong>
              </div>
              <div className="flex justify-between">
                <span>Vision frozen:</span>
                <strong className={verification.vision_frozen ? "text-emerald-400" : "text-zinc-500"}>
                  {verification.vision_frozen ? "YES" : "NO"}
                </strong>
              </div>
            </div>
          </div>
          
          {verification.verification_reasons && verification.verification_reasons.length > 0 && (
            <div className="pt-2">
              <div className="text-zinc-500 font-sans text-[10px] mb-1">Verification Log</div>
              <ul className="space-y-1 list-disc list-inside text-[11px] text-zinc-400">
                {verification.verification_reasons.map((reason, i) => (
                  <li key={i} className="leading-snug">{reason}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 space-y-1 text-[11px]">
          <div className="font-semibold flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5" />
            <span>Dataset Correlation Notice</span>
          </div>
          {warnings.map((w, i) => (
            <p key={i} className="leading-relaxed opacity-90">
              {w}
            </p>
          ))}
        </div>
      )}
    </div>
  );
};
