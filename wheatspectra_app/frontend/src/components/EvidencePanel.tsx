import React, { useEffect, useMemo, useState } from "react";
import { TargetVisualExplanation, VisualsPayload, VerificationDetails } from "../types";
import { VerificationTab } from "./VerificationTab";
import { Focus, Maximize2, ShieldCheck } from "lucide-react";

interface EvidencePanelProps {
  visuals: VisualsPayload;
  verification: VerificationDetails;
  evidence: Record<string, any>;
  warnings: string[];
  onOpenLightbox?: (url: string, title: string) => void;
}

type EvidenceTab = "composite" | "heatmap" | "marked" | "crop" | "zoom" | "spectral" | "bundle";

const captionForTab = (tab: EvidenceTab): string => {
  switch (tab) {
    case "heatmap":
      return "Relative PCTG spatial-token sensitivity for this target and sample; colors are not probabilities.";
    case "marked":
      return "The yellow box marks the frozen PhytoCore direct-sensitivity candidate region.";
    case "crop":
      return "Unscaled target-conditioned crop of the selected HSI composite region.";
    case "zoom":
      return "4× pixel-preserving crop of the strongest connected target-specific evidence region.";
    case "spectral":
      return "Model-internal spectral-token sensitivity by token index; physical wavelengths in nm are not asserted.";
    case "bundle":
      return "Text-labelled visual bundle linking the model output to its spatial and spectral representation.";
    default:
      return "False-color HSI composite for visual orientation; this is not a conventional RGB photograph.";
  }
};

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  visuals,
  verification,
  evidence,
  warnings,
  onOpenLightbox,
}) => {
  const targetEntries = Object.entries(visuals.target_visuals || {});
  const targetNames = targetEntries.map(([target]) => target);
  const [activeTarget, setActiveTarget] = useState<string>(targetNames[0] || "");
  const [activeTab, setActiveTab] = useState<EvidenceTab>(
    targetNames.length > 0 ? "heatmap" : "composite"
  );
  const [showVerification, setShowVerification] = useState<boolean>(false);

  useEffect(() => {
    if (targetNames.length > 0 && !targetNames.includes(activeTarget)) {
      setActiveTarget(targetNames[0]);
    }
  }, [activeTarget, targetNames.join("|")]);

  useEffect(() => {
    if (targetNames.length > 0) setActiveTab("heatmap");
  }, [activeTarget]);

  const activeVisual: TargetVisualExplanation | undefined =
    (visuals.target_visuals || {})[activeTarget];

  const availableTabs = useMemo(() => {
    const tabs: { key: EvidenceTab; label: string; url?: string }[] = [];
    if (activeVisual?.heatmap_url) {
      tabs.push({ key: "heatmap", label: "Target Heatmap", url: activeVisual.heatmap_url });
    }
    if (activeVisual?.marked_url || (!activeVisual && visuals.marked_url)) {
      tabs.push({
        key: "marked",
        label: "Selected Patches",
        url: activeVisual?.marked_url || visuals.marked_url,
      });
    }
    if (activeVisual?.crop_url) {
      tabs.push({
        key: "crop",
        label: "ROI Crop",
        url: activeVisual.crop_url,
      });
    }
    if (activeVisual?.zoom_url || (!activeVisual && visuals.zoom_url)) {
      tabs.push({
        key: "zoom",
        label: "Focused 4× Crop",
        url: activeVisual?.zoom_url || visuals.zoom_url,
      });
    }
    if (activeVisual?.spectral_plot_url || (!activeVisual && visuals.spectral_plot_url)) {
      tabs.push({
        key: "spectral",
        label: "Spectral Tokens",
        url: activeVisual?.spectral_plot_url || visuals.spectral_plot_url,
      });
    }
    if (activeVisual?.bundle_url || (!activeVisual && visuals.bundle_url)) {
      tabs.push({
        key: "bundle",
        label: "Text + Visual Bundle",
        url: activeVisual?.bundle_url || visuals.bundle_url,
      });
    }
    if (tabs.length === 0 && visuals.composite_url) {
      tabs.push({ key: "composite", label: "HSI Composite", url: visuals.composite_url });
    }
    return tabs;
  }, [activeVisual, visuals]);

  if (availableTabs.length === 0) {
    return (
      <div className="my-3 rounded-2xl border border-white/10 bg-[#171717] overflow-hidden text-xs">
        <div className="px-3 py-2.5 border-b border-white/5 bg-[#141414] flex items-center gap-2 text-zinc-200 font-medium">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span>Verification details</span>
          <span className="ml-auto text-[10px] font-mono text-amber-300">
            Visual assets unavailable
          </span>
        </div>
        <VerificationTab
          verification={verification}
          evidence={evidence}
          warnings={warnings}
        />
      </div>
    );
  }
  const currentTab = availableTabs.find((tab) => tab.key === activeTab) || availableTabs[0];

  return (
    <div className="my-3 rounded-2xl border border-white/10 bg-[#171717] overflow-hidden text-xs">
      {targetNames.length > 0 && (
        <div className="px-3 pt-3 pb-2 border-b border-white/5 bg-[#141414] space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider text-zinc-500 font-semibold mr-1">
              Task output
            </span>
            {targetNames.map((target) => (
              <button
                key={target}
                onClick={() => setActiveTarget(target)}
                className={`px-2.5 py-1 rounded-lg text-[11px] font-medium border transition ${
                  activeTarget === target
                    ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
                    : "bg-[#1f1f1f] text-zinc-400 border-white/5 hover:text-zinc-200"
                }`}
              >
                {target}
              </button>
            ))}
          </div>

          {activeVisual && (
            <div className="rounded-xl bg-[#1d1d1d] border border-white/5 px-3 py-2.5 space-y-1.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-zinc-100 font-semibold">
                  <Focus className="w-3.5 h-3.5 text-yellow-300" />
                  <span>{activeVisual.prediction_text}</span>
                </div>
                <span className={`text-[10px] font-medium ${
                  activeVisual.evidence_available ? "text-emerald-300" : "text-amber-300"
                }`}>
                  {activeVisual.evidence_available
                    ? `${activeVisual.selected_token_count} selected patches`
                    : "Spatial evidence unavailable"}
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 leading-relaxed">
                {activeVisual.spatial_summary}
              </p>
            </div>
          )}
        </div>
      )}

      <div className="px-3 pt-2.5 flex items-center justify-between gap-3 border-b border-white/5 bg-[#141414]">
        <div className="flex items-center gap-1 overflow-x-auto pb-2">
          {availableTabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition whitespace-nowrap ${
                currentTab.key === tab.key
                  ? "bg-[#242424] text-white shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-white/5"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <button
          onClick={() => setShowVerification(!showVerification)}
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium transition whitespace-nowrap mb-2 ${
            showVerification
              ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
              : "text-zinc-400 hover:text-zinc-200 hover:bg-white/5"
          }`}
        >
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>Verification</span>
        </button>
      </div>

      {currentTab.url && (
        <div className="p-4 flex flex-col items-center justify-center bg-black/40 relative group">
          <img
            src={currentTab.url}
            alt={`${activeTarget || "HSI"} ${currentTab.label}`}
            className="max-h-80 w-auto object-contain rounded-xl shadow-md cursor-pointer hover:opacity-95 transition"
            onClick={() => onOpenLightbox?.(currentTab.url!, currentTab.label)}
          />
          <button
            onClick={() => onOpenLightbox?.(currentTab.url!, currentTab.label)}
            className="absolute top-6 right-6 p-2 rounded-xl bg-black/70 hover:bg-black/90 text-white backdrop-blur border border-white/10 opacity-0 group-hover:opacity-100 transition shadow-lg"
            title="Expand fullscreen"
            aria-label="Expand evidence image"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
          <span className="mt-2 text-[10px] text-zinc-500 font-mono text-center max-w-2xl">
            {captionForTab(currentTab.key)} Click image to enlarge.
          </span>
        </div>
      )}

      {showVerification && (
        <div className="border-t border-white/10 bg-[#121212]">
          <VerificationTab
            verification={verification}
            evidence={evidence}
            warnings={warnings}
          />
        </div>
      )}
    </div>
  );
};
