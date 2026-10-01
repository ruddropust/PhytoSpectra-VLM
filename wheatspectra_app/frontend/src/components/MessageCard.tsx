import React, { useState } from "react";
import { ChatMessage } from "../types";
import { ResponseBadge } from "./ResponseBadge";
import { PredictionCard } from "./PredictionCard";
import { EvidencePanel } from "./EvidencePanel";
import { Disclaimers } from "./Disclaimers";
import { Sparkles, User, Copy, Check } from "lucide-react";

interface MessageCardProps {
  message: ChatMessage;
  onOpenLightbox?: (url: string, title: string) => void;
}

const copyTextToClipboard = async (text: string): Promise<boolean> => {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (err) {
    // fallback to execCommand below
  }

  try {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.position = "fixed";
    textArea.style.left = "-999999px";
    textArea.style.top = "-999999px";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    const successful = document.execCommand("copy");
    textArea.remove();
    return successful;
  } catch (err) {
    return false;
  }
};

export const MessageCard: React.FC<MessageCardProps> = ({ message, onOpenLightbox }) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.sender === "user";

  const handleCopy = async (textToCopy: string) => {
    const success = await copyTextToClipboard(textToCopy);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (isUser) {
    return (
      <div className="flex justify-end mb-6 group select-text">
        <div className="flex items-start gap-3 max-w-2xl">
          <div className="flex items-center gap-1.5 self-center opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={() => handleCopy(message.text || "")}
              className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-[#2f2f2f] transition"
              title="Copy message"
              aria-label="Copy message"
            >
              {copied ? (
                <span className="flex items-center gap-1 text-[11px] text-emerald-400 font-sans">
                  <Check className="w-3.5 h-3.5" /> Copied
                </span>
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
            </button>
          </div>
          <div className="bg-[#2f2f2f] text-zinc-100 rounded-3xl px-5 py-3 shadow-sm select-text">
            <p className="text-sm font-normal leading-relaxed select-text">{message.text}</p>
          </div>
          <div className="w-8 h-8 rounded-full bg-[#2f2f2f] border border-white/10 flex items-center justify-center text-zinc-300 shrink-0">
            <User className="w-4 h-4" />
          </div>
        </div>
      </div>
    );
  }

  const response = message.response;
  if (!response) return null;

  const selectedPhenotypes = new Set(response.selected_phenotypes || []);
  const visiblePredictions =
    selectedPhenotypes.size > 0
      ? Object.fromEntries(
          Object.entries(response.predictions || {}).filter(([target]) =>
            selectedPhenotypes.has(target)
          )
        )
      : response.predictions;
  const hasPredictions =
    visiblePredictions && Object.keys(visiblePredictions).length > 0;
  const showsImageEvidence = [
    "IMAGE_SUMMARY",
    "MODEL_PREDICTION",
    "EVIDENCE_EXPLANATION",
  ].includes(response.answer_type);

  return (
    <div className="flex justify-start mb-8 animate-fade-in group select-text">
      <div className="flex items-start gap-4 max-w-3xl w-full">
        {/* ChatGPT style assistant avatar */}
        <div className="w-8 h-8 rounded-full bg-emerald-600/20 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shrink-0 shadow-sm mt-0.5">
          <Sparkles className="w-4 h-4" />
        </div>

        <div className="flex-1 space-y-3 min-w-0">
          {/* Top meta pill + Copy button */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ResponseBadge answerType={response.answer_type} />
              <span className="text-[11px] text-zinc-500 font-mono">
                intent: {response.intent}
              </span>
              {response.development_warning && (
                <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wide rounded text-amber-500 bg-amber-500/10 border border-amber-500/20 uppercase">
                  Dev Fold-0
                </span>
              )}
              {response.verification?.fallback_used === true ? (
                <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wide rounded text-amber-300 bg-amber-500/10 border border-amber-500/20 uppercase">
                  Verified fallback
                </span>
              ) : response.answer_mode === "EVIDENCE_BOUNDED" ? (
                <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wide rounded text-violet-300 bg-violet-500/10 border border-violet-500/20 uppercase">
                  Evidence bounded
                </span>
              ) : null}
            </div>

            <button
              onClick={() => handleCopy(response.answer)}
              className="flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] text-zinc-400 hover:text-zinc-200 hover:bg-[#2a2a2a] transition"
              title="Copy answer"
              aria-label="Copy answer"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400 font-medium">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>

          {/* Clean Markdown-like Answer text */}
          <div className="text-zinc-100 text-sm leading-relaxed whitespace-pre-wrap font-sans select-text">
            {response.answer}
          </div>

          {/* Dedicated Head Predictions */}
          {hasPredictions && showsImageEvidence && (
            <PredictionCard predictions={visiblePredictions} />
          )}

          {/* Vision Evidence & Verification */}
          {showsImageEvidence && (
            <EvidencePanel
              visuals={response.visuals}
              verification={response.verification}
              evidence={response.evidence}
              warnings={response.warnings}
              onOpenLightbox={onOpenLightbox}
            />
          )}

          {/* Scope Reason for Abstention */}
          {["CLARIFICATION", "IMAGE_REQUIRED", "UNSUPPORTED", "RUNTIME_ERROR"].includes(response.answer_type) && response.scope.reason && (
            <div className="p-3 rounded-xl border border-white/5 bg-[#171717] text-xs text-zinc-400 font-mono select-text">
              <span className="text-zinc-200 font-medium font-sans">Scope Gate Reason: </span>
              {response.scope.reason}
            </div>
          )}

          {/* Disclaimers */}
          {showsImageEvidence && <Disclaimers />}
        </div>
      </div>
    </div>
  );
};
