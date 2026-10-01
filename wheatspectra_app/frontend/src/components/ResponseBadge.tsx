import React from "react";
import { AnswerType } from "../types";

interface ResponseBadgeProps {
  answerType: AnswerType;
}

const BADGES: Record<AnswerType, { label: string; className: string; dot: string }> = {
  CONCEPT_EXPLANATION: {
    label: "Scientific concept",
    className: "bg-sky-500/10 text-sky-300 border-sky-500/20",
    dot: "bg-sky-300",
  },
  IMAGE_SUMMARY: {
    label: "Image summary",
    className: "bg-cyan-500/10 text-cyan-300 border-cyan-500/20",
    dot: "bg-cyan-300",
  },
  MODEL_PREDICTION: {
    label: "Model prediction",
    className: "bg-emerald-500/10 text-emerald-300 border-emerald-500/20",
    dot: "bg-emerald-300",
  },
  EVIDENCE_EXPLANATION: {
    label: "Model evidence",
    className: "bg-violet-500/10 text-violet-300 border-violet-500/20",
    dot: "bg-violet-300",
  },
  CLARIFICATION: {
    label: "Clarification",
    className: "bg-amber-500/10 text-amber-300 border-amber-500/20",
    dot: "bg-amber-300",
  },
  IMAGE_REQUIRED: {
    label: "Image required",
    className: "bg-amber-500/10 text-amber-300 border-amber-500/20",
    dot: "bg-amber-300",
  },
  UNSUPPORTED: {
    label: "Unavailable",
    className: "bg-zinc-700/40 text-zinc-400 border-zinc-700",
    dot: "bg-zinc-400",
  },
  RUNTIME_ERROR: {
    label: "Runtime unavailable",
    className: "bg-red-500/10 text-red-300 border-red-500/20",
    dot: "bg-red-300",
  },
};

export const ResponseBadge: React.FC<ResponseBadgeProps> = ({ answerType }) => {
  const badge = BADGES[answerType] || BADGES.UNSUPPORTED;
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium border ${badge.className}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${badge.dot}`} />
      {badge.label}
    </span>
  );
};
