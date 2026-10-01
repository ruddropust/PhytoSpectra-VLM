import React from "react";
import { ModelStatus } from "../types";
import { PanelLeftOpen, Cpu, Sparkles } from "lucide-react";

interface HeaderProps {
  status: ModelStatus | null;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  onOpenStatusModal: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  status,
  sidebarOpen,
  onToggleSidebar,
  onOpenStatusModal,
}) => {
  return (
    <header className="h-14 border-b border-white/10 bg-[#212121] px-4 flex items-center justify-between shrink-0 select-none">
      <div className="flex items-center gap-2">
        {!sidebarOpen && (
          <button
            onClick={onToggleSidebar}
            className="p-2 text-zinc-400 hover:text-white rounded-lg hover:bg-white/5 transition mr-1"
            title="Open sidebar"
          >
            <PanelLeftOpen className="w-5 h-5" />
          </button>
        )}

        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <Sparkles className="w-4 h-4" />
          </div>
          <span className="font-semibold text-sm text-zinc-100 tracking-tight">
            WheatSpectraChat
          </span>
          <span className="text-[11px] font-mono text-zinc-500 bg-[#171717] px-2 py-0.5 rounded-full border border-white/5">
            PhytoSpectra-VLM
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onOpenStatusModal}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-white/10 bg-[#171717] hover:bg-[#282828] text-zinc-300 text-xs transition"
          title="View architecture registry"
        >
          <Cpu className="w-3.5 h-3.5 text-emerald-400" />
          <span className="font-mono text-[11px]">
            {status ? `${status.device.toUpperCase()} • 13.93M` : "Online"}
          </span>
        </button>
      </div>
    </header>
  );
};
