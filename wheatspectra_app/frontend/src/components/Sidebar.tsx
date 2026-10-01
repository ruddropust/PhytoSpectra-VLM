import React from "react";
import { ConversationSession, ModelStatus } from "../types";
import { Plus, MessageSquare, Trash2, Cpu, PanelLeftClose } from "lucide-react";

interface SidebarProps {
  conversations: ConversationSession[];
  activeConversationId: string;
  modelStatus: ModelStatus | null;
  isOpen: boolean;
  onToggle: () => void;
  onNewChat: () => void;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string, e: React.MouseEvent) => void;
  onOpenStatusModal: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  conversations,
  activeConversationId,
  modelStatus,
  isOpen,
  onToggle,
  onNewChat,
  onSelectConversation,
  onDeleteConversation,
  onOpenStatusModal,
}) => {
  if (!isOpen) return null;

  return (
    <aside className="w-64 border-r border-white/10 bg-[#171717] p-3 flex flex-col justify-between shrink-0 h-full select-none">
      {/* Top Header: New Chat & Toggle */}
      <div className="space-y-4 flex-1 flex flex-col min-h-0">
        <div className="flex items-center justify-between">
          <button
            onClick={onNewChat}
            className="flex-1 flex items-center gap-2 px-3 py-2 rounded-xl bg-[#212121] hover:bg-[#2a2a2a] border border-white/5 text-sm text-zinc-200 font-medium transition shadow-sm group"
          >
            <Plus className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition" />
            <span>New Chat</span>
          </button>

          <button
            onClick={onToggle}
            className="p-2 ml-2 text-zinc-400 hover:text-white rounded-lg hover:bg-white/5 transition"
            title="Close sidebar"
          >
            <PanelLeftClose className="w-5 h-5" />
          </button>
        </div>

        {/* Conversation List (ONLY conversation list as requested) */}
        <div className="flex-1 flex flex-col min-h-0 space-y-1">
          <div className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider px-2 py-1">
            Chats
          </div>

          <div className="flex-1 overflow-y-auto space-y-1 pr-1">
            {conversations.map((conv) => {
              const isActive = conv.id === activeConversationId;
              return (
                <div
                  key={conv.id}
                  onClick={() => onSelectConversation(conv.id)}
                  className={`group flex items-center justify-between px-3 py-2.5 rounded-xl cursor-pointer text-xs transition ${
                    isActive
                      ? "bg-[#212121] text-white font-medium border border-white/5"
                      : "text-zinc-400 hover:text-zinc-200 hover:bg-[#212121]/50"
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0 flex-1">
                    <MessageSquare className={`w-3.5 h-3.5 shrink-0 ${isActive ? "text-emerald-400" : "text-zinc-500"}`} />
                    <span className="truncate">{conv.title}</span>
                  </div>

                  <button
                    onClick={(e) => onDeleteConversation(conv.id, e)}
                    className="p-1 text-zinc-500 hover:text-rose-400 opacity-0 group-hover:opacity-100 transition rounded shrink-0 ml-1"
                    title="Delete chat"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Bottom Model Status */}
      <div className="pt-3 border-t border-white/10 mt-auto">
        <button
          onClick={onOpenStatusModal}
          className="w-full flex items-center justify-between p-2.5 rounded-xl bg-[#212121] hover:bg-[#2a2a2a] text-xs text-zinc-300 transition border border-white/5"
        >
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="font-mono text-[11px] text-zinc-200">
              {modelStatus
                ? `${modelStatus.device.toUpperCase()} • ${(modelStatus.total_learned_parameters / 1_000_000).toFixed(2)}M`
                : "System"}
            </span>
          </div>
          <Cpu className="w-4 h-4 text-zinc-500" />
        </button>
      </div>
    </aside>
  );
};
