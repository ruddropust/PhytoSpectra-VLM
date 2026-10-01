import React, { useRef, useEffect } from "react";
import { ChatMessage, UploadResult } from "../types";
import { MessageCard } from "./MessageCard";
import { ArrowUp, Paperclip, Sparkles, FileImage, X } from "lucide-react";

interface ChatInterfaceProps {
  messages: ChatMessage[];
  uploadData: UploadResult | null;
  activeFileName: string | null;
  inputPrompt: string;
  isGenerating: boolean;
  isUploading: boolean;
  onInputChange: (val: string) => void;
  onSendMessage: (text?: string) => void;
  onFileUpload: (file: File) => void;
  onRemoveUpload?: () => void;
  onOpenLightbox?: (url: string, title: string) => void;
}

const PROMPT_CARDS = [
  {
    title: "Estimate Stomatal Conductance",
    desc: "Calculate Gsw with 15-model cross-validation ensemble",
    prompt: "Estimate Gsw from this leaf sample.",
    icon: "🍃",
  },
  {
    title: "Predict Grain Weight",
    desc: "Estimate grain weight and analyze fertilizer relationship",
    prompt: "What is the predicted grain weight for this sample?",
    icon: "🌾",
  },
  {
    title: "Identify Wheat Cultivar",
    desc: "Classify variety (Heerup, Kvium, Rembrandt, Sheriff)",
    prompt: "Predict the wheat cultivar for this sample.",
    icon: "🧬",
  },
  {
    title: "Visualize Gsw Evidence",
    desc: "View the target heatmap, selected patches, focused 4× crop, and spectral tokens",
    prompt: "Show a cropped and zoomed visual explanation for Gsw.",
    icon: "🔍",
  },
];

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  uploadData,
  activeFileName,
  inputPrompt,
  isGenerating,
  isUploading,
  onInputChange,
  onSendMessage,
  onFileUpload,
  onRemoveUpload,
  onOpenLightbox,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isGenerating]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(Math.max(textareaRef.current.scrollHeight, 24), 160)}px`;
    }
  }, [inputPrompt]);

  const handlePaste = (e: React.ClipboardEvent) => {
    if (e.clipboardData.files && e.clipboardData.files.length > 0) {
      const file = e.clipboardData.files[0];
      const name = file.name.toLowerCase();
      if (
        name.endsWith(".tif") ||
        name.endsWith(".tiff") ||
        file.type.includes("tiff") ||
        file.type.includes("image")
      ) {
        e.preventDefault();
        onFileUpload(file);
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  const handleSend = (text?: string) => {
    const q = (text || inputPrompt).trim();
    if (!q) return;
    onSendMessage(q);
  };

  const handleCardClick = (promptText: string) => {
    onSendMessage(promptText);
  };

  return (
    <main
      onPaste={handlePaste}
      className="flex-1 flex flex-col h-full bg-[#212121] overflow-hidden relative select-text"
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".tif,.tiff"
        className="hidden"
      />

      {/* Scrollable Conversation View */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-2xl mx-auto space-y-8 py-8 animate-fade-in">
            {/* Clean ChatGPT Header */}
            <div className="space-y-3">
              <div className="w-12 h-12 rounded-2xl bg-emerald-500/15 border border-emerald-500/25 flex items-center justify-center text-emerald-400 mx-auto shadow-md">
                <Sparkles className="w-6 h-6" />
              </div>
              <h2 className="text-2xl font-semibold text-zinc-100 tracking-tight">
                What would you like to analyze today?
              </h2>
              <p className="text-xs text-zinc-400 max-w-md mx-auto leading-relaxed">
                WheatSpectraChat uses the 13.93M parameter PhytoSpectra-VLM architecture to provide grounded
                hyperspectral reasoning for wheat phenotyping.
              </p>
            </div>

            {/* Prompt Cards (Clean body without upload dropzone) */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-xl">
              {PROMPT_CARDS.map((card) => (
                <button
                  key={card.title}
                  onClick={() => handleCardClick(card.prompt)}
                  disabled={isGenerating}
                  className="p-4 rounded-2xl bg-[#171717] hover:bg-[#282828] border border-white/5 hover:border-white/15 text-left transition flex flex-col justify-between gap-2 shadow-sm group"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-base">{card.icon}</span>
                    <span className="text-xs font-semibold text-zinc-200 group-hover:text-white">
                      {card.title}
                    </span>
                  </div>
                  <p className="text-[11px] text-zinc-400 leading-snug">
                    {card.desc}
                  </p>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto w-full pb-8">
            {messages.map((msg) => (
              <MessageCard
                key={msg.id}
                message={msg}
                onOpenLightbox={onOpenLightbox}
              />
            ))}

            {isGenerating && (
              <div className="flex items-center gap-3 py-4 text-xs text-zinc-400 animate-pulse">
                <div className="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
                <span>Thinking & Grounding Reasoning...</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Floating Bottom Input Bar (ChatGPT style with 📎 attachment button) */}
      <div className="p-4 bg-gradient-to-t from-[#212121] via-[#212121]/95 to-transparent shrink-0">
        <div className="max-w-3xl mx-auto space-y-2">
          {/* Active sample chip above input */}
          {uploadData && (
            <div className="flex items-center gap-2 px-3 py-1 bg-[#171717] border border-white/10 rounded-full w-fit text-xs text-zinc-300 shadow-sm animate-fade-in">
              <FileImage className="w-3.5 h-3.5 text-emerald-400" />
              <span className="font-medium truncate max-w-xs">
                {activeFileName || `Sample-${uploadData.image_id.slice(0, 6)}`}
              </span>
              <span className="text-[10px] text-zinc-500 font-mono">
                ({uploadData.canonical_shape.join("×")})
              </span>
              {onRemoveUpload && (
                <button
                  onClick={onRemoveUpload}
                  className="p-0.5 rounded-full hover:bg-white/10 text-zinc-400 hover:text-white transition ml-1"
                  title="Remove sample"
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>
          )}

          {/* Clean Input Pill Container */}
          <div className="flex items-center gap-2 bg-[#2f2f2f] border border-white/10 focus-within:border-white/25 rounded-3xl p-2 px-4 shadow-lg transition">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading || isGenerating}
              className={`p-2 rounded-full transition shrink-0 ${
                uploadData
                  ? "text-emerald-400 hover:bg-emerald-500/10"
                  : "text-zinc-400 hover:text-white hover:bg-white/10"
              }`}
              title={uploadData ? "Change attached TIFF sample" : "Attach HyperLeaf2024 TIFF sample"}
            >
              {isUploading ? (
                <div className="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
              ) : (
                <Paperclip className="w-4 h-4" />
              )}
            </button>

            <textarea
              ref={textareaRef}
              value={inputPrompt}
              onChange={(e) => onInputChange(e.target.value)}
              onKeyDown={handleKeyDown}
              onPaste={handlePaste}
              placeholder={
                uploadData
                  ? "Ask about GrainWeight, Gsw, PhiPS2, Fertilizer, Cultivar..."
                  : "Attach a TIFF sample (📎) or ask a question to begin..."
              }
              disabled={isGenerating}
              rows={1}
              className="flex-1 bg-transparent border-0 resize-none text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none py-1.5"
            />

            <button
              onClick={() => handleSend()}
              disabled={isGenerating || !inputPrompt.trim()}
              aria-label="Send message"
              title="Send message"
              className="w-8 h-8 rounded-full bg-white text-black hover:bg-zinc-200 disabled:bg-zinc-700 disabled:text-zinc-500 flex items-center justify-center transition shrink-0"
            >
              {isGenerating ? (
                <div className="w-3.5 h-3.5 border-2 border-black border-t-transparent rounded-full animate-spin" />
              ) : (
                <ArrowUp className="w-4 h-4 stroke-[2.5]" />
              )}
            </button>
          </div>

          <p className="text-[10px] text-zinc-500 text-center font-sans">
            Ask scientific concept questions without an image. Image-derived outputs require an attached TIFF sample.
          </p>
        </div>
      </div>
    </main>
  );
};
