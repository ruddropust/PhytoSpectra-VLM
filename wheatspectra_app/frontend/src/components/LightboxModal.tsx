import React from "react";
import { X, ZoomIn } from "lucide-react";

interface LightboxModalProps {
  image: { url: string; title: string } | null;
  onClose: () => void;
}

export const LightboxModal: React.FC<LightboxModalProps> = ({ image, onClose }) => {
  if (!image) return null;

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-4 animate-fade-in"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="relative max-w-4xl w-full bg-[#1e1e1e] border border-white/10 rounded-2xl overflow-hidden shadow-2xl flex flex-col"
      >
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/10 bg-[#171717]">
          <div className="flex items-center gap-2 text-sm font-medium text-zinc-200">
            <ZoomIn className="w-4 h-4 text-emerald-400" />
            <span>{image.title}</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-zinc-400 hover:text-white hover:bg-white/10 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-4 flex items-center justify-center bg-black/50 overflow-auto max-h-[80vh]">
          <img
            src={image.url}
            alt={image.title}
            className="max-h-[72vh] w-auto object-contain rounded-lg shadow-lg"
          />
        </div>
      </div>
    </div>
  );
};
