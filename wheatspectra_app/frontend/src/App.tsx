import React from "react";
import { Header } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { ChatInterface } from "./components/ChatInterface";
import { ModelStatusModal } from "./components/ModelStatusModal";
import { LightboxModal } from "./components/LightboxModal";
import { useWheatChat } from "./hooks/useWheatChat";
import { AlertCircle, X } from "lucide-react";

export const App: React.FC = () => {
  const {
    modelStatus,
    conversations,
    activeConversationId,
    uploadData,
    activeFileName,
    messages,
    inputPrompt,
    isUploading,
    isGenerating,
    isStatusModalOpen,
    sidebarOpen,
    lightboxImage,
    errorMessage,
    setInputPrompt,
    setIsStatusModalOpen,
    setSidebarOpen,
    setLightboxImage,
    setErrorMessage,
    handleFileUpload,
    handleRemoveUpload,
    handleSendMessage,
    handleNewChat,
    handleSelectConversation,
    handleDeleteConversation,
  } = useWheatChat();

  return (
    <div className="h-screen w-screen flex flex-col bg-[#212121] text-[#ececec] overflow-hidden font-sans">
      {/* Top Header */}
      <Header
        status={modelStatus}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        onOpenStatusModal={() => setIsStatusModalOpen(true)}
      />

      {/* Global Error Banner */}
      {errorMessage && (
        <div className="bg-rose-500/15 border-b border-rose-500/30 px-6 py-2 flex items-center justify-between text-xs text-rose-300 shrink-0">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{errorMessage}</span>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="p-1 hover:bg-rose-500/20 rounded transition"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Main Workspace Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar: ONLY Conversation List + Top New Chat */}
        <Sidebar
          conversations={conversations}
          activeConversationId={activeConversationId}
          modelStatus={modelStatus}
          isOpen={sidebarOpen}
          onToggle={() => setSidebarOpen(false)}
          onNewChat={handleNewChat}
          onSelectConversation={handleSelectConversation}
          onDeleteConversation={handleDeleteConversation}
          onOpenStatusModal={() => setIsStatusModalOpen(true)}
        />

        {/* Central ChatGPT Chat Interface */}
        <ChatInterface
          messages={messages}
          uploadData={uploadData}
          activeFileName={activeFileName}
          inputPrompt={inputPrompt}
          isGenerating={isGenerating}
          isUploading={isUploading}
          onInputChange={setInputPrompt}
          onSendMessage={handleSendMessage}
          onFileUpload={handleFileUpload}
          onRemoveUpload={handleRemoveUpload}
          onOpenLightbox={(url, title) => setLightboxImage({ url, title })}
        />
      </div>

      {/* Architecture Status Modal */}
      <ModelStatusModal
        status={modelStatus}
        isOpen={isStatusModalOpen}
        onClose={() => setIsStatusModalOpen(false)}
      />

      {/* High-Resolution Evidence Lightbox */}
      <LightboxModal
        image={lightboxImage}
        onClose={() => setLightboxImage(null)}
      />
    </div>
  );
};

export default App;
