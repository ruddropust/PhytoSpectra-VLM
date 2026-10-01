import { useState, useEffect, useCallback } from "react";
import { apiClient } from "../api/client";
import { ModelStatus, ChatMessage, ConversationSession } from "../types";

const STORAGE_KEY = "wheatspectra_conversations_v1";

const createInitialConversation = (): ConversationSession => ({
  id: `conv-${Date.now()}`,
  sessionId: null,
  title: "New Analysis",
  createdAt: Date.now(),
  updatedAt: Date.now(),
  uploadData: null,
  activeFileName: null,
  messages: [],
});

export const useWheatChat = () => {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [conversations, setConversations] = useState<ConversationSession[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed;
        }
      }
    } catch (e) {
      console.error("Failed to load conversations from storage", e);
    }
    return [createInitialConversation()];
  });

  const [activeConversationId, setActiveConversationId] = useState<string>(() => {
    return conversations[0]?.id || `conv-${Date.now()}`;
  });

  const [inputPrompt, setInputPrompt] = useState<string>("");
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [isStatusModalOpen, setIsStatusModalOpen] = useState<boolean>(false);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [lightboxImage, setLightboxImage] = useState<{ url: string; title: string } | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Sync conversations to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch (e) {
      console.error("Failed to save conversations to storage", e);
    }
  }, [conversations]);

  // Load model status on startup
  useEffect(() => {
    const init = async () => {
      try {
        const status = await apiClient.getModelStatus();
        setModelStatus(status);
      } catch (err: any) {
        console.error("Initialization error:", err);
        setErrorMessage("Failed to connect to WheatSpectraChat backend.");
      }
    };
    init();
  }, []);

  const activeConversation = conversations.find((c) => c.id === activeConversationId) || conversations[0];
  const messages = activeConversation?.messages || [];
  const uploadData = activeConversation?.uploadData || null;
  const activeFileName = activeConversation?.activeFileName || null;
  const sessionId = activeConversation?.sessionId || uploadData?.session_id || null;

  const handleNewChat = useCallback(() => {
    const newConv = createInitialConversation();
    setConversations((prev) => [newConv, ...prev]);
    setActiveConversationId(newConv.id);
    setInputPrompt("");
    setErrorMessage(null);
  }, []);

  const handleSelectConversation = useCallback((id: string) => {
    setActiveConversationId(id);
    setInputPrompt("");
    setErrorMessage(null);
  }, []);

  const handleDeleteConversation = useCallback(
    (id: string, e: React.MouseEvent) => {
      e.stopPropagation();
      setConversations((prev) => {
        const updated = prev.filter((c) => c.id !== id);
        if (updated.length === 0) {
          const fresh = createInitialConversation();
          setActiveConversationId(fresh.id);
          return [fresh];
        }
        if (activeConversationId === id) {
          setActiveConversationId(updated[0].id);
        }
        return updated;
      });
    },
    [activeConversationId]
  );

  const handleFileUpload = useCallback(
    async (file: File) => {
      setIsUploading(true);
      setErrorMessage(null);
      try {
        const res = await apiClient.uploadTiff(file, sessionId || undefined);
        setConversations((prev) =>
          prev.map((conv) => {
            if (conv.id === activeConversationId) {
              const updatedTitle =
                conv.messages.length === 0 && conv.title === "New Analysis"
                  ? file.name.replace(/\.[^/.]+$/, "")
                  : conv.title;
              return {
                ...conv,
                sessionId: res.session_id,
                uploadData: res,
                activeFileName: file.name,
                title: updatedTitle,
                updatedAt: Date.now(),
              };
            }
            return conv;
          })
        );
      } catch (err: any) {
        console.error("Upload error:", err);
        setErrorMessage(
          err.response?.data?.detail || "Failed to upload and preprocess TIFF file."
        );
      } finally {
        setIsUploading(false);
      }
    },
    [activeConversationId, sessionId]
  );

  const handleRemoveUpload = useCallback(() => {
    setConversations((prev) =>
      prev.map((conv) => {
        if (conv.id === activeConversationId) {
          return {
            ...conv,
            uploadData: null,
            activeFileName: null,
            updatedAt: Date.now(),
          };
        }
        return conv;
      })
    );
  }, [activeConversationId]);

  const handleSendMessage = useCallback(
    async (overrideText?: string) => {
      const text = (overrideText || inputPrompt).trim();
      if (!text || isGenerating) return;

      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        sender: "user",
        text,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };

      // Update messages immediately with user query
      setConversations((prev) =>
        prev.map((conv) => {
          if (conv.id === activeConversationId) {
            const isFirst = conv.messages.length === 0;
            const updatedTitle = isFirst
              ? text.length > 28
                ? `${text.slice(0, 28)}...`
                : text
              : conv.title;
            return {
              ...conv,
              title: updatedTitle,
              updatedAt: Date.now(),
              messages: [...conv.messages, userMsg],
            };
          }
          return conv;
        })
      );

      setInputPrompt("");
      setIsGenerating(true);
      setErrorMessage(null);

      try {
        let chatSessionId = sessionId;
        if (!chatSessionId) {
          chatSessionId = await apiClient.createSession();
          setConversations((prev) =>
            prev.map((conv) =>
              conv.id === activeConversationId
                ? { ...conv, sessionId: chatSessionId, updatedAt: Date.now() }
                : conv
            )
          );
        }

        const res = await apiClient.sendChat(
          chatSessionId,
          uploadData?.image_id,
          text
        );
        const assistantMsg: ChatMessage = {
          id: `asst-${Date.now()}`,
          sender: "assistant",
          response: res,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        };

        setConversations((prev) =>
          prev.map((conv) => {
            if (conv.id === activeConversationId) {
              return {
                ...conv,
                updatedAt: Date.now(),
                messages: [...conv.messages, assistantMsg],
              };
            }
            return conv;
          })
        );
      } catch (err: any) {
        console.error("Chat error:", err);
        setErrorMessage(
          err.response?.data?.detail || "Inference failed during chat reasoning pipeline."
        );
      } finally {
        setIsGenerating(false);
      }
    },
    [inputPrompt, activeConversationId, uploadData, sessionId, isGenerating]
  );

  return {
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
  };
};
