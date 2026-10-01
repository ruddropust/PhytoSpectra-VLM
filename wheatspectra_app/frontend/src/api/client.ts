import axios from "axios";
import { ModelStatus, UploadResult, ChatResponse } from "../types";

const API_BASE = "";

export const apiClient = {
  async getHealth() {
    const res = await axios.get(`${API_BASE}/api/v1/health`);
    return res.data;
  },

  async getModelStatus(): Promise<ModelStatus> {
    const res = await axios.get(`${API_BASE}/api/v1/model/status`);
    return res.data;
  },

  async createSession(): Promise<string> {
    const res = await axios.post(`${API_BASE}/api/v1/session`);
    return res.data.session_id;
  },

  async uploadTiff(file: File, sessionId?: string): Promise<UploadResult> {
    const formData = new FormData();
    formData.append("file", file);
    if (sessionId) {
      formData.append("session_id", sessionId);
    }
    const res = await axios.post(`${API_BASE}/api/v1/upload`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return res.data;
  },

  async sendChat(
    sessionId: string,
    imageId: string | undefined,
    question: string
  ): Promise<ChatResponse> {
    const res = await axios.post(`${API_BASE}/api/v1/chat`, {
      session_id: sessionId,
      ...(imageId ? { image_id: imageId } : {}),
      question,
    });
    return res.data;
  },
};
