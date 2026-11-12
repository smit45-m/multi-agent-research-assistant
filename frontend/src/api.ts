import type {
  ChatThreadItem,
  ChatTurnItem,
  HistoryThreadsPayload,
  LongTermMemoryItem,
  SessionItem,
  TokenResponse,
  User,
  UserFilesHistoryPayload,
  UserMemoryPayload,
} from "./types";

const TOKEN_STORAGE_KEY = "research_assistant_auth_token";

export function getAuthToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function getAuthHeaders(): Record<string, string> {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export function setAuthToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_STORAGE_KEY, token);
  } catch {
    /* Ignore localStorage restrictions */
  }
}

export function clearAuthToken(): void {
  try {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch {
    /* Ignore localStorage restrictions */
  }
}

export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const token = getAuthToken();
  const headers = new Headers(options.headers || {});
  
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  let response: Response;
  try {
    response = await fetch(path, {
      ...options,
      headers,
      signal: options.signal ?? AbortSignal.timeout(120_000),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw error;
    throw new Error(
      error instanceof DOMException && error.name === "TimeoutError"
        ? "The server took too long to respond. Try again with a more focused question."
        : "Cannot reach the research server. Check that the backend is running, then try again.",
    );
  }
  const json = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = json?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail
              .map((item: { msg?: string }) => item.msg)
              .filter(Boolean)
              .join(". ")
          : "";
    const err = new Error(
      message || `The server returned ${response.status}. Please try again.`,
    ) as Error & { status?: number };
    err.status = response.status;
    throw err;
  }
  if (!json)
    throw new Error(
      "The server returned an unexpected response. Check the API connection.",
    );
  return json as T;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Something went wrong. Please try again.";
}

export function safeUrl(value: string): string | undefined {
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : undefined;
  } catch {
    return undefined;
  }
}

export interface HealthInfo {
  status: string;
  version: string;
  uptime_seconds: number;
  vector_store_documents: number;
  active_concurrent_capacity: number;
  latency_sla_seconds: number;
  accuracy_benchmark_target: number;
}

export async function loginApi(email: string, password: string): Promise<TokenResponse> {
  const data = await api<TokenResponse>("/api/v1/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  setAuthToken(data.access_token);
  return data;
}

export async function signupApi(
  email: string,
  password: string,
  full_name: string,
  role: string = "user"
): Promise<TokenResponse> {
  const data = await api<TokenResponse>("/api/v1/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, full_name, role }),
  });
  setAuthToken(data.access_token);
  return data;
}

export async function fetchMeApi(): Promise<User> {
  return api<User>("/api/v1/auth/me");
}

export async function logoutApi(): Promise<void> {
  try {
    await api<{ message: string }>("/api/v1/auth/logout", {
      method: "POST",
    });
  } finally {
    clearAuthToken();
  }
}

export async function logoutAllApi(): Promise<{ message: string; revoked_count: number }> {
  try {
    return await api<{ message: string; revoked_count: number }>("/api/v1/auth/logout-all", {
      method: "POST",
    });
  } finally {
    clearAuthToken();
  }
}

export async function fetchSessionsApi(): Promise<SessionItem[]> {
  return api<SessionItem[]>("/api/v1/auth/sessions");
}

export async function fetchThreadsApi(
  search?: string,
  limit: number = 40,
): Promise<HistoryThreadsPayload> {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  if (search && search.trim()) params.set("search", search.trim());
  return api<HistoryThreadsPayload>(`/api/v1/history/threads?${params.toString()}`);
}

export async function fetchThreadApi(threadId: string): Promise<ChatThreadItem> {
  return api<ChatThreadItem>(`/api/v1/history/threads/${encodeURIComponent(threadId)}`);
}

export async function fetchChatApi(chatId: string): Promise<ChatTurnItem> {
  return api<ChatTurnItem>(`/api/v1/history/chats/${encodeURIComponent(chatId)}`);
}

export async function deleteThreadApi(
  threadId: string,
): Promise<{ status: string; thread_id: string; deleted_turns: number }> {
  return api<{ status: string; thread_id: string; deleted_turns: number }>(
    `/api/v1/history/threads/${encodeURIComponent(threadId)}`,
    { method: "DELETE" },
  );
}

export async function deleteChatApi(
  chatId: string,
): Promise<{ status: string; chat_id: string }> {
  return api<{ status: string; chat_id: string }>(
    `/api/v1/history/chats/${encodeURIComponent(chatId)}`,
    { method: "DELETE" },
  );
}

export async function fetchUserFilesWithQaApi(): Promise<UserFilesHistoryPayload> {
  return api<UserFilesHistoryPayload>("/api/v1/history/files");
}

export async function fetchMemoryApi(): Promise<UserMemoryPayload> {
  return api<UserMemoryPayload>("/api/v1/memory/");
}

export async function pinMemoryApi(
  title: string,
  summary: string,
  kind: "pinned_note" | "user_preference" = "pinned_note",
): Promise<LongTermMemoryItem> {
  return api<LongTermMemoryItem>("/api/v1/memory/long-term", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, summary, kind }),
  });
}

export async function deleteLongTermMemoryApi(
  memoryId: string,
): Promise<{ status: string; memory_id: string }> {
  return api<{ status: string; memory_id: string }>(
    `/api/v1/memory/long-term/${encodeURIComponent(memoryId)}`,
    { method: "DELETE" },
  );
}

export async function clearShortTermMemoryApi(): Promise<{
  status: string;
  cleared_count: number;
}> {
  return api<{ status: string; cleared_count: number }>("/api/v1/memory/short-term", {
    method: "DELETE",
  });
}
