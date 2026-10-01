import type { TokenResponse, User } from "./types";

const TOKEN_STORAGE_KEY = "research_assistant_auth_token";

export function getAuthToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
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
    throw new Error(
      message || `The server returned ${response.status}. Please try again.`,
    );
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
