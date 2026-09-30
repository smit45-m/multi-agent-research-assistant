export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...options,
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
