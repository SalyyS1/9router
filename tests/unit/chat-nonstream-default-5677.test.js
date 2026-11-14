import { beforeEach, describe, expect, it, vi } from "vitest";
import "../translator/registerAll.js";

const { executeMock, isNativePassthroughMock } = vi.hoisted(() => ({
  executeMock: vi.fn(),
  isNativePassthroughMock: vi.fn(() => false),
}));

vi.mock("../../open-sse/executors/index.js", () => ({
  getExecutor: vi.fn(() => ({
    noAuth: true,
    execute: executeMock,
    refreshCredentials: vi.fn().mockResolvedValue(null),
  })),
}));

vi.mock("../../open-sse/utils/requestLogger.js", () => ({
  createRequestLogger: vi.fn(async () => ({
    logClientRawRequest: vi.fn(),
    logRawRequest: vi.fn(),
    logTargetRequest: vi.fn(),
    logProviderResponse: vi.fn(),
    logConvertedResponse: vi.fn(),
    logError: vi.fn(),
  })),
}));

vi.mock("../../open-sse/utils/clientDetector.js", () => ({
  detectClientTool: vi.fn(() => "openai"),
  isNativePassthrough: isNativePassthroughMock,
}));

vi.mock("@/lib/usageDb.js", () => ({
  trackPendingRequest: vi.fn(),
  appendRequestLog: vi.fn(async () => {}),
  saveRequestDetail: vi.fn(async () => {}),
  saveRequestUsage: vi.fn(async () => {}),
}));

const { handleChatCore } = await import("../../open-sse/handlers/chatCore.js");

const SAMPLE_COMPLETION = {
  id: "chatcmpl-5677",
  object: "chat.completion",
  created: 1700000000,
  model: "deepseek/deepseek-chat",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "pong" },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
};

describe("#5677 non-streaming default and JSON passthrough without trailing [DONE]", { timeout: 20_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks();
    isNativePassthroughMock.mockReturnValue(false);
  });

  it.each([
    ["Accept: */*", { accept: "*/*" }],
    ["no Accept header", {}],
  ])("treats omitted stream as false (%s) and returns clean JSON without data: [DONE]", async (_label, headers) => {
    executeMock.mockResolvedValueOnce({
      response: new Response(JSON.stringify(SAMPLE_COMPLETION), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
      url: "https://openrouter.ai/api/v1/chat/completions",
      headers: {},
      transformedBody: null,
    });

    const body = {
      model: "openrouter/deepseek/deepseek-chat",
      messages: [{ role: "user", content: "ping" }],
    };

    const result = await handleChatCore({
      body,
      modelInfo: { provider: "openrouter", model: "deepseek/deepseek-chat" },
      credentials: { apiKey: "sk-or-test", providerSpecificData: {} },
      clientRawRequest: {
        endpoint: "/v1/chat/completions",
        body,
        headers,
      },
      connectionId: "conn-openrouter",
      log: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
    });

    expect(executeMock).toHaveBeenCalledTimes(1);
    expect(executeMock.mock.calls[0][0].stream).toBe(false);
    expect(result.success).toBe(true);
    expect(result.response.headers.get("content-type")).toContain("application/json");
    const rawText = await result.response.text();
    expect(rawText).not.toContain("[DONE]");
    expect(JSON.parse(rawText)).toMatchObject({
      id: "chatcmpl-5677",
      object: "chat.completion",
    });
  });

  it("keeps Gemini provider non-streaming when OpenAI client omits stream", async () => {
    executeMock.mockResolvedValueOnce({
      response: new Response(
        JSON.stringify({
          candidates: [{ content: { role: "model", parts: [{ text: "ola" }] }, finishReason: "STOP", index: 0 }],
          usageMetadata: { promptTokenCount: 4, candidatesTokenCount: 1, totalTokenCount: 5 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
      url: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
      headers: {},
      transformedBody: null,
    });

    const body = {
      model: "gemini/gemini-2.5-flash",
      messages: [{ role: "user", content: "oi" }],
    };

    const result = await handleChatCore({
      body,
      modelInfo: { provider: "gemini", model: "gemini-2.5-flash" },
      credentials: { apiKey: "gm-test", providerSpecificData: {} },
      clientRawRequest: {
        endpoint: "/v1/chat/completions",
        body,
        headers: { accept: "*/*" },
      },
      connectionId: "conn-gemini",
      log: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
    });

    expect(executeMock.mock.calls[0][0].stream).toBe(false);
    expect(result.success).toBe(true);
    expect(result.response.headers.get("content-type")).toContain("application/json");
    const parsed = await result.response.json();
    expect(parsed.object).toBe("chat.completion");
    expect(parsed.choices[0].message.content).toBe("ola");
  });

  it("never wraps JSON from a forceStream provider (openai passthrough) in SSE with trailing [DONE] when stream is false", async () => {
    isNativePassthroughMock.mockReturnValue(true);
    executeMock.mockResolvedValueOnce({
      response: new Response(
        JSON.stringify({ ...SAMPLE_COMPLETION, model: "gpt-4.1" }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
      url: "https://api.openai.com/v1/chat/completions",
      headers: {},
      transformedBody: null,
    });

    const body = {
      model: "openai/gpt-4.1",
      stream: false,
      messages: [{ role: "user", content: "ping" }],
    };

    const result = await handleChatCore({
      body,
      modelInfo: { provider: "openai", model: "gpt-4.1" },
      credentials: { apiKey: "sk-openai-test", providerSpecificData: {} },
      clientRawRequest: {
        endpoint: "/v1/chat/completions",
        body,
        headers: { accept: "*/*" },
      },
      connectionId: "conn-openai",
      log: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
    });

    expect(result.success).toBe(true);
    expect(result.response.headers.get("content-type")).toContain("application/json");
    const rawText = await result.response.text();
    expect(rawText).not.toContain("[DONE]");
    expect(JSON.parse(rawText).model).toBe("gpt-4.1");
  });

  it("still streams SSE when body.stream is explicitly true", async () => {
    const encoder = new TextEncoder();
    const ssePayload = 'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';
    executeMock.mockResolvedValueOnce({
      response: new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode(ssePayload));
            controller.close();
          },
        }),
        { status: 200, headers: { "content-type": "text/event-stream" } },
      ),
      url: "https://openrouter.ai/api/v1/chat/completions",
      headers: {},
      transformedBody: null,
    });

    const body = {
      model: "openrouter/deepseek/deepseek-chat",
      stream: true,
      messages: [{ role: "user", content: "ping" }],
    };

    const result = await handleChatCore({
      body,
      modelInfo: { provider: "openrouter", model: "deepseek/deepseek-chat" },
      credentials: { apiKey: "sk-or-test", providerSpecificData: {} },
      clientRawRequest: {
        endpoint: "/v1/chat/completions",
        body,
        headers: { accept: "*/*" },
      },
      connectionId: "conn-openrouter",
      log: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
    });

    expect(executeMock.mock.calls[0][0].stream).toBe(true);
    expect(result.success).toBe(true);
    expect(result.response.headers.get("content-type")).toContain("text/event-stream");
  });
});
