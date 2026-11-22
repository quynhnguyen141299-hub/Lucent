import { GoogleGenAI } from '@google/genai';
import { hasConfiguredGeminiKey, serverConfig } from './config.ts';

let cachedClient: GoogleGenAI | null = null;

export class GeminiServiceError extends Error {
  statusCode: number;
  code: string;

  constructor(message: string, statusCode = 500, code = 'GEMINI_ERROR') {
    super(message);
    this.name = 'GeminiServiceError';
    this.statusCode = statusCode;
    this.code = code;
  }
}

export function getGeminiClient(): GoogleGenAI {
  if (!hasConfiguredGeminiKey()) {
    throw new GeminiServiceError(
      'GEMINI_API_KEY is not configured on the server. Add a valid key in Settings > Secrets to enable semantic embeddings and AI answer synthesis.',
      503,
      'API_KEY_MISSING'
    );
  }
  if (!cachedClient) {
    cachedClient = new GoogleGenAI({
      apiKey: serverConfig.geminiApiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return cachedClient;
}

export function normalizeGeminiError(error: unknown): GeminiServiceError {
  if (error instanceof GeminiServiceError) {
    return error;
  }
  const errObj = error as { status?: number; code?: string | number; message?: string };
  const message = errObj?.message || 'An unexpected AI service error occurred.';
  const status = typeof errObj?.status === 'number' ? errObj.status : 500;

  if (message.includes('API_KEY_INVALID') || message.includes('PERMISSION_DENIED') || status === 401 || status === 403) {
    return new GeminiServiceError(
      'The configured Gemini API key was rejected or lacks permission. Verify your key in Settings > Secrets.',
      403,
      'API_KEY_INVALID'
    );
  }
  if (message.includes('RESOURCE_EXHAUSTED') || status === 429) {
    return new GeminiServiceError(
      'AI model quota or rate limit was reached. Retry shortly or verify your API billing tier in Settings > Secrets.',
      429,
      'RESOURCE_EXHAUSTED'
    );
  }
  if (message.includes('NOT_FOUND') || status === 404) {
    return new GeminiServiceError(
      `The requested model could not be found (active defaults: ${serverConfig.generationModel} / ${serverConfig.embeddingModel}). Upstream detail: ${message}`,
      404,
      'MODEL_NOT_FOUND'
    );
  }
  return new GeminiServiceError(message, status >= 400 && status < 600 ? status : 502, 'UPSTREAM_ERROR');
}

export async function withRetry<T>(fn: () => Promise<T>, maxRetries = 2): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      const normalized = normalizeGeminiError(err);
      const isRetryable = normalized.statusCode === 429 || normalized.statusCode >= 500;
      if (!isRetryable || attempt === maxRetries) {
        throw normalized;
      }
      const delayMs = 350 * Math.pow(2, attempt);
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
  throw normalizeGeminiError(lastError);
}
