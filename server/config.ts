import path from 'path';
import type { PublicRuntimeConfig } from '../src/types/lucent.ts';

export const DEFAULT_GENERATION_MODEL = 'gemini-4-hydrogen';
export const DEFAULT_EMBEDDING_MODEL = 'gemini-embedding-2';
export const SUPPORTED_GENERATION_MODELS = ['gemini-4-hydrogen', 'gemini-3.1-flash-lite'];
export const SUPPORTED_EMBEDDING_MODELS = ['gemini-embedding-2'];
export const SUPPORTED_UPLOAD_EXTS = ['.pdf', '.txt', '.docx', '.md'];

function resolveSupportedModel(raw: string | undefined, supported: string[], fallback: string): string {
  const candidate = (raw || '').trim();
  if (candidate && supported.includes(candidate)) {
    return candidate;
  }
  return fallback;
}

function parseCrossEncoderUrl(raw?: string): string | null {
  if (!raw || !raw.trim()) return null;
  try {
    const parsed = new URL(raw.trim());
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return null;
    }
    return parsed.toString().replace(/\/+$/, '');
  } catch {
    return null;
  }
}

export const serverConfig = {
  port: Number(process.env.PORT || 3000),
  nodeEnv: process.env.NODE_ENV || 'development',
  geminiApiKey: (process.env.GEMINI_API_KEY || '').trim(),
  generationModel: resolveSupportedModel(
    process.env.LUCENT_GENERATION_MODEL,
    SUPPORTED_GENERATION_MODELS,
    DEFAULT_GENERATION_MODEL
  ),
  embeddingModel: resolveSupportedModel(
    process.env.LUCENT_EMBEDDING_MODEL,
    SUPPORTED_EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL
  ),
  crossEncoderServiceUrl: parseCrossEncoderUrl(process.env.CROSS_ENCODER_SERVICE_URL),
  crossEncoderTimeoutMs: Math.max(1000, Math.min(60000, Number(process.env.CROSS_ENCODER_TIMEOUT_MS || 15000))),
  rateLimitPerMinute: Math.max(10, Math.min(1000, Number(process.env.LUCENT_RATE_LIMIT_PER_MIN || 120))),
  dataDir: path.resolve(process.cwd(), process.env.LUCENT_DATA_DIR || './data'),
  maxUploadBytes: 10 * 1024 * 1024, // 10 MB
};

export function hasConfiguredGeminiKey(): boolean {
  const key = serverConfig.geminiApiKey;
  return Boolean(key && key !== 'MY_GEMINI_API_KEY' && key.length > 10);
}

export function getMaskedCrossEncoderUrl(): string | null {
  if (!serverConfig.crossEncoderServiceUrl) return null;
  try {
    const parsed = new URL(serverConfig.crossEncoderServiceUrl);
    return `${parsed.protocol}//${parsed.host}`;
  } catch {
    return null;
  }
}

export function getPublicRuntimeConfig(): PublicRuntimeConfig {
  return {
    hasGeminiApiKey: hasConfiguredGeminiKey(),
    hasCrossEncoderService: Boolean(serverConfig.crossEncoderServiceUrl),
    generationModel: serverConfig.generationModel,
    embeddingModel: serverConfig.embeddingModel,
    supportedEmbeddingModels: SUPPORTED_EMBEDDING_MODELS,
    maxUploadBytes: serverConfig.maxUploadBytes,
    supportedFileExtensions: SUPPORTED_UPLOAD_EXTS,
    rateLimitPerMinute: serverConfig.rateLimitPerMinute,
    environment: serverConfig.nodeEnv,
  };
}
