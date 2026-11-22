import type { NextFunction, Request, Response } from 'express';
import { serverConfig } from '../config.ts';

interface RateBucket {
  count: number;
  resetAt: number;
}

const buckets = new Map<string, RateBucket>();

/**
 * Sliding-window per-client rate limiter for API routes.
 */
export function apiRateLimiter(req: Request, res: Response, next: NextFunction): void {
  const now = Date.now();
  const windowMs = 60_000;
  const maxRequests = serverConfig.rateLimitPerMinute;
  const clientKey = req.ip || req.socket.remoteAddress || 'local';

  let bucket = buckets.get(clientKey);
  if (!bucket || now >= bucket.resetAt) {
    bucket = { count: 0, resetAt: now + windowMs };
    buckets.set(clientKey, bucket);
  }

  bucket.count++;
  const remaining = Math.max(0, maxRequests - bucket.count);
  res.setHeader('X-RateLimit-Limit', String(maxRequests));
  res.setHeader('X-RateLimit-Remaining', String(remaining));
  res.setHeader('X-RateLimit-Reset', String(Math.ceil(bucket.resetAt / 1000)));

  if (bucket.count > maxRequests) {
    const retryAfterSec = Math.max(1, Math.ceil((bucket.resetAt - now) / 1000));
    res.setHeader('Retry-After', String(retryAfterSec));
    res.status(429).json({
      error: `Rate limit exceeded (${maxRequests} requests/minute). Retry in ${retryAfterSec}s.`,
      code: 'RATE_LIMIT_EXCEEDED',
    });
    return;
  }

  // Periodically prune expired buckets
  if (buckets.size > 500) {
    for (const [key, b] of buckets.entries()) {
      if (now >= b.resetAt) buckets.delete(key);
    }
  }

  next();
}

/**
 * Structured request logger for /api/* endpoints. Never logs API keys or credentials.
 */
export function apiRequestLogger(req: Request, res: Response, next: NextFunction): void {
  const start = performance.now();
  res.on('finish', () => {
    const durationMs = Math.round(performance.now() - start);
    if (req.path.startsWith('/api/')) {
      console.log(
        JSON.stringify({
          ts: new Date().toISOString(),
          method: req.method,
          path: req.path,
          status: res.statusCode,
          durationMs,
        })
      );
    }
  });
  next();
}

/**
 * Strips ASCII control characters from short metadata strings and clamps length.
 */
export function sanitizeShortString(value: unknown, maxLen: number): string {
  if (typeof value !== 'string') return '';
  return value.replace(/[\x00-\x1F\x7F]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, maxLen);
}
