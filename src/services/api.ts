import type {
  AnalyticsSnapshot,
  CrossEncoderServiceStatus,
  DocumentChunk,
  EngineSettings,
  EvaluationMethodId,
  EvaluationQuery,
  EvaluationRunRecord,
  PublicRuntimeConfig,
  ResearchDocument,
  SearchComparisonResponse,
  SearchPipelineTrace,
  SearchRequestPayload,
  SearchResponsePayload,
  VectorIndexStats,
} from '../types/lucent';

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...(init?.headers || {}),
    },
  });

  const text = await response.text();
  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = { error: text };
    }
  }

  if (!response.ok) {
    const errMsg =
      typeof parsed === 'object' && parsed !== null && 'error' in parsed && typeof (parsed as { error: unknown }).error === 'string'
        ? (parsed as { error: string }).error
        : `Request failed with HTTP ${response.status}`;
    throw new Error(errMsg);
  }

  return parsed as T;
}

export const lucentApi = {
  getConfig(): Promise<{ runtime: PublicRuntimeConfig; settings: EngineSettings }> {
    return requestJson('/api/config');
  },

  listDocuments(): Promise<{
    documents: Array<Omit<ResearchDocument, 'content'>>;
    vectorIndex: VectorIndexStats;
  }> {
    return requestJson('/api/documents');
  },

  getDocument(id: string): Promise<{ document: ResearchDocument; chunks: DocumentChunk[] }> {
    return requestJson(`/api/documents/${encodeURIComponent(id)}`);
  },

  createDocument(payload: {
    title: string;
    source?: string;
    documentType: string;
    publishedDate?: string;
    authors?: string;
    sourceUrl?: string;
    content: string;
  }): Promise<{
    document: ResearchDocument;
    embeddingWarning: string | null;
    vectorIndex: VectorIndexStats;
  }> {
    return requestJson('/api/documents', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  uploadDocumentWithProgress(
    formData: FormData,
    onProgress: (percent: number, phase: 'uploading' | 'processing') => void
  ): Promise<{
    document: ResearchDocument;
    embeddingWarning: string | null;
    vectorIndex: VectorIndexStats;
  }> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/documents/upload');

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && event.total > 0) {
          const pct = Math.min(95, Math.round((event.loaded / event.total) * 95));
          onProgress(pct, pct >= 95 ? 'processing' : 'uploading');
        }
      };

      xhr.upload.onload = () => {
        onProgress(95, 'processing');
      };

      xhr.onerror = () => {
        reject(new Error('Network error while uploading document.'));
      };

      xhr.onload = () => {
        let parsed: unknown = null;
        try {
          parsed = xhr.responseText ? JSON.parse(xhr.responseText) : null;
        } catch {
          parsed = { error: xhr.responseText || 'Invalid server response' };
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          onProgress(100, 'processing');
          resolve(
            parsed as {
              document: ResearchDocument;
              embeddingWarning: string | null;
              vectorIndex: VectorIndexStats;
            }
          );
        } else {
          const msg =
            typeof parsed === 'object' && parsed !== null && 'error' in parsed && typeof (parsed as { error: unknown }).error === 'string'
              ? (parsed as { error: string }).error
              : `Upload failed with HTTP ${xhr.status}`;
          reject(new Error(msg));
        }
      };

      xhr.send(formData);
    });
  },

  reprocessDocument(
    id: string,
    forceReembed = false
  ): Promise<{
    document: ResearchDocument;
    embeddedNow: number;
    skippedUnchanged: number;
    embeddingWarning: string | null;
    vectorIndex: VectorIndexStats;
  }> {
    return requestJson(`/api/documents/${encodeURIComponent(id)}/reprocess`, {
      method: 'POST',
      body: JSON.stringify({ forceReembed }),
    });
  },

  embedDocument(
    id: string,
    force = false
  ): Promise<{
    document: ResearchDocument;
    embeddedNow: number;
    skippedUnchanged: number;
    model: string;
    vectorIndex: VectorIndexStats;
  }> {
    return requestJson(`/api/documents/${encodeURIComponent(id)}/embed`, {
      method: 'POST',
      body: JSON.stringify({ force }),
    });
  },

  deleteDocument(id: string): Promise<{ deleted: boolean; id: string; vectorIndex: VectorIndexStats }> {
    return requestJson(`/api/documents/${encodeURIComponent(id)}`, { method: 'DELETE' });
  },

  restoreDemoCorpus(): Promise<{ restoredCount: number }> {
    return requestJson('/api/documents/demo/restore', { method: 'POST' });
  },

  deleteDemoDocuments(): Promise<{
    deletedDocuments: number;
    deletedEvaluationQueries: number;
    vectorIndex: VectorIndexStats;
  }> {
    return requestJson('/api/documents/demo', { method: 'DELETE' });
  },

  reindexEmbeddings(force = false): Promise<{
    ok: boolean;
    embeddedNow: number;
    skippedUnchanged: number;
    totalTarget: number;
    model: string;
    vectorIndex: VectorIndexStats;
  }> {
    return requestJson('/api/documents/reindex-embeddings', {
      method: 'POST',
      body: JSON.stringify({ force }),
    });
  },

  search(payload: SearchRequestPayload): Promise<SearchResponsePayload> {
    return requestJson('/api/search', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  compareSearchModes(payload: {
    query: string;
    filters?: SearchRequestPayload['filters'];
    topK?: number;
    includeCrossEncoder?: boolean;
    includeLlmReranker?: boolean;
  }): Promise<SearchComparisonResponse> {
    return requestJson('/api/search/compare', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  getPipelineStatus(): Promise<{
    runtime: PublicRuntimeConfig;
    settings: EngineSettings;
    crossEncoder: CrossEncoderServiceStatus;
    vectorIndex: VectorIndexStats;
    indexSummary: { totalDocuments: number; totalChunks: number; embeddedChunks: number };
    lastTrace: SearchPipelineTrace | null;
  }> {
    return requestJson('/api/pipeline/status');
  },

  getRerankerStatus(): Promise<{
    crossEncoder: CrossEncoderServiceStatus;
    llmReranker: { configured: boolean; modelName: string; label: string };
  }> {
    return requestJson('/api/reranker/status');
  },

  getEvaluation(): Promise<{ queries: EvaluationQuery[]; runs: EvaluationRunRecord[] }> {
    return requestJson('/api/evaluation');
  },

  createEvaluationQuery(payload: {
    query: string;
    relevanceGrades: Record<string, number>;
    category: string;
    notes: string;
  }): Promise<{ query: EvaluationQuery }> {
    return requestJson('/api/evaluation/queries', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  updateEvaluationQuery(
    id: string,
    payload: {
      query?: string;
      relevanceGrades?: Record<string, number>;
      category?: string;
      notes?: string;
    }
  ): Promise<{ query: EvaluationQuery }> {
    return requestJson(`/api/evaluation/queries/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    });
  },

  deleteEvaluationQuery(id: string): Promise<{ deleted: boolean }> {
    return requestJson(`/api/evaluation/queries/${encodeURIComponent(id)}`, { method: 'DELETE' });
  },

  importEvaluationDataset(format: 'json' | 'csv', content: string): Promise<{
    importedCount: number;
    queries: EvaluationQuery[];
  }> {
    return requestJson('/api/evaluation/import', {
      method: 'POST',
      body: JSON.stringify({ format, content }),
    });
  },

  runEvaluation(payload: {
    methods: EvaluationMethodId[];
    k: number;
    datasetScope: 'all' | 'human_only' | 'demo_only';
  }): Promise<{ run: EvaluationRunRecord }> {
    return requestJson('/api/evaluation/run', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  getAnalytics(): Promise<AnalyticsSnapshot> {
    return requestJson('/api/analytics');
  },

  clearAnalyticsLogs(): Promise<{ cleared: boolean }> {
    return requestJson('/api/analytics/queries', { method: 'DELETE' });
  },

  getSettings(): Promise<{
    settings: EngineSettings;
    runtime: PublicRuntimeConfig;
    crossEncoder: CrossEncoderServiceStatus;
  }> {
    return requestJson('/api/settings');
  },

  updateSettings(partial: Partial<EngineSettings>): Promise<{
    settings: EngineSettings;
    rechunked: { documentsRechunked: number; totalChunks: number } | null;
  }> {
    return requestJson('/api/settings', {
      method: 'PATCH',
      body: JSON.stringify(partial),
    });
  },
};
