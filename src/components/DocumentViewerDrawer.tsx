import React, { useEffect, useState } from 'react';
import { Download, ExternalLink, FileText, Layers, X } from 'lucide-react';
import { lucentApi } from '../services/api';
import type { DocumentChunk, ResearchDocument } from '../types/lucent';

interface DocumentViewerDrawerProps {
  documentId: string | null;
  highlightChunkId?: string | null;
  onClose: () => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function DocumentViewerDrawer({
  documentId,
  highlightChunkId,
  onClose,
}: DocumentViewerDrawerProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [doc, setDoc] = useState<ResearchDocument | null>(null);
  const [chunks, setChunks] = useState<DocumentChunk[]>([]);
  const [tab, setTab] = useState<'document' | 'chunks'>('chunks');

  useEffect(() => {
    if (!documentId) {
      setDoc(null);
      setChunks([]);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    setTab(highlightChunkId ? 'chunks' : 'document');

    lucentApi
      .getDocument(documentId)
      .then((res) => {
        if (!cancelled) {
          setDoc(res.document);
          setChunks(res.chunks);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load document.');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [documentId, highlightChunkId]);

  if (!documentId) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div
        className="fixed inset-0 bg-black/70"
        onClick={onClose}
        aria-hidden="true"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Document inspector"
        className="relative z-10 w-full max-w-2xl bg-[#0A0F1C] border-l border-[#1E293B] h-full flex flex-col"
      >
        <div className="px-6 py-4 border-b border-[#1E293B] flex items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-xs text-[#94A3B8] font-mono truncate">{documentId}</p>
            <h2 className="mt-1 font-display text-lg font-semibold text-white leading-snug">
              {doc?.title || 'Loading document...'}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close document inspector"
            className="p-2 text-[#94A3B8] hover:text-white rounded-md shrink-0"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {loading && (
          <div className="p-6 space-y-3">
            <div className="h-4 w-2/3 bg-[#131B2E] rounded animate-pulse" />
            <div className="h-4 w-full bg-[#131B2E] rounded animate-pulse" />
            <div className="h-24 w-full bg-[#131B2E] rounded animate-pulse" />
          </div>
        )}

        {error && (
          <div className="m-6 p-4 rounded-md border border-red-500/30 bg-red-950/30 text-sm text-red-200">
            {error}
          </div>
        )}

        {!loading && doc && (
          <>
            <div className="px-6 py-4 border-b border-[#1E293B] bg-[#0D1321] space-y-3">
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[#94A3B8]">
                <span>{doc.isDemo ? 'Demonstration Document' : 'User Library Document'}</span>
                <span aria-hidden="true">·</span>
                <span>{doc.documentType}</span>
                <span aria-hidden="true">·</span>
                <span>{doc.source || 'Source not specified'}</span>
                <span aria-hidden="true">·</span>
                <span className="tabular-nums">{doc.publishedDate || 'Undated'}</span>
                {doc.pageCount !== null && (
                  <>
                    <span aria-hidden="true">·</span>
                    <span className="tabular-nums">{doc.pageCount} pages</span>
                  </>
                )}
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1 text-xs">
                <div>
                  <div className="text-[#64748B]">Original File</div>
                  <div className="text-[#E6EDF7] font-mono truncate mt-0.5">
                    {doc.originalFilename || 'Direct text entry'}
                  </div>
                </div>
                <div>
                  <div className="text-[#64748B]">Size / Words</div>
                  <div className="text-[#E6EDF7] font-mono tabular-nums mt-0.5">
                    {formatBytes(doc.fileSizeBytes)} · {doc.wordCount.toLocaleString()} w
                  </div>
                </div>
                <div>
                  <div className="text-[#64748B]">Chunks Embedded</div>
                  <div className="text-[#E6EDF7] font-mono tabular-nums mt-0.5">
                    {doc.embeddedChunkCount} / {doc.chunkCount}
                  </div>
                </div>
                <div>
                  <div className="text-[#64748B]">Uploaded</div>
                  <div className="text-[#E6EDF7] font-mono tabular-nums mt-0.5">
                    {new Date(doc.uploadedAt).toLocaleDateString()}
                  </div>
                </div>
              </div>

              {doc.embeddingError && (
                <div className="p-2.5 rounded border border-red-500/30 bg-red-950/30 text-xs text-red-200">
                  Embedding error: {doc.embeddingError}
                </div>
              )}

              <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                <div className="inline-flex items-center p-1 rounded-md bg-[#070B14] border border-[#1E293B]">
                  <button
                    type="button"
                    onClick={() => setTab('document')}
                    className={`px-3 py-1.5 text-xs font-medium rounded transition-colors flex items-center gap-1.5 whitespace-nowrap ${
                      tab === 'document'
                        ? 'bg-[#131B2E] text-white'
                        : 'text-[#94A3B8] hover:text-white'
                    }`}
                  >
                    <FileText className="w-3.5 h-3.5" />
                    <span>Full Text</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setTab('chunks')}
                    className={`px-3 py-1.5 text-xs font-medium rounded transition-colors flex items-center gap-1.5 whitespace-nowrap ${
                      tab === 'chunks'
                        ? 'bg-[#131B2E] text-white'
                        : 'text-[#94A3B8] hover:text-white'
                    }`}
                  >
                    <Layers className="w-3.5 h-3.5" />
                    <span>Indexed Chunks ({chunks.length})</span>
                  </button>
                </div>

                <div className="flex items-center gap-2">
                  <a
                    href={`/api/documents/${encodeURIComponent(doc.id)}/download`}
                    className="px-3 py-1.5 text-xs font-medium rounded-md border border-[#24324D] bg-[#131B2E] hover:bg-[#1B2640] text-[#E6EDF7] flex items-center gap-1.5 whitespace-nowrap"
                  >
                    <Download className="w-3.5 h-3.5 text-[#3B82F6]" />
                    <span>Download File</span>
                  </a>
                  {doc.sourceUrl && (
                    <a
                      href={doc.sourceUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-3 py-1.5 text-xs font-medium rounded-md border border-[#24324D] hover:bg-[#131B2E] text-[#94A3B8] hover:text-white flex items-center gap-1.5 whitespace-nowrap"
                    >
                      <ExternalLink className="w-3.5 h-3.5" />
                      <span>Source URL</span>
                    </a>
                  )}
                </div>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-6">
              {tab === 'document' ? (
                <div className="text-sm text-[#CBD5E1] leading-relaxed whitespace-pre-wrap max-w-[68ch]">
                  {doc.content}
                </div>
              ) : (
                <div className="space-y-4">
                  {chunks.map((chunk) => {
                    const isTarget = highlightChunkId === chunk.id;
                    return (
                      <div
                        key={chunk.id}
                        className={`p-4 rounded-md border ${
                          isTarget
                            ? 'border-[#3B82F6] bg-[#101A33]'
                            : 'border-[#1E293B] bg-[#0D1321]'
                        }`}
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-[#94A3B8] pb-2 border-b border-[#1E293B]">
                          <span className="font-mono text-[#E6EDF7]">{chunk.id}</span>
                          <div className="flex items-center gap-2 font-mono tabular-nums">
                            {chunk.pageNumber !== null && <span>Page {chunk.pageNumber}</span>}
                            {chunk.pageNumber !== null && <span aria-hidden="true">·</span>}
                            <span>{chunk.wordCount} words</span>
                            <span aria-hidden="true">·</span>
                            <span className={chunk.hasEmbedding ? 'text-emerald-400' : 'text-amber-400'}>
                              {chunk.hasEmbedding ? `Vectorized (${chunk.embeddingModel})` : 'Not embedded'}
                            </span>
                          </div>
                        </div>
                        <p className="mt-3 text-sm text-[#CBD5E1] leading-relaxed">{chunk.text}</p>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
