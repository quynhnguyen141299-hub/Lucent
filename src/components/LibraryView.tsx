import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  Cpu,
  Download,
  Eye,
  FileUp,
  Loader2,
  Plus,
  RefreshCw,
  RotateCcw,
  Trash2,
} from 'lucide-react';
import { lucentApi } from '../services/api';
import type { DocumentType, ResearchDocument, VectorIndexStats } from '../types/lucent';

const DOC_TYPES: DocumentType[] = [
  'Research Paper',
  'Technical Report',
  'Policy Brief',
  'Clinical Study',
  'Dataset Notes',
  'Uploaded Document',
];

const MAX_CLIENT_UPLOAD_BYTES = 10 * 1024 * 1024;
const ALLOWED_EXTS = ['.pdf', '.txt', '.docx', '.md'];

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

interface LibraryViewProps {
  onInspectDocument: (documentId: string) => void;
}

export function LibraryView({ onInspectDocument }: LibraryViewProps) {
  const [documents, setDocuments] = useState<Array<Omit<ResearchDocument, 'content'>>>([]);
  const [vectorStats, setVectorStats] = useState<VectorIndexStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);

  // Filter state
  const [filterText, setFilterText] = useState('');
  const [originFilter, setOriginFilter] = useState<'all' | 'user' | 'demo'>('all');

  // Upload & Paste modal state
  const [showAddPanel, setShowAddPanel] = useState(false);
  const [addMode, setAddMode] = useState<'file' | 'text'>('file');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [title, setTitle] = useState('');
  const [source, setSource] = useState('');
  const [documentType, setDocumentType] = useState<DocumentType>('Research Paper');
  const [publishedDate, setPublishedDate] = useState('');
  const [authors, setAuthors] = useState('');
  const [sourceUrl, setSourceUrl] = useState('');
  const [textContent, setTextContent] = useState('');

  // Upload progress & per-doc busy state
  const [submitting, setSubmitting] = useState(false);
  const [uploadPercent, setUploadPercent] = useState(0);
  const [uploadPhase, setUploadPhase] = useState<'uploading' | 'processing' | null>(null);
  const [busyDocId, setBusyDocId] = useState<string | null>(null);
  const [reindexingAll, setReindexingAll] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadLibrary = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await lucentApi.listDocuments();
      setDocuments(res.documents);
      setVectorStats(res.vectorIndex);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load Document Library.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadLibrary();
  }, []);

  const resetForm = () => {
    setSelectedFile(null);
    setTitle('');
    setSource('');
    setPublishedDate('');
    setAuthors('');
    setSourceUrl('');
    setTextContent('');
    setUploadPercent(0);
    setUploadPhase(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const validateAndSelectFile = (file: File | null) => {
    setError(null);
    if (!file) {
      setSelectedFile(null);
      return;
    }
    const lower = file.name.toLowerCase();
    const validExt = ALLOWED_EXTS.some((ext) => lower.endsWith(ext));
    if (!validExt) {
      setError(`Unsupported file type "${file.name}". Supported formats: PDF (.pdf), Word (.docx), TXT (.txt), Markdown (.md).`);
      return;
    }
    if (file.size === 0) {
      setError('Selected file is empty (0 bytes).');
      return;
    }
    if (file.size > MAX_CLIENT_UPLOAD_BYTES) {
      setError(`File exceeds 10 MB limit (${formatBytes(file.size)}).`);
      return;
    }
    setSelectedFile(file);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setNotice(null);
    setWarning(null);
    setSubmitting(true);

    try {
      if (addMode === 'file') {
        if (!selectedFile) {
          throw new Error('Select a PDF, DOCX, TXT, or Markdown file to upload.');
        }
        const formData = new FormData();
        formData.append('file', selectedFile);
        if (title.trim()) formData.append('title', title.trim());
        if (source.trim()) formData.append('source', source.trim());
        formData.append('documentType', documentType);
        if (publishedDate) formData.append('publishedDate', publishedDate);
        if (authors.trim()) formData.append('authors', authors.trim());
        if (sourceUrl.trim()) formData.append('sourceUrl', sourceUrl.trim());

        setUploadPercent(5);
        setUploadPhase('uploading');
        const res = await lucentApi.uploadDocumentWithProgress(formData, (pct, phase) => {
          setUploadPercent(pct);
          setUploadPhase(phase);
        });
        setNotice(
          `Ingested "${res.document.title}" (${res.document.chunkCount} chunks${
            res.document.pageCount ? `, ${res.document.pageCount} pages` : ''
          }).`
        );
        if (res.embeddingWarning) {
          setWarning(`Document text is indexed for BM25 search, but embedding failed: ${res.embeddingWarning}`);
        }
      } else {
        if (!title.trim()) throw new Error('Document title is required.');
        if (textContent.trim().length < 20) {
          throw new Error('Provide at least 20 characters of document text.');
        }
        const res = await lucentApi.createDocument({
          title: title.trim(),
          source: source.trim() || undefined,
          documentType,
          publishedDate: publishedDate || undefined,
          authors: authors.trim() || undefined,
          sourceUrl: sourceUrl.trim() || undefined,
          content: textContent,
        });
        setNotice(`Ingested "${res.document.title}" (${res.document.chunkCount} chunks).`);
        if (res.embeddingWarning) {
          setWarning(`Document text is indexed for BM25 search, but embedding failed: ${res.embeddingWarning}`);
        }
      }

      resetForm();
      setShowAddPanel(false);
      await loadLibrary();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Document ingestion failed.');
    } finally {
      setSubmitting(false);
      setUploadPhase(null);
    }
  };

  const handleReprocess = async (docId: string, forceReembed: boolean) => {
    setBusyDocId(docId);
    setError(null);
    setNotice(null);
    setWarning(null);
    try {
      const res = await lucentApi.reprocessDocument(docId, forceReembed);
      setNotice(
        `Reprocessed "${res.document.title}": ${res.document.chunkCount} chunks (${res.embeddedNow} newly embedded, ${res.skippedUnchanged} reused from cache).`
      );
      if (res.embeddingWarning) {
        setWarning(`Reprocessing completed, but embedding returned an error: ${res.embeddingWarning}`);
      }
      await loadLibrary();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Reprocessing failed.');
    } finally {
      setBusyDocId(null);
    }
  };

  const handleRetryEmbed = async (docId: string) => {
    setBusyDocId(docId);
    setError(null);
    setNotice(null);
    setWarning(null);
    try {
      const res = await lucentApi.embedDocument(docId, false);
      setNotice(
        `Embedded "${res.document.title}" with ${res.model}: ${res.embeddedNow} new vectors, ${res.skippedUnchanged} unchanged skipped.`
      );
      await loadLibrary();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Embedding generation failed.');
      await loadLibrary();
    } finally {
      setBusyDocId(null);
    }
  };

  const handleDelete = async (docId: string, docTitle: string) => {
    setBusyDocId(docId);
    setError(null);
    try {
      await lucentApi.deleteDocument(docId);
      setNotice(`Deleted "${docTitle}" and removed its chunks and vectors from the index.`);
      await loadLibrary();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to delete document.');
    } finally {
      setBusyDocId(null);
    }
  };

  const handleReindexAll = async () => {
    setReindexingAll(true);
    setError(null);
    setNotice(null);
    setWarning(null);
    try {
      const res = await lucentApi.reindexEmbeddings(false);
      setNotice(
        `Vector index synced (${res.model}): generated ${res.embeddedNow} new chunk embeddings, reused ${res.skippedUnchanged} unchanged vectors.`
      );
      await loadLibrary();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Batch embedding sync failed.');
      await loadLibrary();
    } finally {
      setReindexingAll(false);
    }
  };

  const handleRestoreDemo = async () => {
    setError(null);
    try {
      const res = await lucentApi.restoreDemoCorpus();
      setNotice(
        res.restoredCount > 0
          ? `Restored ${res.restoredCount} clearly labeled demonstration documents.`
          : 'All demonstration documents are already present in the library.'
      );
      await loadLibrary();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to restore demonstration corpus.');
    }
  };

  const filteredDocs = useMemo(() => {
    const q = filterText.trim().toLowerCase();
    return documents.filter((doc) => {
      if (originFilter === 'user' && doc.isDemo) return false;
      if (originFilter === 'demo' && !doc.isDemo) return false;
      if (!q) return true;
      return (
        doc.title.toLowerCase().includes(q) ||
        (doc.source || '').toLowerCase().includes(q) ||
        (doc.originalFilename || '').toLowerCase().includes(q) ||
        doc.documentType.toLowerCase().includes(q)
      );
    });
  }, [documents, filterText, originFilter]);

  const totalChunks = documents.reduce((s, d) => s + d.chunkCount, 0);
  const embeddedChunks = documents.reduce((s, d) => s + d.embeddedChunkCount, 0);
  const userDocsCount = documents.filter((d) => !d.isDemo).length;
  const demoDocsCount = documents.filter((d) => d.isDemo).length;

  return (
    <div className="space-y-6">
      {/* Header Row */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-semibold text-white">Document Library</h1>
          <p className="text-sm text-[#94A3B8] mt-1">
            Upload PDF, DOCX, TXT, or Markdown research documents for cleaning, page-aware chunking,
            and persistent vector indexing.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={handleReindexAll}
            disabled={reindexingAll || documents.length === 0}
            className="h-9 px-3.5 rounded-md border border-[#24324D] bg-[#0D1321] hover:bg-[#131B2E] disabled:opacity-40 text-xs font-medium text-[#E6EDF7] flex items-center gap-1.5 whitespace-nowrap"
          >
            {reindexingAll ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-[#8B5CF6]" />
            ) : (
              <Cpu className="w-3.5 h-3.5 text-[#8B5CF6]" />
            )}
            <span>Embed Pending Chunks</span>
          </button>

          <button
            type="button"
            onClick={() => setShowAddPanel((v) => !v)}
            className="h-9 px-4 rounded-md bg-[#3B82F6] hover:bg-[#2563EB] text-xs font-medium text-white flex items-center gap-1.5 whitespace-nowrap"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>{showAddPanel ? 'Close Uploader' : 'Upload Document'}</span>
          </button>
        </div>
      </div>

      {/* Corpus Summary Strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 border-y border-[#1E293B] py-4">
        <div>
          <div className="text-xs text-[#64748B]">Indexed Documents</div>
          <div className="mt-1 font-mono text-xl font-semibold text-white tabular-nums">
            {documents.length}
          </div>
          <div className="text-[11px] text-[#94A3B8] mt-0.5 tabular-nums">
            {userDocsCount} uploaded · {demoDocsCount} demo
          </div>
        </div>
        <div>
          <div className="text-xs text-[#64748B]">Segmented Chunks</div>
          <div className="mt-1 font-mono text-xl font-semibold text-white tabular-nums">
            {totalChunks}
          </div>
          <div className="text-[11px] text-[#94A3B8] mt-0.5">Sentence-aware overlapping windows</div>
        </div>
        <div>
          <div className="text-xs text-[#64748B]">Vector Index Coverage</div>
          <div className="mt-1 font-mono text-xl font-semibold text-white tabular-nums">
            {embeddedChunks} / {totalChunks}
          </div>
          <div className="text-[11px] text-[#94A3B8] mt-0.5 tabular-nums">
            {vectorStats?.dimensions ? `${vectorStats.dimensions}-dim Float32` : 'Awaiting first embed'}
          </div>
        </div>
        <div>
          <div className="text-xs text-[#64748B]">Vector Storage</div>
          <div className="mt-1 font-mono text-xl font-semibold text-white tabular-nums">
            {vectorStats ? formatBytes(vectorStats.storageBytes) : '0 B'}
          </div>
          <div className="text-[11px] text-[#94A3B8] mt-0.5 truncate">
            {vectorStats?.backend || 'Local FlatIP Float32 Index'}
          </div>
        </div>
      </div>

      {/* Status Messages */}
      {error && (
        <div
          role="alert"
          className="p-3.5 rounded-lg border border-red-500/40 bg-red-950/30 text-xs text-red-200 flex items-start gap-2.5"
        >
          <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}
      {warning && (
        <div className="p-3.5 rounded-lg border border-amber-500/30 bg-amber-950/20 text-xs text-amber-200">
          {warning}
        </div>
      )}
      {notice && (
        <div className="p-3.5 rounded-lg border border-emerald-500/30 bg-emerald-950/20 text-xs text-emerald-200">
          {notice}
        </div>
      )}

      {/* Upload / Text Ingestion Panel */}
      {showAddPanel && (
        <section className="rounded-lg border border-[#24324D] bg-[#0D1321] p-6 space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[#1E293B]">
            <div>
              <h2 className="font-display text-base font-semibold text-white">
                Ingest Research Document
              </h2>
              <p className="text-xs text-[#94A3B8] mt-0.5">
                Leave Source or Publication Date blank if unknown — Lucent never fabricates missing
                metadata.
              </p>
            </div>
            <div className="inline-flex p-1 rounded-md bg-[#070B14] border border-[#1E293B] text-xs">
              <button
                type="button"
                onClick={() => setAddMode('file')}
                className={`px-3 py-1 rounded transition-colors ${
                  addMode === 'file' ? 'bg-[#131B2E] text-white' : 'text-[#94A3B8] hover:text-white'
                }`}
              >
                File Upload (PDF / DOCX / TXT)
              </button>
              <button
                type="button"
                onClick={() => setAddMode('text')}
                className={`px-3 py-1 rounded transition-colors ${
                  addMode === 'text' ? 'bg-[#131B2E] text-white' : 'text-[#94A3B8] hover:text-white'
                }`}
              >
                Direct Text Entry
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {addMode === 'file' ? (
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  validateAndSelectFile(e.dataTransfer.files?.[0] || null);
                }}
                className="p-6 rounded-lg border border-dashed border-[#334155] bg-[#070B14] text-center space-y-2"
              >
                <FileUp className="w-6 h-6 text-[#3B82F6] mx-auto" />
                <div className="text-sm text-[#E6EDF7]">
                  {selectedFile ? (
                    <span className="font-mono">
                      {selectedFile.name} ({formatBytes(selectedFile.size)})
                    </span>
                  ) : (
                    'Drop a PDF, DOCX, TXT, or Markdown file here, or browse'
                  )}
                </div>
                <p className="text-xs text-[#64748B]">
                  Maximum size 10 MB · PDF page numbers are preserved automatically during chunking
                </p>
                <div className="pt-1">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.docx,.txt,.md"
                    onChange={(e) => validateAndSelectFile(e.target.files?.[0] || null)}
                    className="text-xs text-[#94A3B8] file:mr-3 file:py-1.5 file:px-3 file:rounded file:border file:border-[#24324D] file:bg-[#131B2E] file:text-[#E6EDF7] file:text-xs hover:file:bg-[#1B2640]"
                  />
                </div>
              </div>
            ) : (
              <label className="block space-y-1 text-xs">
                <span className="text-[#94A3B8]">Document Text Content *</span>
                <textarea
                  rows={6}
                  value={textContent}
                  onChange={(e) => setTextContent(e.target.value)}
                  placeholder="Paste research abstract, technical report, or article text..."
                  className="w-full p-3 rounded-md bg-[#070B14] border border-[#1E293B] text-sm text-[#E6EDF7] placeholder-[#64748B] focus:border-[#3B82F6] focus:outline-none"
                />
              </label>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              <label className="block space-y-1 text-xs">
                <span className="text-[#94A3B8]">
                  Document Title {addMode === 'text' ? '*' : '(optional — defaults to file/PDF title)'}
                </span>
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="e.g. Dense Passage Retrieval Evaluation"
                  className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
                />
              </label>

              <label className="block space-y-1 text-xs">
                <span className="text-[#94A3B8]">Source / Venue (optional)</span>
                <input
                  type="text"
                  value={source}
                  onChange={(e) => setSource(e.target.value)}
                  placeholder="e.g. arXiv, Nature, Internal Memo"
                  className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
                />
              </label>

              <label className="block space-y-1 text-xs">
                <span className="text-[#94A3B8]">Document Type</span>
                <select
                  value={documentType}
                  onChange={(e) => setDocumentType(e.target.value as DocumentType)}
                  className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
                >
                  {DOC_TYPES.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </label>

              <label className="block space-y-1 text-xs">
                <span className="text-[#94A3B8]">Publication Date (optional)</span>
                <input
                  type="date"
                  value={publishedDate}
                  onChange={(e) => setPublishedDate(e.target.value)}
                  className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7] font-mono"
                />
              </label>

              <label className="block space-y-1 text-xs">
                <span className="text-[#94A3B8]">Authors (optional, comma-separated)</span>
                <input
                  type="text"
                  value={authors}
                  onChange={(e) => setAuthors(e.target.value)}
                  placeholder="e.g. A. Chen, M. Sato"
                  className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
                />
              </label>

              <label className="block space-y-1 text-xs">
                <span className="text-[#94A3B8]">Original URL (optional, https://)</span>
                <input
                  type="url"
                  value={sourceUrl}
                  onChange={(e) => setSourceUrl(e.target.value)}
                  placeholder="https://..."
                  className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
                />
              </label>
            </div>

            {/* Real Upload & Processing Progress Indicator */}
            {submitting && (
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center justify-between text-xs text-[#94A3B8]">
                  <span>
                    {uploadPhase === 'uploading'
                      ? 'Uploading file to server...'
                      : 'Extracting text, segmenting chunks, and generating vector embeddings...'}
                  </span>
                  <span className="font-mono tabular-nums">{uploadPercent}%</span>
                </div>
                <div className="h-1.5 w-full rounded bg-[#131B2E] overflow-hidden">
                  <div
                    className="h-full bg-[#3B82F6] transition-all duration-150"
                    style={{ width: `${Math.max(8, uploadPercent)}%` }}
                  />
                </div>
              </div>
            )}

            <div className="flex items-center justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => {
                  resetForm();
                  setShowAddPanel(false);
                }}
                className="h-9 px-4 rounded-md border border-[#1E293B] text-xs text-[#94A3B8] hover:text-white"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={submitting}
                className="h-9 px-5 rounded-md bg-[#3B82F6] hover:bg-[#2563EB] disabled:opacity-50 text-xs font-medium text-white flex items-center gap-2"
              >
                {submitting && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                <span>{submitting ? 'Ingesting...' : 'Ingest & Index'}</span>
              </button>
            </div>
          </form>
        </section>
      )}

      {/* Filter Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="text"
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
            placeholder="Filter by title, filename, source, or type..."
            className="h-9 w-64 sm:w-80 px-3 rounded-md bg-[#0D1321] border border-[#1E293B] text-xs text-[#E6EDF7] placeholder-[#64748B]"
          />

          <div className="inline-flex p-1 rounded-md bg-[#0D1321] border border-[#1E293B] text-xs">
            {(
              [
                ['all', `All (${documents.length})`],
                ['user', `Uploaded (${userDocsCount})`],
                ['demo', `Demo (${demoDocsCount})`],
              ] as const
            ).map(([val, label]) => (
              <button
                key={val}
                type="button"
                onClick={() => setOriginFilter(val)}
                className={`px-2.5 py-1 rounded transition-colors whitespace-nowrap ${
                  originFilter === val
                    ? 'bg-[#131B2E] text-white'
                    : 'text-[#94A3B8] hover:text-white'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {demoDocsCount > 0 && (
            <button
              type="button"
              onClick={async () => {
                setError(null);
                try {
                  const res = await lucentApi.deleteDemoDocuments();
                  setNotice(`Deleted ${res.deletedDocuments} demonstration documents.`);
                  await loadLibrary();
                } catch (err: unknown) {
                  setError(err instanceof Error ? err.message : 'Failed to delete demonstration documents.');
                }
              }}
              className="h-9 px-3 rounded-md border border-red-500/30 bg-red-950/20 hover:bg-red-950/40 text-xs text-red-200 flex items-center gap-1.5 whitespace-nowrap"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Delete Demo Docs</span>
            </button>
          )}
          <button
            type="button"
            onClick={handleRestoreDemo}
            className="h-9 px-3 rounded-md border border-[#1E293B] bg-[#0D1321] hover:bg-[#131B2E] text-xs text-[#94A3B8] hover:text-white flex items-center gap-1.5 whitespace-nowrap"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Restore Demo Corpus</span>
          </button>
        </div>
      </div>

      {/* Document Data Table */}
      <div className="rounded-lg border border-[#1E293B] bg-[#0D1321] overflow-x-auto">
        {loading ? (
          <div className="p-6 space-y-3">
            {[1, 2, 3, 4].map((n) => (
              <div key={n} className="h-10 w-full bg-[#131B2E] rounded animate-pulse" />
            ))}
          </div>
        ) : filteredDocs.length === 0 ? (
          <div className="p-10 text-center space-y-2">
            <p className="text-sm font-medium text-white">No documents in library</p>
            <p className="text-xs text-[#94A3B8]">
              Click Upload Document to add PDF, DOCX, TXT, or Markdown files to your search index.
            </p>
          </div>
        ) : (
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#1E293B] text-[#64748B] font-medium">
                <th className="py-3 px-4">Document &amp; Metadata</th>
                <th className="py-3 px-3 whitespace-nowrap">Processing</th>
                <th className="py-3 px-3 text-right whitespace-nowrap">Chunks</th>
                <th className="py-3 px-3 whitespace-nowrap">Embeddings</th>
                <th className="py-3 px-3 whitespace-nowrap">Uploaded</th>
                <th className="py-3 px-4 text-right whitespace-nowrap">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1E293B]">
              {filteredDocs.map((doc) => {
                const isBusy = busyDocId === doc.id;
                return (
                  <tr key={doc.id} className="hover:bg-[#11192C] transition-colors align-top">
                    <td className="py-3.5 px-4 max-w-md">
                      <button
                        type="button"
                        onClick={() => onInspectDocument(doc.id)}
                        className="text-left font-medium text-sm text-[#E6EDF7] hover:text-[#60A5FA] transition-colors leading-snug"
                      >
                        {doc.title}
                      </button>
                      <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-[#94A3B8]">
                        <span>{doc.documentType}</span>
                        <span aria-hidden="true">·</span>
                        <span>{doc.source || 'Source not specified'}</span>
                        <span aria-hidden="true">·</span>
                        <span className="tabular-nums">{doc.publishedDate || 'Undated'}</span>
                        {doc.originalFilename && (
                          <>
                            <span aria-hidden="true">·</span>
                            <span className="font-mono text-[#CBD5E1]">{doc.originalFilename}</span>
                          </>
                        )}
                        {doc.isDemo && (
                          <>
                            <span aria-hidden="true">·</span>
                            <span className="text-amber-300/90">Demonstration Data</span>
                          </>
                        )}
                      </div>
                      {doc.embeddingError && (
                        <div className="mt-1.5 text-[11px] text-red-300">
                          Embedding error: {doc.embeddingError}
                        </div>
                      )}
                    </td>

                    <td className="py-3.5 px-3 whitespace-nowrap">
                      <span
                        className={
                          doc.processingStatus === 'indexed'
                            ? 'text-emerald-400'
                            : doc.processingStatus === 'error'
                              ? 'text-red-400'
                              : 'text-amber-400'
                        }
                      >
                        {doc.processingStatus === 'indexed'
                          ? 'Indexed'
                          : doc.processingStatus === 'processing'
                            ? 'Processing'
                            : doc.processingStatus === 'error'
                              ? 'Error'
                              : 'Pending'}
                      </span>
                    </td>

                    <td className="py-3.5 px-3 text-right font-mono tabular-nums text-[#E6EDF7] whitespace-nowrap">
                      <div>{doc.chunkCount}</div>
                      {doc.pageCount !== null && (
                        <div className="text-[11px] text-[#64748B]">{doc.pageCount}p</div>
                      )}
                    </td>

                    <td className="py-3.5 px-3 whitespace-nowrap font-mono tabular-nums">
                      <div
                        className={
                          doc.embeddingStatus === 'completed'
                            ? 'text-emerald-400'
                            : doc.embeddingStatus === 'failed'
                              ? 'text-red-400'
                              : 'text-amber-400'
                        }
                      >
                        {doc.embeddingStatus === 'completed'
                          ? `Ready (${doc.embeddedChunkCount}/${doc.chunkCount})`
                          : doc.embeddingStatus === 'failed'
                            ? `Failed (${doc.embeddedChunkCount}/${doc.chunkCount})`
                            : doc.embeddingStatus === 'partial'
                              ? `Partial (${doc.embeddedChunkCount}/${doc.chunkCount})`
                              : `Pending (0/${doc.chunkCount})`}
                      </div>
                      {doc.embeddingModel && (
                        <div className="text-[11px] text-[#64748B]">{doc.embeddingModel}</div>
                      )}
                    </td>

                    <td className="py-3.5 px-3 font-mono tabular-nums text-[#94A3B8] whitespace-nowrap">
                      {new Date(doc.uploadedAt).toLocaleDateString()}
                    </td>

                    <td className="py-3.5 px-4 text-right whitespace-nowrap">
                      <div className="inline-flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() => onInspectDocument(doc.id)}
                          title="Inspect document and chunks"
                          className="p-1.5 rounded hover:bg-[#1B2640] text-[#94A3B8] hover:text-white"
                        >
                          <Eye className="w-3.5 h-3.5" />
                        </button>
                        {doc.embeddingStatus !== 'completed' && (
                          <button
                            type="button"
                            disabled={isBusy}
                            onClick={() => handleRetryEmbed(doc.id)}
                            title="Generate or retry chunk embeddings"
                            className="p-1.5 rounded hover:bg-[#1B2640] text-[#8B5CF6] hover:text-white disabled:opacity-40"
                          >
                            {isBusy ? (
                              <Loader2 className="w-3.5 h-3.5 animate-spin" />
                            ) : (
                              <Cpu className="w-3.5 h-3.5" />
                            )}
                          </button>
                        )}
                        <button
                          type="button"
                          disabled={isBusy}
                          onClick={() => handleReprocess(doc.id, false)}
                          title="Reprocess chunking and sync embeddings"
                          className="p-1.5 rounded hover:bg-[#1B2640] text-[#94A3B8] hover:text-white disabled:opacity-40"
                        >
                          <RefreshCw className={`w-3.5 h-3.5 ${isBusy ? 'animate-spin' : ''}`} />
                        </button>
                        <a
                          href={`/api/documents/${encodeURIComponent(doc.id)}/download`}
                          title="Download original file or text"
                          className="p-1.5 rounded hover:bg-[#1B2640] text-[#94A3B8] hover:text-white"
                        >
                          <Download className="w-3.5 h-3.5" />
                        </a>
                        <button
                          type="button"
                          disabled={isBusy}
                          onClick={() => handleDelete(doc.id, doc.title)}
                          title="Delete document and remove vectors"
                          className="p-1.5 rounded hover:bg-red-950/60 text-[#94A3B8] hover:text-red-400 disabled:opacity-40"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
