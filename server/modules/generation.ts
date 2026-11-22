import { ThinkingLevel, Type } from '@google/genai';
import { serverConfig } from '../config.ts';
import { getGeminiClient, normalizeGeminiError, withRetry } from '../geminiClient.ts';
import type {
  AnswerStyle,
  QueryCategory,
  QueryClassificationResult,
  RagAnswer,
  RagFinding,
  RagSourceReference,
  SearchResultItem,
} from '../../src/types/lucent.ts';

const VALID_QUERY_CATEGORIES: Set<QueryCategory> = new Set([
  'factual',
  'comparative',
  'historical',
  'analytical',
  'explanatory',
]);

const VALID_ANSWER_STYLES: Set<AnswerStyle> = new Set([
  'concise',
  'detailed',
  'bullets',
  'comparative',
]);

export function normalizeAnswerStyle(style?: string): AnswerStyle {
  if (style && VALID_ANSWER_STYLES.has(style as AnswerStyle)) {
    return style as AnswerStyle;
  }
  return 'detailed';
}

/**
 * Classifies a natural-language query into one of 5 research intent categories
 * using structured JSON output with strict enum validation.
 */
export async function classifyQuery(query: string): Promise<QueryClassificationResult> {
  const start = performance.now();

  try {
    const ai = getGeminiClient();
    const response = await withRetry(() =>
      ai.models.generateContent({
        model: serverConfig.generationModel,
        contents: `Classify the following research query:\n"${query}"`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          systemInstruction: [
            'Classify the user search query into exactly one category:',
            '- factual: asks for a specific fact, definition, metric, or entity attribute.',
            '- comparative: compares two or more methods, systems, policies, or outcomes.',
            '- historical: asks about evolution over time, chronology, milestones, or past events.',
            '- analytical: asks for root causes, trade-offs, evaluation, risk analysis, or synthesis.',
            '- explanatory: asks how or why a mechanism, process, or algorithm works.',
            'Return strict JSON with category, confidence (0..1), and a 1-sentence reasoning.',
          ].join('\n'),
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              category: {
                type: Type.STRING,
                enum: ['factual', 'comparative', 'historical', 'analytical', 'explanatory'],
              },
              confidence: { type: Type.NUMBER },
              reasoning: { type: Type.STRING },
            },
            required: ['category', 'confidence', 'reasoning'],
          },
        },
      })
    );

    const raw = (response.text || '').trim();
    const parsed = JSON.parse(raw) as {
      category?: string;
      confidence?: number;
      reasoning?: string;
    };

    if (!parsed.category || !VALID_QUERY_CATEGORIES.has(parsed.category as QueryCategory)) {
      throw new Error(`Invalid classification category returned: "${String(parsed.category)}"`);
    }

    const confidence =
      typeof parsed.confidence === 'number' && Number.isFinite(parsed.confidence)
        ? Math.max(0, Math.min(1, Math.round(parsed.confidence * 100) / 100))
        : 0.75;

    return {
      category: parsed.category as QueryCategory,
      confidence,
      reasoning: (parsed.reasoning || '').trim().slice(0, 240),
      model: serverConfig.generationModel,
      status: 'classified',
      latencyMs: Math.max(1, Math.round(performance.now() - start)),
    };
  } catch (err) {
    const normalized = normalizeGeminiError(err);
    // Deterministic lightweight classification label ONLY clearly flagged as fallback/error
    const lower = query.toLowerCase();
    let fallbackCategory: QueryCategory = 'explanatory';
    if (/\b(vs\.?|versus|compare|comparison|difference|better than)\b/.test(lower)) {
      fallbackCategory = 'comparative';
    } else if (/\b(history|timeline|evolution|when did|origin|since \d{4})\b/.test(lower)) {
      fallbackCategory = 'historical';
    } else if (/\b(why|cause|impact|trade-?off|evaluate|risk|challenge)\b/.test(lower)) {
      fallbackCategory = 'analytical';
    } else if (/\b(what is|who|which metric|define|how many|how much)\b/.test(lower)) {
      fallbackCategory = 'factual';
    }

    return {
      category: fallbackCategory,
      confidence: 0,
      reasoning: `Model classification unavailable (${normalized.message}). Applied rule-based fallback label.`,
      model: null,
      status: 'fallback',
      latencyMs: Math.max(1, Math.round(performance.now() - start)),
    };
  }
}

export interface RawGeneratedRagPayload {
  summary: string;
  keyFindings: Array<{
    claim: string;
    evidenceType?: string;
    citationIndices: number[];
  }>;
  conflictingEvidence?: string | null;
  evidenceGaps?: string | null;
  usedSourceIndices?: number[];
}

/**
 * Strictly validates every inline [n] marker and structured citation index against
 * the actual retrieved passages so the LLM can never manufacture references.
 */
export function validateAndFilterCitations(
  raw: RawGeneratedRagPayload,
  evidencePool: SearchResultItem[]
): {
  sanitizedSummary: string;
  keyFindings: RagFinding[];
  sourcesUsed: RagSourceReference[];
  rejectedCitationCount: number;
} {
  const maxIndex = evidencePool.length;
  let rejectedCitationCount = 0;
  const citedSet = new Set<number>();

  // Validate inline [n] citations inside summary; strip any out-of-range markers
  const sanitizedSummary = (raw.summary || '')
    .replace(/\[(\d+)\]/g, (fullMatch, numStr: string) => {
      const num = Number(numStr);
      if (Number.isInteger(num) && num >= 1 && num <= maxIndex) {
        citedSet.add(num);
        return fullMatch;
      }
      rejectedCitationCount++;
      return '';
    })
    .replace(/\s{2,}/g, ' ')
    .trim();

  const keyFindings: RagFinding[] = [];
  if (Array.isArray(raw.keyFindings)) {
    for (const item of raw.keyFindings) {
      if (!item || typeof item.claim !== 'string' || !item.claim.trim()) continue;
      const validCitations: number[] = [];
      for (const idx of Array.isArray(item.citationIndices) ? item.citationIndices : []) {
        if (Number.isInteger(idx) && idx >= 1 && idx <= maxIndex) {
          validCitations.push(idx);
          citedSet.add(idx);
        } else {
          rejectedCitationCount++;
        }
      }
      const uniqueCitations = Array.from(new Set(validCitations)).sort((a, b) => a - b);
      // Only include claims that remain anchored to at least one verified retrieved chunk
      if (uniqueCitations.length > 0) {
        keyFindings.push({
          claim: item.claim.trim(),
          evidenceType: item.evidenceType === 'interpretation' ? 'interpretation' : 'fact',
          citationIndices: uniqueCitations,
        });
      }
    }
  }

  if (Array.isArray(raw.usedSourceIndices)) {
    for (const idx of raw.usedSourceIndices) {
      if (Number.isInteger(idx) && idx >= 1 && idx <= maxIndex) {
        citedSet.add(idx);
      } else {
        rejectedCitationCount++;
      }
    }
  }

  const sourcesUsed: RagSourceReference[] = Array.from(citedSet)
    .sort((a, b) => a - b)
    .map((citationIndex) => {
      const res = evidencePool[citationIndex - 1];
      return {
        citationIndex,
        chunkId: res.chunkId,
        documentId: res.documentId,
        title: res.title,
        source: res.source,
        publishedDate: res.publishedDate,
        pageNumber: res.pageNumber,
        excerpt: res.excerpt,
        sourceUrl: res.sourceUrl,
        hasUploadedFile: res.hasUploadedFile,
        isDemo: res.isDemo,
      };
    });

  return {
    sanitizedSummary,
    keyFindings,
    sourcesUsed,
    rejectedCitationCount,
  };
}

function getStyleGuidance(style: AnswerStyle): string {
  switch (style) {
    case 'concise':
      return 'Write a concise summary (2 to 3 sentences maximum) with high factual density and 2 to 3 keyFindings.';
    case 'bullets':
      return 'Write a brief 1 to 2 sentence framing summary and focus detail inside 4 to 6 crisp, self-contained keyFindings.';
    case 'comparative':
      return 'Structure the summary and keyFindings to directly contrast methods, trade-offs, quantitative metrics, and conditions across the retrieved sources.';
    case 'detailed':
    default:
      return 'Write a thorough, well-structured research synthesis (4 to 6 sentences) integrating evidence across passages, paired with 3 to 5 detailed keyFindings.';
  }
}

/**
 * Generates an evidence-grounded RAG synthesis using server-side Gemini API.
 * Enforces strict citation validation against actual retrieved chunks.
 */
export async function generateGroundedAnswer(params: {
  query: string;
  results: SearchResultItem[];
  enabled: boolean;
  answerStyle: AnswerStyle;
  queryCategory: QueryCategory | null;
}): Promise<RagAnswer> {
  const { query, results, enabled, answerStyle, queryCategory } = params;

  if (!enabled) {
    return {
      status: 'skipped',
      model: null,
      answerStyle,
      queryCategory,
      summary: '',
      keyFindings: [],
      conflictingEvidence: null,
      evidenceGaps: null,
      sourcesUsed: [],
      rejectedCitationCount: 0,
      latencyMs: 0,
    };
  }

  if (results.length === 0) {
    return {
      status: 'no_evidence',
      model: serverConfig.generationModel,
      answerStyle,
      queryCategory,
      summary: 'No matching evidence passages were retrieved from the current corpus for this query.',
      keyFindings: [],
      conflictingEvidence: null,
      evidenceGaps:
        'No indexed document chunks matched your query and active filters. Adjust filters or upload relevant documents to the Document Library.',
      sourcesUsed: [],
      rejectedCitationCount: 0,
      latencyMs: 0,
    };
  }

  const start = performance.now();
  const topEvidence = results.slice(0, 6);

  const evidenceContext = topEvidence
    .map((item, idx) => {
      const num = idx + 1;
      const dateStr = item.publishedDate || 'Undated';
      const sourceStr = item.source || 'Source not specified';
      const pageStr = item.pageNumber !== null ? `, Page ${item.pageNumber}` : '';
      const demoTag = item.isDemo ? ' [Demonstration Document]' : '';
      return `[${num}] Chunk ID: ${item.chunkId}\nTitle: ${item.title}${demoTag}\nSource: ${sourceStr} (${dateStr}${pageStr})\nPassage:\n${item.chunkText || item.excerpt}`;
    })
    .join('\n\n---\n\n');

  const systemInstruction = [
    'You are Lucent, an evidence-grounded research synthesis engine.',
    `Query classification: ${queryCategory || 'unspecified'}.`,
    `Formatting directive: ${getStyleGuidance(answerStyle)}`,
    'Strict Grounding Rules:',
    '1. Answer ONLY using the numbered evidence passages [1]..[N] supplied below. Never use ungrounded outside facts.',
    '2. Cite supporting passages inline in the summary using bracketed numbers like [1] or [1][2]. Only cite indices that exist in the supplied passages.',
    '3. In keyFindings, classify each claim as either "fact" (explicitly stated in the passage) or "interpretation" (careful synthesis across passages).',
    '4. If passages contradict each other or report differing metrics/conditions, explicitly describe the conflict in conflictingEvidence; otherwise return an empty string.',
    '5. If the retrieved evidence is insufficient to fully answer the question, state that clearly in evidenceGaps.',
    '6. Never invent citations, URLs, publication dates, or source names.',
  ].join('\n');

  const prompt = `Research Question:\n${query}\n\nRetrieved Evidence Passages (1 to ${topEvidence.length}):\n${evidenceContext}`;

  try {
    const ai = getGeminiClient();
    const response = await withRetry(() =>
      ai.models.generateContent({
        model: serverConfig.generationModel,
        contents: prompt,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          systemInstruction,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              summary: {
                type: Type.STRING,
                description: 'Evidence-grounded answer with inline numbered citations [1], [2].',
              },
              keyFindings: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    claim: { type: Type.STRING },
                    evidenceType: { type: Type.STRING, enum: ['fact', 'interpretation'] },
                    citationIndices: {
                      type: Type.ARRAY,
                      items: { type: Type.INTEGER },
                    },
                  },
                  required: ['claim', 'evidenceType', 'citationIndices'],
                },
              },
              conflictingEvidence: {
                type: Type.STRING,
                description: 'Description of any contradictions across retrieved passages, or empty string if none.',
              },
              evidenceGaps: {
                type: Type.STRING,
                description: 'Explicit statement of what the supplied passages do not cover or where evidence is insufficient.',
              },
              usedSourceIndices: {
                type: Type.ARRAY,
                items: { type: Type.INTEGER },
              },
            },
            required: ['summary', 'keyFindings', 'conflictingEvidence', 'evidenceGaps', 'usedSourceIndices'],
          },
        },
      })
    );

    const rawText = (response.text || '').trim();
    if (!rawText) {
      throw new Error('Model returned an empty response payload.');
    }

    const parsed = JSON.parse(rawText) as RawGeneratedRagPayload;
    const validated = validateAndFilterCitations(parsed, topEvidence);

    return {
      status: 'generated',
      model: serverConfig.generationModel,
      answerStyle,
      queryCategory,
      summary: validated.sanitizedSummary,
      keyFindings: validated.keyFindings,
      conflictingEvidence: parsed.conflictingEvidence?.trim() || null,
      evidenceGaps: parsed.evidenceGaps?.trim() || null,
      sourcesUsed: validated.sourcesUsed,
      rejectedCitationCount: validated.rejectedCitationCount,
      latencyMs: Math.max(1, Math.round(performance.now() - start)),
    };
  } catch (err) {
    const normalized = normalizeGeminiError(err);
    return {
      status: 'error',
      model: serverConfig.generationModel,
      answerStyle,
      queryCategory,
      summary: '',
      keyFindings: [],
      conflictingEvidence: null,
      evidenceGaps: null,
      sourcesUsed: [],
      rejectedCitationCount: 0,
      latencyMs: Math.max(1, Math.round(performance.now() - start)),
      errorMessage: normalized.message,
    };
  }
}
