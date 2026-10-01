import { useEffect, useMemo, useRef, useState } from 'react';
import DOMPurify from 'dompurify';
import { marked } from 'marked';
import {
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  Atom,
  BookOpen,
  Check,
  CheckCheck,
  ChevronDown,
  Clock3,
  Copy,
  FileText,
  GitBranch,
  Globe2,
  Layers3,
  Lightbulb,
  LoaderCircle,
  Network,
  Paperclip,
  PenLine,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Square,
  X,
  Zap,
} from 'lucide-react';
import { api, errorMessage, safeUrl } from '../api';
import type { ResearchResponse, DocumentResponse } from '../types';

type Mode = 'hybrid' | 'agentic' | 'vectorless' | 'hierarchical' | 'multi_query' | 'vector' | 'bm25';
type Engine = 'auto' | 'langgraph' | 'crewai' | 'direct';
type SpeedMode = 'fast' | 'research' | 'privacy';

const examples = [
  { icon: Layers3, category: 'COMPARE & CONTRAST', title: 'RAG vs. fine-tuning', description: 'Find the right approach for a domain-specific AI assistant.', query: 'Compare retrieval-augmented generation and fine-tuning for a domain-specific AI assistant. Explain trade-offs, cost drivers, and when to use each.', color: 'purple' },
  { icon: Atom, category: 'EXPLORE A TOPIC', title: 'The next generation of batteries', description: 'Explore solid-state technology and the challenges ahead.', query: 'Explain how solid-state batteries differ from lithium-ion batteries, including energy density, safety, and manufacturing challenges.', color: 'peach' },
  { icon: Network, category: 'GO DEEPER', title: 'Inside distributed systems', description: 'Understand how systems agree, even when things fail.', query: 'How does Raft consensus handle leader election, network partitions, and recovery? Explain with practical examples.', color: 'teal' },
];

const agents = [
  { key: 'planner', name: 'Planner', role: 'Breaks down intent & hypotheses', icon: GitBranch, color: 'purple' },
  { key: 'retriever', name: 'Retriever', role: 'Hybrid RAG (FAISS + BM25 + Web)', icon: Search, color: 'blue' },
  { key: 'analyzer', name: 'Analyzer', role: 'Connects findings & themes', icon: ShieldCheck, color: 'peach' },
  { key: 'fact_checker', name: 'Fact-Checker', role: 'Verifies citations & facts', icon: CheckCheck, color: 'teal' },
  { key: 'supervisor', name: 'Supervisor', role: 'Balances trade-offs & workflow', icon: Layers3, color: 'purple' },
  { key: 'writer', name: 'Writer', role: 'Creates structured, explainable report', icon: PenLine, color: 'teal' },
] as const;

const modeDescriptions: Record<Mode, string> = {
  hybrid: 'Combines meaning-based and keyword search. A balanced starting point.',
  agentic: 'Adds corrective retrieval to help refine the available evidence.',
  vectorless: 'Uses keywords and relationships rather than vector similarity.',
  hierarchical: 'Retrieves small passages with their broader document context.',
  multi_query: 'Explores multiple versions of your question for wider coverage.',
  vector: 'Finds passages with similar meaning using embeddings.',
  bm25: 'Matches exact terms and keywords in your documents.',
};

interface Props {
  onShowToast: (message: string) => void;
  onResult: (result: ResearchResponse) => void;
  result: ResearchResponse | null;
  resetKey: number;
  onBusyChange: (busy: boolean) => void;
  onOpenLibrary: () => void;
}

export function StudioTab({ onShowToast, onResult, result, resetKey, onBusyChange, onOpenLibrary }: Props) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const [ragMode, setRagMode] = useState<Mode>('hybrid');
  const [engine, setEngine] = useState<Engine>('auto');
  const [speedMode, setSpeedMode] = useState<SpeedMode>('fast');
  const [tab, setTab] = useState<'report' | 'sources' | 'activity'>('report');
  const [copied, setCopied] = useState(false);
  const [stopped, setStopped] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [streamingReport, setStreamingReport] = useState('');
  const [stageMessage, setStageMessage] = useState('');

  const controller = useRef<AbortController | null>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const reportRef = useRef<HTMLElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const copyTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lock = useRef(false);

  const uploadedDocCache = useRef<Map<string, string>>(new Map());

  useEffect(() => { if (resetKey) textarea.current?.focus(); }, [resetKey]);
  useEffect(() => () => { controller.current?.abort(); if (copyTimer.current) clearTimeout(copyTimer.current); }, []);

  const reportHtml = useMemo(() => {
    if (!result?.report) return '';
    return DOMPurify.sanitize(marked.parse(result.report, { async: false }), {
      FORBID_TAGS: ['style', 'form', 'input'],
      FORBID_ATTR: ['style'],
    });
  }, [result?.report]);

  const streamingReportHtml = useMemo(() => {
    if (!streamingReport) return '';
    return DOMPurify.sanitize(marked.parse(streamingReport, { async: false }), {
      FORBID_TAGS: ['style', 'form', 'input'],
      FORBID_ATTR: ['style'],
    });
  }, [streamingReport]);

  function handleFileSelect(event: React.ChangeEvent<HTMLInputElement>) {
    const files = event.target.files;
    if (!files || !files.length) return;
    const list = Array.from(files);
    setAttachedFiles(prev => [...prev, ...list]);
    onShowToast(`Attached ${list.length} file${list.length > 1 ? 's' : ''}`);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  function removeAttachment(index: number) {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  }

  async function executeResearch() {
    if (lock.current) return;
    if (query.trim().length < 3) {
      setError('Add a question with at least 3 characters to get started.');
      textarea.current?.focus();
      return;
    }
    lock.current = true;
    setLoading(true);
    onBusyChange(true);
    setError('');
    setStopped(false);
    setElapsed(0);
    const abort = new AbortController();
    controller.current = abort;
    const started = Date.now();
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    const timeout = setTimeout(() => abort.abort('timeout'), 120_000);

    try {
      // If user attached files directly, upload them once and cache document_id
      const uploadedAttachmentIds: string[] = [];
      if (attachedFiles.length > 0) {
        for (const file of attachedFiles) {
          const fileKey = `${file.name}_${file.size}_${file.lastModified}`;
          if (uploadedDocCache.current.has(fileKey)) {
            uploadedAttachmentIds.push(uploadedDocCache.current.get(fileKey)!);
            continue;
          }
          const form = new FormData();
          form.append('file', file);
          try {
            setStageMessage(`Parsing & indexing attachment: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)...`);
            const upRes = await api<DocumentResponse>('/api/v1/documents/upload', {
              method: 'POST',
              body: form,
              signal: abort.signal,
            });
            if (upRes?.document_id) {
              uploadedDocCache.current.set(fileKey, upRes.document_id);
              uploadedAttachmentIds.push(upRes.document_id);
            }
          } catch (uploadErr) {
            console.warn(`File upload skipped for ${file.name}:`, uploadErr);
          }
        }
        // Once attached & cached into library, clear file chips so future questions don't re-upload
        setAttachedFiles([]);
      }

      setStreamingReport('');
      setStageMessage('Planner decomposing intent and hypothesis...');

      let data: ResearchResponse | null = null;
      try {
        const streamRes = await fetch('/api/v1/research/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: abort.signal,
          body: JSON.stringify({
            query: query.trim(),
            mode: speedMode,
            rag_mode: ragMode,
            orchestrator: speedMode === 'fast' ? 'direct' : (speedMode === 'privacy' ? 'direct' : engine),
            privacy_mode: speedMode === 'privacy',
            attachment_ids: uploadedAttachmentIds,
          }),
        });

        if (streamRes.ok && streamRes.body) {
          const reader = streamRes.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';

            for (const line of lines) {
              const trimmed = line.trim();
              if (trimmed.startsWith('data: ')) {
                try {
                  const ev = JSON.parse(trimmed.slice(6));
                  if (ev.type === 'stage') {
                    setStageMessage(ev.message || `Agent stage: ${ev.stage}`);
                  } else if (ev.type === 'token') {
                    setStreamingReport(ev.accumulated || (prev => prev + (ev.token || '')));
                  } else if (ev.type === 'complete') {
                    data = ev.response;
                  }
                } catch {
                  // Ignore JSON parse error on malformed chunks
                }
              }
            }
          }
        }
      } catch (streamErr) {
        if (abort.signal.aborted) throw streamErr;
        console.warn('Streaming connection failed, falling back to sync endpoint:', streamErr);
      }

      // If streaming didn't produce complete response (or failed), fallback to sync
      if (!data) {
        data = await api<ResearchResponse>('/api/v1/research/sync', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: abort.signal,
          body: JSON.stringify({
            query: query.trim(),
            mode: speedMode,
            rag_mode: ragMode,
            orchestrator: speedMode === 'fast' ? 'direct' : (speedMode === 'privacy' ? 'direct' : engine),
            privacy_mode: speedMode === 'privacy',
            attachment_ids: uploadedAttachmentIds,
          }),
        });
      }

      if (abort.signal.aborted) return;
      if (data.status !== 'completed') {
        throw new Error(data.report || 'Research did not complete. Please try again.');
      }
      if (!data.report?.trim()) {
        throw new Error('The server completed without a report. Try a more specific question.');
      }

      const clientDuration = (Date.now() - started) / 1000;
      data.processing_time_seconds = Number(clientDuration.toFixed(2));

      setStreamingReport('');
      setStageMessage('');
      onResult(data);
      setTab('report');
      onShowToast('Your research report is ready.');

      // Instantly scroll smoothly to the report
      setTimeout(() => {
        reportRef.current?.scrollIntoView({
          behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
          block: 'start',
        });
      }, 50);
    } catch (err) {
      if (abort.signal.aborted && abort.signal.reason !== 'timeout') {
        setStopped(true);
      } else {
        setError(abort.signal.reason === 'timeout'
          ? 'The request timed out after 2 minutes. Try again with a more focused question.'
          : errorMessage(err));
      }
    } finally {
      clearInterval(timer);
      clearTimeout(timeout);
      controller.current = null;
      lock.current = false;
      setLoading(false);
      onBusyChange(false);
    }
  }

  async function copyReport() {
    try {
      await navigator.clipboard.writeText(result?.report || '');
      setCopied(true);
      onShowToast('Report copied as Markdown.');
      if (copyTimer.current) clearTimeout(copyTimer.current);
      copyTimer.current = setTimeout(() => setCopied(false), 2500);
    } catch {
      onShowToast('Clipboard access is unavailable. Use Download instead.');
    }
  }

  function downloadReport() {
    const blob = new Blob([result?.report || ''], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `research-${result?.task_id || 'report'}.md`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <div className="studio-page">
      <section className="studio-heading">
        <div className="eyebrow"><span className="eyebrow-line" /> YOUR CURIOSITY, AMPLIFIED</div>
        <h1>Big questions.<br /><span>Clearer answers.</span></h1>
        <p>A team of AI agents to explore, connect, and make sense of what matters.<br className="desktop-break" /> Start with a question. Leave with a research report.</p>
        <div className="hero-orbit" aria-hidden="true">
          <span className="orbit-ring ring-one" />
          <span className="orbit-ring ring-two" />
          <span className="orbit-ring ring-three" />
          <span className="orbit-center"><Sparkles size={30} /></span>
          <span className="orbit-point point-one"><Search size={16} /></span>
          <span className="orbit-point point-two"><FileText size={16} /></span>
          <span className="orbit-point point-three"><GitBranch size={16} /></span>
          <span className="orbit-dot" />
        </div>
      </section>

      <div className="studio-grid">
        <div className="studio-main">
          <section className="composer panel" aria-labelledby="question-label">
            <div className="composer-header">
              <label id="question-label" htmlFor="research-question">
                <Sparkles size={16} /> What would you like to understand?
              </label>
              <div className="mode-pill-container" role="radiogroup" aria-label="Research Speed and Depth">
                <button
                  type="button"
                  role="radio"
                  aria-checked={speedMode === 'fast'}
                  className={`mode-pill ${speedMode === 'fast' ? 'active' : ''}`}
                  onClick={() => setSpeedMode('fast')}
                  title="Fast Mode: Sub-5s turnaround with concise tables & pointwise takeaways"
                >
                  ⚡ Fast (&lt;5s)
                </button>
                <button
                  type="button"
                  role="radio"
                  aria-checked={speedMode === 'research'}
                  className={`mode-pill ${speedMode === 'research' ? 'active' : ''}`}
                  onClick={() => setSpeedMode('research')}
                  title="Researched Mode: Deep multi-agent deliberation and exhaustive verification"
                >
                  🔬 Deep Research
                </button>
                <button
                  type="button"
                  role="radio"
                  aria-checked={speedMode === 'privacy'}
                  className={`mode-pill privacy ${speedMode === 'privacy' ? 'active' : ''}`}
                  onClick={() => setSpeedMode('privacy')}
                  title="Private Mode: 100% Air-Gapped. Zero cloud API calls (Gemini/Groq disabled). Runs local RAG on your machine."
                >
                  🔒 Private (Air-Gapped)
                </button>
              </div>
            </div>

            {speedMode === 'privacy' && (
              <div className="privacy-security-banner">
                <ShieldCheck size={16} />
                <div>
                  <strong>100% Air-Gapped Corporate Privacy Active:</strong> Cloud APIs (Gemini, Groq, Web Search) are blocked. Queries and documents are processed locally on your machine using local embeddings and offline RAG synthesis.
                </div>
              </div>
            )}

            <textarea
              id="research-question"
              ref={textarea}
              placeholder="Ask a question, explore an idea, or compare different perspectives..."
              value={query}
              maxLength={1500}
              disabled={loading}
              aria-describedby="question-hint"
              onChange={event => {
                setQuery(event.target.value);
                if (error) setError('');
              }}
              onKeyDown={event => {
                if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                  event.preventDefault();
                  void executeResearch();
                }
              }}
            />

            {attachedFiles.length > 0 && (
              <div className="attachment-chips">
                {attachedFiles.map((file, idx) => (
                  <span key={`${file.name}-${idx}`} className="attachment-chip">
                    <Paperclip size={12} />
                    <span>{file.name}</span>
                    <small>({(file.size / 1024).toFixed(0)} KB)</small>
                    <button type="button" onClick={() => removeAttachment(idx)} aria-label="Remove attachment">
                      <X size={12} />
                    </button>
                  </span>
                ))}
              </div>
            )}

            <div className="composer-hint" id="question-hint">
              <span>Try adding context, a goal, or a specific comparison.</span>
              <span>{query.length}<span className="muted"> / 1,500</span></span>
            </div>

            <div className="composer-footer">
              <input
                type="file"
                ref={fileInputRef}
                multiple
                accept="image/*,audio/*,.pdf,.docx,.txt,.csv,.json,.html,.htm,.md,.xlsx,.tsv,.py,.js,.ts,.sh,.yaml,.yml,.xml"
                style={{ display: 'none' }}
                onChange={handleFileSelect}
              />
              <button
                type="button"
                className="attach-button"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading}
                title="Attach photos, audio recordings, JSON, PDF, CSV or documents"
              >
                <Paperclip size={16} />
                <span>{attachedFiles.length > 0 ? `${attachedFiles.length} attached` : 'Add files / media'}</span>
              </button>
              <div className="composer-submit">
                <kbd>Ctrl / ⌘ ↵</kbd>
                <button
                  className="button primary"
                  disabled={loading || query.trim().length < 3}
                  onClick={() => void executeResearch()}
                >
                  {loading ? (
                    <><LoaderCircle size={16} className="spin" /> Researching</>
                  ) : (
                    <>Start research <ArrowRight size={16} /></>
                  )}
                </button>
              </div>
            </div>
          </section>

          {error && (
            <div className="alert error" role="alert">
              <div>
                <strong>We couldn't complete that request</strong>
                <p>{error}</p>
              </div>
              <button className="button small" onClick={() => void executeResearch()} disabled={loading}>
                Try again
              </button>
            </div>
          )}

          {stopped && (
            <div className="alert" role="status">
              You stopped waiting. The backend may still finish this request; no server-side cancellation is available.
            </div>
          )}

          {loading && (
            <section className="research-progress panel" aria-live="polite">
              <div className="progress-title">
                <LoaderCircle size={20} className="spin" />
                <div>
                  <strong>Your research is underway</strong>
                  <p>
                    {stageMessage || (speedMode === 'privacy'
                      ? 'Air-Gapped Privacy Mode: Local RAG synthesis underway (0 cloud egress)...'
                      : speedMode === 'fast'
                      ? 'Fast mode active: Synthesizing quick response with Gemini Flash...'
                      : 'Deep research active: Multi-agent coordination in progress...')}
                  </p>
                </div>
                <span className="elapsed" aria-hidden="true">{elapsed}s</span>
              </div>
              <div className="progress-track"><span /></div>
              <div className="progress-footer">
                <span>Multi-agent workflow: Planner &rarr; Retriever &rarr; Analyzer &rarr; Fact-Checker &rarr; Supervisor &rarr; Writer.</span>
                <button className="text-button" onClick={() => controller.current?.abort()}>
                  <Square size={12} /> Stop waiting
                </button>
              </div>
            </section>
          )}

          {/* Live streaming preview container */}
          {loading && streamingReport && (
            <section className="report-panel panel live-streamed" aria-label="Streaming research preview">
              <div className="report-heading">
                <div>
                  <span className="eyebrow live-badge"><Sparkles size={14} className="spin-slow" /> STREAMING LIVE IN REAL-TIME</span>
                  <h2>{query}</h2>
                </div>
              </div>
              <div className="result-content" style={{ padding: '24px' }}>
                <div className="prose" dangerouslySetInnerHTML={{ __html: streamingReportHtml }} />
                <div className="streaming-cursor-container">
                  <span className="streaming-cursor">▍</span>
                  <span className="streaming-hint">Streaming tokens progressively...</span>
                </div>
              </div>
            </section>
          )}

          {/* Report Panel is rendered DIRECTLY HERE when result is available for maximum visibility! */}
          {result && (
            <section ref={reportRef} className="report-panel panel" aria-label="Research results">
              <div className="report-heading">
                <div>
                  <span className="eyebrow"><CheckCheck size={14} /> RESEARCH COMPLETE</span>
                  <h2>{result.query}</h2>
                </div>
                <div className="report-actions">
                  <button className="icon-button" aria-label="Copy report" title="Copy Markdown" onClick={() => void copyReport()}>
                    {copied ? <Check size={17} /> : <Copy size={17} />}
                  </button>
                  <button className="button small" onClick={downloadReport}>
                    <ArrowDownToLine size={15} /> Download
                  </button>
                </div>
              </div>
              <div className="report-summary">
                <span><Clock3 size={14} /> {result.processing_time_seconds.toFixed(2)}s</span>
                <span><BookOpen size={14} /> {result.sources?.length || 0} sources returned</span>
                <span><Zap size={14} /> {result.mode === 'privacy' || speedMode === 'privacy' ? '🔒 Air-Gapped Local Mode' : speedMode === 'fast' ? '⚡ Fast Mode (<5s)' : '🔬 Deep Research'}</span>
                <span>Engine: {result.mode === 'privacy' || speedMode === 'privacy' ? 'Local RAG (0 Cloud Egress)' : result.orchestrator === 'direct' ? 'Jev System-1 Direct' : result.orchestrator}</span>
                {result.response_accuracy_score && (
                  <span>Accuracy: {(result.response_accuracy_score * 100).toFixed(1)}%</span>
                )}
              </div>
              <div className="result-tabs" role="tablist" aria-label="Research result views">
                {(['report', 'sources', 'activity'] as const).map((item) => (
                  <button
                    key={item}
                    role="tab"
                    id={`tab-${item}`}
                    aria-controls={`panel-${item}`}
                    aria-selected={tab === item}
                    tabIndex={tab === item ? 0 : -1}
                    onClick={() => setTab(item)}
                    className={tab === item ? 'active' : ''}
                  >
                    {item === 'report' ? 'Research report' : item === 'sources' ? `Sources (${result.sources?.length || 0})` : 'Agent activity'}
                  </button>
                ))}
              </div>
              <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} tabIndex={0} className="result-content">
                {tab === 'report' && (
                  <>
                    <div className="prose" dangerouslySetInnerHTML={{ __html: reportHtml }} />
                    <p className="report-disclaimer">
                      <ShieldCheck size={14} /> Generated with multi-agent verification. Review original sources before relying on conclusions.
                    </p>
                  </>
                )}
                {tab === 'sources' && (
                  <div className="source-list">
                    {result.sources?.length ? (
                      result.sources.map((source, index) => (
                        <article className="source-card" key={`${source.url_or_path}-${index}`}>
                          <span className="source-number">{String(index + 1).padStart(2, '0')}</span>
                          <div>
                            <span className="subtle-badge">{source.source_type}</span>
                            <h3>{source.title || source.url_or_path || 'Untitled source'}</h3>
                            <p>{source.snippet || 'No excerpt was returned for this source.'}</p>
                            {safeUrl(source.url_or_path) ? (
                              <a href={safeUrl(source.url_or_path)} target="_blank" rel="noopener noreferrer">
                                Open original source <ArrowUpRight size={13} />
                              </a>
                            ) : (
                              <span className="source-path">{source.url_or_path}</span>
                            )}
                          </div>
                        </article>
                      ))
                    ) : (
                      <div className="empty-state">
                        <BookOpen size={28} />
                        <h3>General Model Synthesis</h3>
                        <p>This report was synthesized from model knowledge. Upload documents via "Add files / media" to ground with specific citations.</p>
                      </div>
                    )}
                  </div>
                )}
                {tab === 'activity' && (
                  <>
                    <p className="section-description">Timings and execution metrics reported by the backend for each coordinated agent.</p>
                    <div className="timing-list">
                      {agents.map(({ key, name, icon: Icon }) => {
                        const timing = result.telemetry?.[`${key}_time_ms`];
                        return (
                          <div key={key}>
                            <span><Icon size={16} />{name}</span>
                            <strong>{typeof timing === 'number' && timing > 0 ? (timing < 1 ? `${timing.toFixed(1)} ms` : `${timing.toFixed(0)} ms`) : (key === 'planner' ? 'Active (Direct Plan)' : 'Active / Evaluated')}</strong>
                          </div>
                        );
                      })}
                    </div>
                    <div className="alert" style={{ marginTop: '16px' }}>
                      Backend accuracy score: {(result.response_accuracy_score * 100).toFixed(1)}% &middot; Synthesis speedup: {(result.synthesis_speedup_ratio * 100).toFixed(0)}%
                    </div>
                  </>
                )}
              </div>
            </section>
          )}

          {/* Research Team Grid */}
          <section className="team-panel panel" aria-labelledby="team-title">
            <div className="section-header">
              <h2 id="team-title"><span className="team-dot" /> Your research team</h2>
              <span>Six coordinated agents. One clear outcome.</span>
            </div>
            <div className="agent-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))' }}>
              {agents.map(({ key, name, role, icon: Icon, color }, index) => (
                <div key={key} className="agent-step">
                  <div className="agent-step-top">
                    <span className={`agent-icon ${color}`}><Icon size={18} /></span>
                    <span className="agent-number">0{index + 1}</span>
                  </div>
                  <strong>{name}</strong>
                  <p>{role}</p>
                  {index < agents.length - 1 && <ChevronDown className="agent-connector" size={14} />}
                </div>
              ))}
            </div>
            <div className="team-caption">
              <ShieldCheck size={13} />
              <span>Coordinated across Planning, Retrieval, Analysis, Fact-Checking, Supervision, and Synthesis.</span>
            </div>
          </section>

          {!result && !loading && (
            <section className="inspiration" aria-labelledby="inspiration-title">
              <div className="section-header">
                <h2 id="inspiration-title">A little inspiration</h2>
                <span>Pick a starting point <ArrowDownToLine size={12} /></span>
              </div>
              <div className="example-grid">
                {examples.map(({ icon: Icon, ...item }) => (
                  <button
                    className="example-card"
                    key={item.title}
                    onClick={() => {
                      setQuery(item.query);
                      setError('');
                      textarea.current?.focus();
                    }}
                  >
                    <span className={`example-icon ${item.color}`}><Icon size={19} /></span>
                    <span className="example-category">{item.category}</span>
                    <strong>{item.title}</strong>
                    <p>{item.description}</p>
                    <ArrowUpRight className="example-arrow" size={16} />
                  </button>
                ))}
              </div>
            </section>
          )}
        </div>

        <aside className="research-settings" aria-label="Research settings">
          <section className="settings-panel panel">
            <div className="settings-title">
              <SlidersHorizontal size={16} />
              <h2>Make it your own</h2>
            </div>
            <p className="settings-intro">A good starting point, with room to fine-tune.</p>

            <fieldset disabled={loading}>
              <legend>RETRIEVAL APPROACH</legend>
              <div className="recommendation">
                <span className="recommendation-icon"><Sparkles size={17} /></span>
                <div>
                  <strong>{ragMode === 'hybrid' ? 'Balanced research' : 'Custom research'}</strong>
                  <span>{ragMode === 'hybrid' ? 'Hybrid search · Recommended' : `${ragMode.replace('_', ' ')} search`}</span>
                </div>
                <Check size={15} />
              </div>
              <p className="setting-description">{modeDescriptions[ragMode]}</p>

              <details className="advanced-settings">
                <summary>Advanced settings <ChevronDown size={14} /></summary>
                <div className="field">
                  <label htmlFor="retrieval-mode">Retrieval mode</label>
                  <select id="retrieval-mode" value={ragMode} onChange={event => setRagMode(event.target.value as Mode)}>
                    {Object.keys(modeDescriptions).map(mode => (
                      <option key={mode} value={mode}>
                        {({
                          hybrid: 'Hybrid (recommended)',
                          agentic: 'Agentic / corrective',
                          vectorless: 'Vectorless / graph',
                          hierarchical: 'Hierarchical',
                          multi_query: 'Multi-query',
                          vector: 'Dense vectors',
                          bm25: 'Keyword / BM25',
                        } as Record<string, string>)[mode]}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="engine">Orchestration engine</label>
                  <select id="engine" value={engine} onChange={event => setEngine(event.target.value as Engine)}>
                    <option value="auto">Auto (Jev System-1 Optimized)</option>
                    <option value="direct">Direct (&lt;5s Low Latency)</option>
                    <option value="langgraph">LangGraph</option>
                    <option value="crewai">CrewAI</option>
                  </select>
                  <p>The backend automatically orchestrates between LangGraph, CrewAI, and Jev Direct.</p>
                </div>
                <button
                  type="button"
                  className="text-button"
                  onClick={() => {
                    setRagMode('hybrid');
                    setEngine('auto');
                    setSpeedMode('fast');
                  }}
                >
                  Restore recommended settings
                </button>
              </details>
            </fieldset>

            <div className="settings-divider" />
            <div className="settings-label">YOUR RESEARCH TOOLKIT</div>
            <div className="toolkit-item">
              <Globe2 size={17} />
              <div>
                <strong>Web & research sources</strong>
                <span>{speedMode === 'privacy' ? '🔒 Blocked (Air-Gapped Privacy Active)' : 'Available when configured on the server'}</span>
              </div>
            </div>
            <div className="toolkit-item">
              <BookOpen size={17} />
              <div>
                <strong>Your knowledge library</strong>
                <span>Add documents, photos, audio & data to give agents context</span>
              </div>
            </div>
            <button type="button" className="button library-button" onClick={onOpenLibrary}>
              <Paperclip size={14} /> Manage documents <ArrowUpRight size={13} />
            </button>

            <div className="settings-divider" />
            <div className="settings-label">WHAT YOU'LL GET</div>
            <ul className="outcomes">
              <li><Check size={14} /> A structured research report</li>
              <li><Check size={14} /> Sources returned by the agents</li>
              <li><Check size={14} /> Comparative tables & pointwise insights</li>
              <li><Check size={14} /> Markdown you can take anywhere</li>
            </ul>
          </section>

          <div className="tip-card">
            <Lightbulb size={18} />
            <div>
              <strong>Better questions, better research.</strong>
              <p>Be specific about what you're exploring and why. Context helps your agents focus.</p>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
