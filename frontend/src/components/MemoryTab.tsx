import { useCallback, useEffect, useState, type FormEvent } from 'react';
import {
  Brain,
  ChevronDown,
  ChevronUp,
  Clock3,
  Database,
  FileText,
  History,
  KeyRound,
  LoaderCircle,
  LogOut,
  MessageSquare,
  Pin,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  Trash2,
} from 'lucide-react';
import {
  clearShortTermMemoryApi,
  deleteLongTermMemoryApi,
  deleteThreadApi,
  errorMessage,
  fetchMemoryApi,
  fetchSessionsApi,
  fetchThreadsApi,
  logoutAllApi,
  pinMemoryApi,
} from '../api';
import type {
  ChatThreadItem,
  ChatTurnItem,
  SessionItem,
  User,
  UserMemoryPayload,
  WorkspaceStats,
} from '../types';

interface Props {
  currentUser: User | null;
  onShowToast: (message: string) => void;
  onOpenAuth: () => void;
  onOpenTurn: (turn: ChatTurnItem, threadId: string) => void;
  onContinueThread: (threadId: string) => void;
  onAuthRevokedAll: () => void;
  refreshSignal: number;
}

export function MemoryTab({
  currentUser,
  onShowToast,
  onOpenAuth,
  onOpenTurn,
  onContinueThread,
  onAuthRevokedAll,
  refreshSignal,
}: Props) {
  const [threads, setThreads] = useState<ChatThreadItem[]>([]);
  const [stats, setStats] = useState<WorkspaceStats | null>(null);
  const [memoryData, setMemoryData] = useState<UserMemoryPayload | null>(null);
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [expandedThreadId, setExpandedThreadId] = useState<string | null>(null);
  const [pinTitle, setPinTitle] = useState('');
  const [pinSummary, setPinSummary] = useState('');
  const [pinKind, setPinKind] = useState<'user_preference' | 'pinned_note'>('user_preference');
  const [savingPin, setSavingPin] = useState(false);

  const loadWorkspace = useCallback(async (searchNeedle?: string) => {
    if (!currentUser) {
      setThreads([]);
      setStats(null);
      setMemoryData(null);
      setSessions([]);
      return;
    }
    setLoading(true);
    setError('');
    try {
      const [threadsRes, memRes, sessionsRes] = await Promise.all([
        fetchThreadsApi(searchNeedle),
        fetchMemoryApi(),
        fetchSessionsApi(),
      ]);
      setThreads(threadsRes.threads || []);
      setStats(memRes.stats || threadsRes.stats || null);
      setMemoryData(memRes);
      setSessions(sessionsRes || []);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, [currentUser]);

  useEffect(() => {
    void loadWorkspace();
  }, [loadWorkspace, currentUser?.id, refreshSignal]);

  async function handleSearchSubmit(e: FormEvent) {
    e.preventDefault();
    await loadWorkspace(search);
  }

  async function handlePinSubmit(e: FormEvent) {
    e.preventDefault();
    if (!pinTitle.trim() || !pinSummary.trim() || savingPin) return;
    setSavingPin(true);
    try {
      await pinMemoryApi(pinTitle.trim(), pinSummary.trim(), pinKind);
      setPinTitle('');
      setPinSummary('');
      onShowToast('Saved to your private long-term memory.');
      await loadWorkspace(search);
    } catch (err) {
      onShowToast(errorMessage(err));
    } finally {
      setSavingPin(false);
    }
  }

  async function handleDeleteThread(tid: string) {
    try {
      await deleteThreadApi(tid);
      setThreads((prev) => prev.filter((t) => t.thread_id !== tid));
      onShowToast('Thread removed from your private database.');
      await loadWorkspace(search);
    } catch (err) {
      onShowToast(errorMessage(err));
    }
  }

  async function handleDeleteLongTerm(memId: string) {
    try {
      await deleteLongTermMemoryApi(memId);
      onShowToast('Long-term memory removed.');
      await loadWorkspace(search);
    } catch (err) {
      onShowToast(errorMessage(err));
    }
  }

  async function handleClearShortTerm() {
    try {
      const res = await clearShortTermMemoryApi();
      onShowToast(`Cleared ${res.cleared_count} short-term working memory items.`);
      await loadWorkspace(search);
    } catch (err) {
      onShowToast(errorMessage(err));
    }
  }

  async function handleRevokeAllSessions() {
    try {
      const res = await logoutAllApi();
      onShowToast(`Signed out across ${res.revoked_count} active sessions.`);
      onAuthRevokedAll();
    } catch (err) {
      onShowToast(errorMessage(err));
    }
  }

  if (!currentUser) {
    return (
      <div className="secondary-page">
        <div className="page-heading">
          <div className="eyebrow">ISOLATED USER MEMORY &amp; PAST CHATS</div>
          <h1>Your own private database.<br />Every chat &amp; insight remembered.</h1>
          <p>Sign in or create an account to provision a dedicated MongoDB database strictly isolated for your chats, uploaded files, short-term buffer, and long-term LLM context.</p>
        </div>
        <section className="panel memory-guest-card">
          <ShieldCheck size={34} />
          <h2>Sign in to unlock your personal memory vault</h2>
          <p>Each registered account gets a separate database (`db_name`) so nobody else can view your questions, answers, files, or remembered preferences—and the AI recalls your context whenever you return.</p>
          <button type="button" className="button primary" onClick={onOpenAuth}>
            Sign In / Create Account
          </button>
        </section>
      </div>
    );
  }

  const shortTerm = memoryData?.short_term || [];
  const longTerm = memoryData?.long_term || [];

  return (
    <div className="secondary-page">
      <div className="page-heading">
        <div className="eyebrow">PRIVATE MONGODB WORKSPACE</div>
        <h1>{currentUser.full_name.split(' ')[0]}&apos;s Memory &amp; Past Chats</h1>
        <p>Every conversation, linked file Q&amp;A, short-term context window, and long-term knowledge record stored inside your isolated database.</p>
      </div>

      <section className="panel workspace-db-card">
        <div className="workspace-db-top">
          <div className="workspace-db-badge">
            <Database size={18} />
            <div>
              <strong>Database: <code>{currentUser.db_name || stats?.db_name}</code></strong>
              <span>Backend: {stats?.backend === 'pymongo' ? 'Official MongoDB Atlas (PyMongo Stable API v1)' : (stats?.backend || 'mongo')} · Authenticated User: {currentUser.email}</span>
            </div>
          </div>
          <button type="button" className="icon-button" onClick={() => void loadWorkspace(search)} disabled={loading} aria-label="Refresh workspace">
            <RefreshCw size={16} className={loading ? 'spin' : ''} />
          </button>
        </div>

        <div className="workspace-stats-grid">
          <div className="workspace-stat">
            <span>Past Threads</span>
            <strong>{stats?.total_threads ?? threads.length}</strong>
          </div>
          <div className="workspace-stat">
            <span>Total Q&amp;As</span>
            <strong>{stats?.total_chats ?? 0}</strong>
          </div>
          <div className="workspace-stat">
            <span>Uploaded Files</span>
            <strong>{stats?.uploaded_files ?? 0}</strong>
          </div>
          <div className="workspace-stat">
            <span>Short-Term Buffer</span>
            <strong>{memoryData?.short_term_count ?? shortTerm.length}</strong>
          </div>
          <div className="workspace-stat">
            <span>Long-Term Memory</span>
            <strong>{memoryData?.long_term_count ?? longTerm.length}</strong>
          </div>
        </div>
      </section>

      {error && (
        <div className="alert error" role="alert">
          <div>
            <strong>Could not load workspace history</strong>
            <p>{error}</p>
          </div>
          <button type="button" className="button small" onClick={() => void loadWorkspace(search)}>Retry</button>
        </div>
      )}

      {/* ── Section 1: Private Past Chats & Threads ── */}
      <section className="panel memory-section">
        <div className="memory-section-header">
          <div>
            <h2><History size={18} /> Private Past Chats &amp; Threads</h2>
            <p>Click any past question to inspect the full report or continue asking follow-up questions in the same thread.</p>
          </div>
          <form className="memory-search-form" onSubmit={handleSearchSubmit}>
            <label className="search-field">
              <Search size={15} />
              <input
                aria-label="Search past chats"
                placeholder="Search past chats or files..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </label>
            <button type="submit" className="button small">Search</button>
          </form>
        </div>

        {loading && threads.length === 0 ? (
          <div className="empty-state">
            <LoaderCircle size={24} className="spin" />
            <p>Loading your private threads...</p>
          </div>
        ) : threads.length === 0 ? (
          <div className="empty-state">
            <MessageSquare size={26} />
            <h3>No saved chats yet</h3>
            <p>Ask a question in Research Studio while signed in and your Q&amp;As will appear privately here.</p>
          </div>
        ) : (
          <div className="thread-list">
            {threads.map((thread) => {
              const isOpen = expandedThreadId === thread.thread_id;
              return (
                <article key={thread.thread_id} className="thread-card">
                  <div className="thread-card-header">
                    <button
                      type="button"
                      className="thread-toggle-btn"
                      onClick={() => setExpandedThreadId(isOpen ? null : thread.thread_id)}
                    >
                      <div className="thread-title-group">
                        <strong>{thread.title}</strong>
                        <div className="thread-meta-row">
                          <span><Clock3 size={12} /> {new Date(thread.updated_at).toLocaleString()}</span>
                          <span>{thread.turn_count} {thread.turn_count === 1 ? 'turn' : 'turns'}</span>
                          <span>{thread.mode}</span>
                          {thread.file_names?.map((fn) => (
                            <span key={fn} className="subtle-badge"><FileText size={11} /> {fn}</span>
                          ))}
                        </div>
                      </div>
                      {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </button>
                    <div className="thread-actions">
                      <button
                        type="button"
                        className="button small"
                        onClick={() => {
                          const lastTurn = thread.turns[thread.turns.length - 1];
                          if (lastTurn) onOpenTurn(lastTurn, thread.thread_id);
                          else onContinueThread(thread.thread_id);
                        }}
                      >
                        Open &amp; continue
                      </button>
                      <button
                        type="button"
                        className="icon-button danger"
                        aria-label={`Delete thread ${thread.title}`}
                        onClick={() => void handleDeleteThread(thread.thread_id)}
                      >
                        <Trash2 size={15} />
                      </button>
                    </div>
                  </div>
                  {!isOpen && thread.latest_summary && (
                    <p className="thread-preview">{thread.latest_summary}</p>
                  )}
                  {isOpen && (
                    <div className="thread-turns-list">
                      {thread.turns.map((turn, idx) => (
                        <div key={turn.chat_id} className="thread-turn-item">
                          <div className="thread-turn-text">
                            <span className="turn-index">Turn {idx + 1}</span>
                            <strong>Q: {turn.query}</strong>
                            {turn.report_summary && <p>A: {turn.report_summary}</p>}
                            <small>
                              {new Date(turn.created_at).toLocaleString()}
                              {turn.attachment_names?.length ? ` · Files: ${turn.attachment_names.join(', ')}` : ''}
                            </small>
                          </div>
                          <button
                            type="button"
                            className="button small"
                            onClick={() => onOpenTurn(turn, thread.thread_id)}
                          >
                            View report
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        )}
      </section>

      {/* ── Section 2: Short-Term & Long-Term Memory Grid ── */}
      <div className="memory-columns">
        <section className="panel memory-section">
          <div className="memory-section-header">
            <div>
              <h2><Sparkles size={18} /> Short-Term Working Memory</h2>
              <p>Recent Q&amp;As kept in active sliding context so follow-up questions (&ldquo;explain that further&rdquo;) stay seamless.</p>
            </div>
            {shortTerm.length > 0 && (
              <button type="button" className="button small" onClick={() => void handleClearShortTerm()}>
                Clear buffer
              </button>
            )}
          </div>
          {shortTerm.length === 0 ? (
            <div className="empty-state compact">
              <p>Short-term buffer is empty.</p>
            </div>
          ) : (
            <div className="memory-list">
              {shortTerm.map((item) => (
                <div key={item.memory_id} className="memory-item-card">
                  <div className="memory-item-head">
                    <strong>{item.query}</strong>
                    <small>{new Date(item.created_at).toLocaleString()}</small>
                  </div>
                  <p>{item.answer_summary}</p>
                  {item.files_used?.length > 0 && (
                    <div className="memory-tags">
                      {item.files_used.map((f) => <span key={f} className="subtle-badge">{f}</span>)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="panel memory-section">
          <div className="memory-section-header">
            <div>
              <h2><Brain size={18} /> Long-Term Persistent Memory</h2>
              <p>Facts, recurring topics, document insights &amp; preferences remembered permanently across sessions.</p>
            </div>
          </div>

          <form className="pin-memory-form" onSubmit={handlePinSubmit}>
            <div className="pin-form-row">
              <input
                type="text"
                placeholder="Memory title (e.g. Coding preference)"
                value={pinTitle}
                onChange={(e) => setPinTitle(e.target.value)}
                required
              />
              <select value={pinKind} onChange={(e) => setPinKind(e.target.value as 'user_preference' | 'pinned_note')}>
                <option value="user_preference">User Preference</option>
                <option value="pinned_note">Pinned Note</option>
              </select>
            </div>
            <div className="pin-form-row">
              <input
                type="text"
                placeholder="What should the AI always remember about you or your projects?"
                value={pinSummary}
                onChange={(e) => setPinSummary(e.target.value)}
                required
              />
              <button type="submit" className="button small" disabled={savingPin}>
                <Pin size={14} /> {savingPin ? 'Saving...' : 'Remember'}
              </button>
            </div>
          </form>

          {longTerm.length === 0 ? (
            <div className="empty-state compact">
              <p>No long-term memories recorded yet. Ask questions or pin a preference above.</p>
            </div>
          ) : (
            <div className="memory-list">
              {longTerm.map((m) => (
                <div key={m.memory_id} className="memory-item-card">
                  <div className="memory-item-head">
                    <div>
                      <span className={`subtle-badge ${m.pinned ? 'pinned' : ''}`}>{m.kind.replace('_', ' ')}</span>
                      <strong>{m.title}</strong>
                    </div>
                    <button
                      type="button"
                      className="icon-button danger"
                      aria-label={`Forget ${m.title}`}
                      onClick={() => void handleDeleteLongTerm(m.memory_id)}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                  <p>{m.summary}</p>
                  <small>
                    Recalled {m.access_count}x · Updated {new Date(m.updated_at).toLocaleString()}
                    {m.source_files?.length ? ` · Files: ${m.source_files.join(', ')}` : ''}
                  </small>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      {/* ── Section 3: Faithful Stateful JWT Sessions ── */}
      <section className="panel memory-section">
        <div className="memory-section-header">
          <div>
            <h2><KeyRound size={18} /> Active Authenticated Sessions</h2>
            <p>Stateful JWT session tracking. Signing out revokes your token immediately on the server.</p>
          </div>
          <button type="button" className="button danger-button small" onClick={() => void handleRevokeAllSessions()}>
            <LogOut size={14} /> Sign out all sessions
          </button>
        </div>
        <div className="sessions-list">
          {sessions.map((s) => (
            <div key={s.jti} className="session-row">
              <div>
                <strong>Session {s.jti.slice(0, 8)}...</strong>
                {s.is_current && <span className="status-pill success">Current</span>}
                {s.revoked && <span className="subtle-badge">Revoked</span>}
                <small>{s.user_agent || 'Web Client'} · Created {new Date(s.created_at).toLocaleString()}</small>
              </div>
              <small>Expires {new Date(s.expires_at).toLocaleDateString()}</small>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
