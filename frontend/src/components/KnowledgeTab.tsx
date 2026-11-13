import { Fragment, useCallback, useEffect, useRef, useState } from 'react';
import { BookOpen, CheckCircle2, ChevronDown, ChevronUp, Database, FileText, LoaderCircle, MessageSquare, RefreshCw, Search, ShieldCheck, Trash2, UploadCloud } from 'lucide-react';
import { api, errorMessage } from '../api';
import type { DocumentResponse, DocumentListResponse, User } from '../types';

const formats = ['pdf', 'docx', 'txt', 'csv', 'json', 'html', 'htm', 'md', 'xlsx', 'tsv', 'py', 'js', 'ts', 'sh', 'yaml', 'yml', 'xml'];
interface Props {
  onShowToast: (message: string) => void;
  currentUser?: User | null;
  onOpenChat?: (chatId: string) => void;
  onOpenAuth?: () => void;
  onWorkspaceChange?: () => void;
}
export function KnowledgeTab({ onShowToast, currentUser, onOpenChat, onOpenAuth, onWorkspaceChange }: Props) {
  const [docs, setDocs] = useState<DocumentResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState('');
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [search, setSearch] = useState('');
  const [expandedDocId, setExpandedDocId] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<DocumentResponse | null>(null);
  const [deleting, setDeleting] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const uploadLock = useRef(false);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    setLoading(true); setError('');
    try {
      const data = await api<DocumentListResponse>('/api/v1/documents/', signal ? { signal } : {});
      setDocs(data.documents || []);
    } catch (err) {
      if (!signal?.aborted) setError(errorMessage(err));
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    return () => controller.abort();
  }, [refresh, currentUser?.id]);

  useEffect(() => { if (pendingDelete) dialog.current?.showModal(); else dialog.current?.close(); }, [pendingDelete]);

  async function uploadFiles(files: File[]) {
    if (uploadLock.current || !files.length) return;
    uploadLock.current = true; setError('');
    const failures: string[] = []; let count = 0;
    for (const [index, file] of files.entries()) {
      const extension = file.name.split('.').pop()?.toLowerCase() || '';
      if (!formats.includes(extension)) { failures.push(`${file.name}: unsupported format.`); continue; }
      if (!file.size || file.size > 20 * 1024 * 1024) { failures.push(`${file.name}: choose a non-empty file under 20 MB.`); continue; }
      setUploading(`${index + 1} of ${files.length} · ${file.name}`);
      const form = new FormData(); form.append('file', file);
      try { await api<DocumentResponse>('/api/v1/documents/upload', { method: 'POST', body: form }); count++; }
      catch (err) { failures.push(`${file.name}: ${errorMessage(err)}`); }
    }
    setUploading(''); uploadLock.current = false;
    if (input.current) input.current.value = '';
    if (count) {
      onShowToast(`${count} ${count === 1 ? 'document' : 'documents'} processed.`);
      await refresh();
      onWorkspaceChange?.();
    }
    if (failures.length) setError(failures.join('\n'));
  }

  async function removeDocument() {
    if (!pendingDelete || deleting) return;
    setDeleting(true);
    try {
      await api(`/api/v1/documents/${encodeURIComponent(pendingDelete.document_id)}`, { method: 'DELETE' });
      setDocs(items => items.filter(item => item.document_id !== pendingDelete.document_id));
      if (expandedDocId === pendingDelete.document_id) setExpandedDocId(null);
      setPendingDelete(null);
      onShowToast(currentUser ? 'Document permanently removed from your private database and index.' : 'Document removed from the guest library.');
      onWorkspaceChange?.();
    } catch (err) { setPendingDelete(null); setError(errorMessage(err)); }
    finally { setDeleting(false); }
  }

  const filtered = docs.filter(doc => doc.filename.toLowerCase().includes(search.toLowerCase()));

  return <div className="secondary-page"><div className="page-heading"><div className="eyebrow">YOUR RESEARCH FOUNDATION</div><h1>{currentUser ? `${currentUser.full_name.split(' ')[0]}'s private library.` : 'A little context goes a long way.'}</h1><p>{currentUser ? 'Every uploaded file and the Q&As grounded in it are stored in your isolated personal database.' : 'Bring your documents together. Sign in to bind files & Q&A links strictly to your private database.'}</p></div>
    <section className="panel library-panel">
      <div className={`workspace-isolation-banner ${currentUser ? 'private' : 'guest'}`}>
        <div className="workspace-isolation-meta">
          {currentUser ? <Database size={16} /> : <ShieldCheck size={16} />}
          <div>
            <strong>{currentUser ? 'Private Per-User MongoDB Database Active' : 'Guest Sandbox Mode'}</strong>
            <span>{currentUser ? `Isolated DB: ${currentUser.db_name || currentUser.id} · Files & vector index hidden from all other users.` : 'Sign in to isolate uploaded files per account and keep persistent Q&A provenance across logins.'}</span>
          </div>
        </div>
        {!currentUser && onOpenAuth && (
          <button type="button" className="button small" onClick={onOpenAuth}>Sign in for private DB</button>
        )}
      </div>

      <button className={`upload-dropzone ${dragOver ? 'dragover' : ''}`} disabled={!!uploading} onClick={() => input.current?.click()} onDragOver={event => { event.preventDefault(); if (!uploading) setDragOver(true); }} onDragLeave={() => setDragOver(false)} onDrop={event => { event.preventDefault(); setDragOver(false); void uploadFiles(Array.from(event.dataTransfer.files)); }}>{uploading ? <LoaderCircle size={30} className="spin" /> : <UploadCloud size={30} />}<strong>{uploading ? 'Processing your documents' : 'Drop your documents here'}</strong><span>{uploading || 'or click to browse files'}</span><small>PDF, Word, text, spreadsheets, data & code · Up to 20 MB per file</small></button><input ref={input} type="file" hidden multiple accept={formats.map(format => `.${format}`).join(',')} aria-label="Choose documents to upload" onChange={event => void uploadFiles(Array.from(event.target.files || []))} />
      {error && <div className="alert error" role="alert"><div><strong>Something needs your attention</strong><p className="preserve-lines">{error}</p></div><button className="button small" onClick={() => void refresh()}>Retry connection</button></div>}
      <div className="library-toolbar"><h2>Your documents <span className="count">{docs.length}</span></h2><div className="toolbar-actions"><label className="search-field"><Search size={16} /><input aria-label="Search documents" placeholder="Find a document..." value={search} onChange={event => setSearch(event.target.value)} /></label><button className="icon-button" aria-label="Refresh document list" onClick={() => void refresh()} disabled={loading}><RefreshCw size={16} className={loading ? 'spin' : ''} /></button></div></div>
      <div className="table-scroll"><table className="data-table"><thead><tr><th scope="col">Document</th><th scope="col">Format</th><th scope="col">Passages</th><th scope="col">Linked Q&amp;As</th><th scope="col">Status</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead><tbody>{loading ? <tr><td colSpan={6}><div className="empty-state"><LoaderCircle size={25} className="spin" /><p>Loading your library...</p></div></td></tr> : filtered.length ? filtered.map(doc => {
        const links = doc.qa_links || [];
        const qaCount = doc.qa_count ?? links.length;
        const isExpanded = expandedDocId === doc.document_id;
        return <Fragment key={doc.document_id}>
          <tr>
            <td><div className="document-name"><span className="document-icon"><FileText size={19} /></span><div><strong>{doc.filename}</strong><small>{new Date(doc.uploaded_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}</small></div></div></td>
            <td><span className="subtle-badge">{doc.format.toUpperCase()}</span></td>
            <td>{doc.chunk_count}</td>
            <td>
              {links.length > 0 ? (
                <button type="button" className="qa-link-toggle" onClick={() => setExpandedDocId(isExpanded ? null : doc.document_id)}>
                  <MessageSquare size={13} />
                  <span>{qaCount} {qaCount === 1 ? 'Q&A' : 'Q&As'}</span>
                  {isExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                </button>
              ) : (
                <span className="muted-cell">0 Q&amp;As</span>
              )}
            </td>
            <td><span className="status-pill success"><CheckCircle2 size={12} />{doc.status}</span></td>
            <td><button className="icon-button danger" aria-label={`Remove ${doc.filename} from library list`} onClick={() => setPendingDelete(doc)}><Trash2 size={16} /></button></td>
          </tr>
          {isExpanded && links.length > 0 && (
            <tr className="qa-links-row">
              <td colSpan={6}>
                <div className="qa-links-panel">
                  <div className="qa-links-heading">Questions asked &amp; answers grounded in {doc.filename}</div>
                  {links.map((link) => (
                    <div key={link.chat_id} className="qa-link-card">
                      <div className="qa-link-body">
                        <strong>Q: {link.query}</strong>
                        {link.answer_summary && <p>A: {link.answer_summary}</p>}
                        <small>{new Date(link.created_at).toLocaleString()}</small>
                      </div>
                      {onOpenChat && (
                        <button type="button" className="button small" onClick={() => onOpenChat(link.chat_id)}>
                          Open answer
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </td>
            </tr>
          )}
        </Fragment>;
      }) : <tr><td colSpan={6}><div className="empty-state"><BookOpen size={30} /><h3>{search ? 'No matching documents' : error ? 'Library unavailable' : 'Make this library yours'}</h3><p>{search ? 'Try another filename or clear your search.' : error ? 'Connect the backend and refresh to see your documents.' : 'Add your first document to give your research more context.'}</p>{search && <button className="text-button" onClick={() => setSearch('')}>Clear search</button>}</div></td></tr>}</tbody></table></div>
      <p className="panel-footnote">{currentUser ? `Document metadata and Q&A history are persisted inside your private MongoDB database (${currentUser.db_name || currentUser.id}), while vector embeddings are isolated in your FAISS workspace.` : 'Guest documents are shared only in the temporary sandbox session. Sign in for strict per-user database isolation.'}</p></section>
    <dialog ref={dialog} className="confirm-dialog" onCancel={event => { event.preventDefault(); if (!deleting) setPendingDelete(null); }}><h2>Remove this document?</h2><p><strong>{pendingDelete?.filename}</strong></p><p>{currentUser ? 'This permanently deletes the document metadata from your private MongoDB database and removes its vector chunks from your isolated FAISS index.' : 'This removes the document from the guest library.'}</p><div className="dialog-actions"><button className="button" disabled={deleting} onClick={() => setPendingDelete(null)}>Keep document</button><button className="button danger-button" disabled={deleting} onClick={() => void removeDocument()}>{deleting ? 'Removing...' : 'Remove document'}</button></div></dialog>
  </div>;
}
