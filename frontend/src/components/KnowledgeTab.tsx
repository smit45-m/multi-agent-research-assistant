import React, { useState, useEffect } from 'react';
import { UploadCloud, Trash2, FileText, CheckCircle2 } from 'lucide-react';
import type { DocumentResponse, DocumentListResponse } from '../types';

interface KnowledgeTabProps {
  onShowToast: (msg: string) => void;
}

export const KnowledgeTab: React.FC<KnowledgeTabProps> = ({ onShowToast }) => {
  const [docs, setDocs] = useState<DocumentResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const fetchDocuments = async () => {
    try {
      const res = await fetch('/api/v1/documents/');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: DocumentListResponse = await res.json();
      setDocs(data.documents || []);
    } catch (err: any) {
      onShowToast('Failed to load documents: ' + err.message);
    }
  };

  const uploadFile = async (file: File) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    onShowToast(`Uploading & indexing ${file.name}...`);
    try {
      const res = await fetch('/api/v1/documents/upload', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error('Upload failed');
      onShowToast(`Successfully indexed ${file.name} across hybrid RAG!`);
      fetchDocuments();
    } catch (err: any) {
      onShowToast('Upload error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const deleteDocument = async (id: string) => {
    try {
      await fetch(`/api/v1/documents/${id}`, { method: 'DELETE' });
      onShowToast('Document removed from vector index.');
      fetchDocuments();
    } catch (err: any) {
      onShowToast('Deletion error');
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      uploadFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="table-card" style={{ padding: '1.6rem' }}>
      <div style={{ marginBottom: '1.25rem' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>
          15+ Multi-Format Document Ingestion & Chunking
        </h3>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginTop: '0.2rem' }}>
          Direct local indexing into Hybrid FAISS vector store and BM25 sparse index.
        </p>
      </div>

      {/* Upload Box */}
      <div
        className={`upload-dropzone ${dragOver ? 'dragover' : ''}`}
        onDragEnter={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById('doc-file-input')?.click()}
      >
        <UploadCloud size={34} style={{ color: 'var(--brand-primary)', marginBottom: '0.5rem' }} />
        <div style={{ fontSize: '0.88rem', fontWeight: 600 }}>
          {loading ? 'Uploading & chunking document...' : 'Drag and drop files here, or click to browse'}
        </div>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.3rem' }}>
          Supported: .pdf, .docx, .csv, .json, .html, .md, .xlsx, .tsv, .py, .yaml, .xml
        </div>
        <input
          type="file"
          id="doc-file-input"
          style={{ display: 'none' }}
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              uploadFile(e.target.files[0]);
            }
          }}
        />
      </div>

      {/* Document Library Table */}
      <div style={{ marginTop: '1.75rem' }}>
        <h4 style={{ fontSize: '0.88rem', fontWeight: 700, marginBottom: '0.85rem' }}>
          Indexed Document Store ({docs.length})
        </h4>
        <table className="clean-table" style={{ border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
          <thead>
            <tr>
              <th>Filename</th>
              <th>Format</th>
              <th>Chunks Indexed</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {docs.length > 0 ? (
              docs.map((d) => (
                <tr key={d.document_id}>
                  <td style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <FileText size={14} color="var(--brand-primary)" />
                    <span>{d.filename}</span>
                  </td>
                  <td>
                    <span className="source-tag">{d.format}</span>
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                    {d.chunk_count}
                  </td>
                  <td>
                    <span className="strip-pill green" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
                      <CheckCircle2 size={11} />
                      {d.status}
                    </span>
                  </td>
                  <td>
                    <button
                      onClick={() => deleteDocument(d.document_id)}
                      style={{ background: 'transparent', border: 'none', color: 'var(--accent-rose)', cursor: 'pointer' }}
                      title="Delete document"
                    >
                      <Trash2 size={14} />
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={5} style={{ textAlign: 'center', color: 'var(--text-tertiary)', padding: '1.75rem' }}>
                  No documents uploaded yet. Upload files to add to the knowledge base.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
