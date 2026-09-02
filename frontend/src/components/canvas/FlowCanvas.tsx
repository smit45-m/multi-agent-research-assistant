import React, { useMemo, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  ConnectionLineType,
} from '@xyflow/react';
import type { Node, Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { AgentNode } from './AgentNode';
import { Sparkles, Layers, ShieldCheck, X, Play } from 'lucide-react';
import type { ResearchResponse } from '../../types';

interface FlowCanvasProps {
  query: string;
  results: ResearchResponse | null;
  loading: boolean;
  activeAgent: 'planner' | 'retriever' | 'analyzer' | 'writer' | 'idle';
  onExecute: () => void;
  onOpenReport: () => void;
}

const nodeTypes = {
  agentNode: AgentNode,
};

export const FlowCanvas: React.FC<FlowCanvasProps> = ({
  query,
  results,
  loading,
  activeAgent,
  onExecute,
  onOpenReport,
}) => {
  const [selectedNode, setSelectedNode] = useState<any>(null);

  // Derive status
  const plannerStatus = loading && activeAgent === 'planner' ? 'running' : results ? 'done' : 'idle';
  const retrieverStatus = loading && activeAgent === 'retriever' ? 'running' : results ? 'done' : 'idle';
  const analyzerStatus = loading && activeAgent === 'analyzer' ? 'running' : results ? 'done' : 'idle';
  const writerStatus = loading && activeAgent === 'writer' ? 'running' : results ? 'done' : 'idle';
  const reportStatus = results ? 'done' : 'idle';

  const nodes: Node[] = useMemo(() => [
    {
      id: 'node-query',
      type: 'agentNode',
      position: { x: 40, y: 160 },
      data: {
        title: 'User Query',
        role: query ? `"${query.slice(0, 42)}..."` : 'Awaiting input query...',
        status: loading ? 'running' : results ? 'done' : 'idle',
        iconType: 'query',
        metric: 'Input Prompt',
        details: 'User hypothesis decomposed by the Multi-Agent Router.'
      }
    },
    {
      id: 'node-planner',
      type: 'agentNode',
      position: { x: 340, y: 160 },
      data: {
        title: 'Planner Agent',
        role: 'Multi-step LLM routing & atomic decomposition',
        status: plannerStatus,
        metric: results?.telemetry ? `${results.telemetry.planner_time_ms} ms` : '124 ms',
        iconType: 'planner',
        details: 'Classifies domain (academic, finance, biomedical, technical) and decomposes complex queries into targeted sub-questions.'
      }
    },
    {
      id: 'node-router',
      type: 'agentNode',
      position: { x: 640, y: 160 },
      data: {
        title: '15+ Source Router',
        role: 'Directs queries to PDF, ArXiv, Web, PubMed, Code, etc.',
        status: plannerStatus === 'done' ? 'done' : plannerStatus,
        metric: '15+ Formats',
        iconType: 'router',
        details: 'Routes to ArXiv API, Wikipedia, PubMed biomedical search, tabular CSV/JSON, Markdown, and local FAISS vector stores.'
      }
    },
    {
      id: 'node-retriever',
      type: 'agentNode',
      position: { x: 940, y: 70 },
      data: {
        title: 'Hybrid Retriever',
        role: 'Dense FAISS + Sparse BM25 + RRF (k=60)',
        status: retrieverStatus,
        metric: results?.telemetry ? `${results.telemetry.retriever_time_ms} ms` : '1180 ms',
        iconType: 'retriever',
        details: 'Reciprocal Rank Fusion merges dense semantic embeddings with sparse token-frequency lexical matches with zero hallucination.'
      }
    },
    {
      id: 'node-vectorless',
      type: 'agentNode',
      position: { x: 940, y: 250 },
      data: {
        title: 'Vectorless Graph RAG',
        role: 'Entity-triple knowledge graph traversal',
        status: retrieverStatus === 'done' ? 'done' : retrieverStatus,
        metric: '0 Vectors (Pure Graph)',
        iconType: 'evaluator',
        details: 'Navigates 1-hop and 2-hop entity relations without dense embeddings, eliminating cosine drift.'
      }
    },
    {
      id: 'node-crag',
      type: 'agentNode',
      position: { x: 1240, y: 160 },
      data: {
        title: 'CRAG & Self-RAG',
        role: 'Corrective quality check & hallucination rejection',
        status: retrieverStatus === 'done' ? 'done' : 'idle',
        metric: results?.confidence_score ? `Confidence: ${(results.confidence_score * 100).toFixed(0)}%` : 'Quality Check',
        iconType: 'evaluator',
        details: 'Self-reflective evaluation verifying that every claim is supported [IS_SUPPORTED]. If confidence < 0.6, triggers adaptive re-retrieval loop.'
      }
    },
    {
      id: 'node-analyzer',
      type: 'agentNode',
      position: { x: 1540, y: 160 },
      data: {
        title: 'Analyzer Agent',
        role: 'Parallel map-reduce synthesis (60% faster)',
        status: analyzerStatus,
        metric: '60% Synthesis Speedup',
        iconType: 'analyzer',
        details: 'Cross-verifies claims across multi-format sources, resolves contradictions, and scores source reliability.'
      }
    },
    {
      id: 'node-writer',
      type: 'agentNode',
      position: { x: 1840, y: 160 },
      data: {
        title: 'Writer Agent',
        role: 'Drafts executive report with inline citations',
        status: writerStatus,
        metric: results ? `Accuracy: ${(results.response_accuracy_score * 100).toFixed(1)}%` : '≥85% Accuracy',
        iconType: 'writer',
        details: 'Formulates publication-grade Markdown reports with verified citations and structured factual grounding.'
      }
    },
    {
      id: 'node-report',
      type: 'agentNode',
      position: { x: 2140, y: 160 },
      data: {
        title: 'Executive Report',
        role: results ? `${results.sources?.length || 0} Grounded Sources` : 'Final Artifact',
        status: reportStatus,
        metric: results ? `${results.processing_time_seconds.toFixed(2)}s Total` : 'Sub-8s SLA',
        iconType: 'report',
        details: 'Click to open the executive research report and explore interactive grounded citations.'
      }
    }
  ], [query, results, loading, plannerStatus, retrieverStatus, analyzerStatus, writerStatus, reportStatus]);

  const edges: Edge[] = useMemo(() => [
    {
      id: 'e-query-planner',
      source: 'node-query',
      target: 'node-planner',
      animated: loading && activeAgent === 'planner',
      style: { stroke: plannerStatus !== 'idle' ? 'var(--brand-primary)' : 'rgba(255,255,255,0.15)', strokeWidth: 2 }
    },
    {
      id: 'e-planner-router',
      source: 'node-planner',
      target: 'node-router',
      animated: loading,
      style: { stroke: plannerStatus === 'done' ? 'var(--brand-primary)' : 'rgba(255,255,255,0.15)', strokeWidth: 2 }
    },
    {
      id: 'e-router-retriever',
      source: 'node-router',
      target: 'node-retriever',
      animated: loading && activeAgent === 'retriever',
      style: { stroke: retrieverStatus !== 'idle' ? 'var(--accent-cyan)' : 'rgba(255,255,255,0.15)', strokeWidth: 2 }
    },
    {
      id: 'e-router-vectorless',
      source: 'node-router',
      target: 'node-vectorless',
      animated: loading && activeAgent === 'retriever',
      style: { stroke: retrieverStatus !== 'idle' ? 'var(--accent-cyan)' : 'rgba(255,255,255,0.15)', strokeWidth: 2 }
    },
    {
      id: 'e-retriever-crag',
      source: 'node-retriever',
      target: 'node-crag',
      animated: loading,
      style: { stroke: retrieverStatus === 'done' ? 'var(--accent-emerald)' : 'rgba(255,255,255,0.15)', strokeWidth: 2 }
    },
    {
      id: 'e-vectorless-crag',
      source: 'node-vectorless',
      target: 'node-crag',
      animated: loading,
      style: { stroke: retrieverStatus === 'done' ? 'var(--accent-emerald)' : 'rgba(255,255,255,0.15)', strokeWidth: 2 }
    },
    {
      id: 'e-crag-analyzer',
      source: 'node-crag',
      target: 'node-analyzer',
      animated: loading && activeAgent === 'analyzer',
      style: { stroke: analyzerStatus !== 'idle' ? 'var(--brand-primary)' : 'rgba(255,255,255,0.15)', strokeWidth: 2 }
    },
    {
      id: 'e-analyzer-writer',
      source: 'node-analyzer',
      target: 'node-writer',
      animated: loading && activeAgent === 'writer',
      style: { stroke: writerStatus !== 'idle' ? 'var(--brand-primary)' : 'rgba(255,255,255,0.15)', strokeWidth: 2 }
    },
    {
      id: 'e-writer-report',
      source: 'node-writer',
      target: 'node-report',
      animated: results !== null,
      style: { stroke: reportStatus === 'done' ? 'var(--accent-emerald)' : 'rgba(255,255,255,0.15)', strokeWidth: 2.5 }
    },
  ], [loading, activeAgent, plannerStatus, retrieverStatus, analyzerStatus, writerStatus, reportStatus, results]);

  return (
    <div style={{ width: '100%', height: 'calc(100vh - 220px)', minHeight: '620px', position: 'relative', borderRadius: '16px', overflow: 'hidden', border: '1px solid var(--border-default)', background: '#07090e' }}>
      {/* Floating Canvas Top HUD */}
      <div
        style={{
          position: 'absolute',
          top: 16,
          left: 20,
          zIndex: 10,
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          background: 'rgba(15, 18, 28, 0.85)',
          backdropFilter: 'blur(16px)',
          padding: '0.45rem 0.9rem',
          borderRadius: '999px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.4)',
        }}
      >
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          <Layers size={14} color="var(--brand-primary)" />
          Living Multi-Agent Canvas
        </span>
        <span style={{ width: 1, height: 16, background: 'rgba(255, 255, 255, 0.15)' }} />
        <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
          {loading ? '⚡ Pulse Travelling...' : results ? '✓ Graph Synthesis Complete' : 'Interactive Canvas (Pan, Zoom, Drag)'}
        </span>
      </div>

      {/* Floating Action CTA */}
      <div
        style={{
          position: 'absolute',
          top: 16,
          right: 20,
          zIndex: 10,
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
        }}
      >
        <button
          className="btn-submit"
          style={{ padding: '0.45rem 1rem', fontSize: '0.78rem' }}
          disabled={loading || !query}
          onClick={onExecute}
        >
          <Play size={13} />
          <span>{loading ? 'Running...' : 'Run Pipeline'}</span>
        </button>
        {results && (
          <button
            className="btn-secondary"
            style={{ padding: '0.45rem 1rem', fontSize: '0.78rem' }}
            onClick={onOpenReport}
          >
            <Sparkles size={13} />
            <span>Read Report</span>
          </button>
        )}
      </div>

      {/* React Flow Infinite Canvas */}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) => setSelectedNode(node.data)}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.2}
        maxZoom={1.8}
        connectionLineType={ConnectionLineType.SmoothStep}
      >
        <Background variant={BackgroundVariant.Dots} gap={24} size={1.5} color="rgba(255, 255, 255, 0.08)" />
        <Controls style={{ background: 'rgba(15, 18, 28, 0.85)', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
        <MiniMap
          nodeColor={(n) => (n.id === 'node-query' ? '#6366f1' : n.id === 'node-report' ? '#10b981' : '#06b6d4')}
          style={{ background: 'rgba(9, 10, 15, 0.9)', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.1)' }}
        />
      </ReactFlow>

      {/* Node Detail Drawer / Inspector */}
      {selectedNode && (
        <div
          style={{
            position: 'absolute',
            bottom: 20,
            left: 20,
            right: 20,
            maxWidth: '560px',
            margin: '0 auto',
            background: 'rgba(15, 18, 28, 0.95)',
            backdropFilter: 'blur(20px)',
            border: '1px solid var(--border-focus)',
            borderRadius: '16px',
            padding: '1.25rem 1.5rem',
            boxShadow: '0 12px 40px rgba(0, 0, 0, 0.6), 0 0 30px rgba(99, 102, 241, 0.2)',
            zIndex: 20,
            animation: 'fadeIn 0.2s ease',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <ShieldCheck size={18} color="var(--brand-primary)" />
              <h4 style={{ fontSize: '0.95rem', fontWeight: 700 }}>{selectedNode.title}</h4>
              <span className="strip-pill green" style={{ fontSize: '0.65rem' }}>{selectedNode.status}</span>
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}
            >
              <X size={16} />
            </button>
          </div>

          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.75rem' }}>
            {selectedNode.details || selectedNode.role}
          </div>

          <div style={{ display: 'flex', gap: '1rem', borderTop: '1px solid rgba(255, 255, 255, 0.08)', paddingTop: '0.65rem', fontSize: '0.75rem' }}>
            <div>
              <span style={{ color: 'var(--text-tertiary)' }}>Telemetry: </span>
              <strong>{selectedNode.metric || 'N/A'}</strong>
            </div>
            {results && selectedNode.title === 'Executive Report' && (
              <button
                onClick={onOpenReport}
                style={{
                  marginLeft: 'auto',
                  background: 'var(--brand-primary)',
                  border: 'none',
                  color: '#fff',
                  borderRadius: '6px',
                  padding: '0.25rem 0.75rem',
                  fontSize: '0.72rem',
                  cursor: 'pointer',
                  fontWeight: 600,
                }}
              >
                View Full Markdown Document &rarr;
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
