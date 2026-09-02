import React from 'react';
import { Handle, Position } from '@xyflow/react';
import { Brain, Search, Scale, PenTool, Database, Compass, CheckCircle2, ShieldCheck, Sparkles, Terminal } from 'lucide-react';

interface AgentNodeData {
  title: string;
  role: string;
  status: 'idle' | 'running' | 'done';
  metric?: string;
  iconType: 'planner' | 'retriever' | 'analyzer' | 'writer' | 'router' | 'evaluator' | 'query' | 'report';
  details?: string;
}

const getIcon = (type: string) => {
  switch (type) {
    case 'planner': return <Brain size={18} />;
    case 'retriever': return <Search size={18} />;
    case 'analyzer': return <Scale size={18} />;
    case 'writer': return <PenTool size={18} />;
    case 'router': return <Compass size={18} />;
    case 'evaluator': return <ShieldCheck size={18} />;
    case 'query': return <Terminal size={18} />;
    case 'report': return <Sparkles size={18} />;
    default: return <Database size={18} />;
  }
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'running': return 'var(--brand-primary)';
    case 'done': return 'var(--accent-emerald)';
    default: return 'var(--text-tertiary)';
  }
};

export const AgentNode: React.FC<{ data: AgentNodeData }> = ({ data }) => {
  const { title, role, status, metric, iconType } = data;
  const isRunning = status === 'running';
  const isDone = status === 'done';

  return (
    <div
      style={{
        background: 'rgba(34, 33, 32, 0.94)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        border: `1.5px solid ${isRunning ? 'var(--brand-primary)' : isDone ? 'rgba(45, 122, 88, 0.55)' : 'rgba(255, 255, 255, 0.1)'}`,
        borderRadius: '12px',
        padding: '1rem 1.25rem',
        minWidth: '240px',
        maxWidth: '280px',
        boxShadow: isRunning
          ? '0 0 25px rgba(204, 120, 92, 0.35), 0 10px 30px rgba(0, 0, 0, 0.5)'
          : isDone
          ? '0 0 20px rgba(45, 122, 88, 0.2), 0 8px 24px rgba(0, 0, 0, 0.4)'
          : '0 8px 24px rgba(0, 0, 0, 0.35)',
        transition: 'all 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
        position: 'relative',
        color: 'var(--text-primary)',
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: getStatusColor(status), width: 10, height: 10, border: '2px solid #141413' }} />

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.45rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem' }}>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              background: isRunning
                ? 'rgba(204, 120, 92, 0.2)'
                : isDone
                ? 'rgba(45, 122, 88, 0.2)'
                : 'rgba(204, 120, 92, 0.12)',
              color: isRunning
                ? 'var(--brand-primary)'
                : isDone
                ? 'var(--accent-emerald)'
                : 'var(--brand-primary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {getIcon(iconType)}
          </div>
          <div>
            <div style={{ fontSize: '0.88rem', fontWeight: 600, letterSpacing: '-0.01em' }}>{title}</div>
          </div>
        </div>

        <span
          style={{
            fontSize: '0.65rem',
            fontWeight: 700,
            textTransform: 'uppercase',
            padding: '0.15rem 0.5rem',
            borderRadius: '999px',
            background: isRunning ? 'rgba(204, 120, 92, 0.2)' : isDone ? 'rgba(45, 122, 88, 0.2)' : 'rgba(255, 255, 255, 0.06)',
            color: isRunning ? 'var(--brand-primary)' : isDone ? 'var(--accent-emerald)' : 'var(--text-tertiary)',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.25rem',
          }}
        >
          {isDone && <CheckCircle2 size={10} />}
          {status}
        </span>
      </div>

      <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: 1.4, marginBottom: '0.45rem' }}>
        {role}
      </div>

      {metric && (
        <div
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.7rem',
            color: isRunning ? 'var(--accent-cyan)' : isDone ? 'var(--accent-emerald)' : 'var(--text-secondary)',
            borderTop: '1px solid rgba(255, 255, 255, 0.07)',
            paddingTop: '0.4rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span>Telemetry:</span>
          <strong>{metric}</strong>
        </div>
      )}

      <Handle type="source" position={Position.Right} style={{ background: getStatusColor(status), width: 10, height: 10, border: '2px solid #090a0f' }} />
    </div>
  );
};
