import React from 'react';
import { Sun, Moon, ExternalLink } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  theme: 'dark' | 'light';
  setTheme: (theme: 'dark' | 'light') => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, theme, setTheme }) => {
  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
  };

  return (
    <nav className="navbar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
        <a href="#" className="nav-brand" onClick={(e) => { e.preventDefault(); setActiveTab('studio'); }}>
          <div className="anthropic-logo" style={{ color: 'var(--brand-primary)', display: 'flex', alignItems: 'center' }}>
            <svg height="20" viewBox="0 0 35 24" fill="currentColor">
              <path d="M24.5475 0H19.3384L28.8374 24H34.0465L24.5475 0Z" fill="currentColor" />
              <path d="M9.49897 0L0 24H5.31125L7.25395 18.96H17.1914L19.1341 24H24.4454L14.9464 0H9.49897ZM8.97193 14.5029L12.2227 6.06857L15.4735 14.5029H8.97193Z" fill="currentColor" />
            </svg>
          </div>
          <div className="brand-meta">
            <h1 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.15rem', fontWeight: 500, letterSpacing: '-0.01em' }}>
              Anthropic <span style={{ color: 'var(--brand-primary)', fontWeight: 300 }}>\ Research</span>
            </h1>
          </div>
        </a>

        <div className="nav-segments">
          <button
            className={`segment-btn ${activeTab === 'studio' ? 'active' : ''}`}
            onClick={() => setActiveTab('studio')}
          >
            Research Studio
          </button>
          <button
            className={`segment-btn ${activeTab === 'benchmarks' ? 'active' : ''}`}
            onClick={() => setActiveTab('benchmarks')}
          >
            200+ Benchmarks
          </button>
          <button
            className={`segment-btn analyst-nav-btn ${activeTab === 'analyst' ? 'active' : ''}`}
            onClick={() => setActiveTab('analyst')}
          >
            <span className="nav-dot" />
            Analyst Lab
          </button>
          <button
            className={`segment-btn ${activeTab === 'knowledge' ? 'active' : ''}`}
            onClick={() => setActiveTab('knowledge')}
          >
            Knowledge Corpus
          </button>
          <button
            className={`segment-btn ${activeTab === 'concurrency' ? 'active' : ''}`}
            onClick={() => setActiveTab('concurrency')}
          >
            Concurrency SLA
          </button>
        </div>
      </div>

      <div className="nav-actions">
        <a
          href="/docs"
          target="_blank"
          rel="noopener noreferrer"
          style={{
            fontSize: '0.78rem',
            color: 'var(--text-secondary)',
            textDecoration: 'none',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.25rem',
            fontWeight: 500,
          }}
        >
          API Docs <ExternalLink size={12} />
        </a>

        <div className="badge-pulse">
          <span>50+ Concurrent SLA</span>
        </div>

        <button
          className="icon-btn"
          onClick={toggleTheme}
          title={`Switch to ${theme === 'dark' ? 'Light (Ivory)' : 'Dark (Slate)'} theme`}
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>
    </nav>
  );
};
