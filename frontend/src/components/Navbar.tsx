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
          <div className="brand-icon-wrapper" style={{ width: 32, height: 32 }}>
            <svg className="brand-icon-svg" style={{ width: 20, height: 20 }} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="12" cy="12" r="3.2" className="brand-icon-core" />
              <path
                d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"
                className="brand-icon-teeth"
              />
            </svg>
          </div>
          <div className="brand-meta" style={{ marginLeft: 10 }}>
            <h1 style={{ fontFamily: 'var(--heading)', fontSize: '1.15rem', fontWeight: 700, letterSpacing: '-0.02em', margin: 0 }}>
              ScholarAgent <span style={{ color: 'var(--brand-primary)', fontWeight: 400 }}>\ Research</span>
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
