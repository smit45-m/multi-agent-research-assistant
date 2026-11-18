import React from 'react';
import { Sun, Moon, ExternalLink } from 'lucide-react';
import { ScholarLogo } from './ScholarLogo';

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
          <div className="brand-icon-wrapper" style={{ width: 34, height: 34 }}>
            <ScholarLogo size={24} className="brand-icon-svg" />
          </div>
          <div className="brand-meta" style={{ marginLeft: 10 }}>
            <h1 style={{ fontFamily: 'var(--heading)', fontSize: '1.15rem', fontWeight: 700, letterSpacing: '-0.02em', margin: 0 }}>
              Scholar<span className="brand-name-accent">Agent</span> <span style={{ color: 'var(--brand-primary)', fontWeight: 400 }}>\ Research</span>
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
