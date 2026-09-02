import React from 'react';
import { Sparkles, Sun, Moon, Cpu } from 'lucide-react';

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
      <a href="#" className="nav-brand" onClick={(e) => { e.preventDefault(); setActiveTab('studio'); }}>
        <div className="logo-badge">
          <Sparkles size={20} />
        </div>
        <div className="brand-meta">
          <h1>Aether Research</h1>
          <p>Multi-Agent Autonomous Assistant &bull; Jan 2025 – May 2025</p>
        </div>
      </a>

      <div className="nav-segments">
        <button
          className={`segment-btn ${activeTab === 'studio' ? 'active' : ''}`}
          onClick={() => setActiveTab('studio')}
        >
          <Cpu size={14} />
          Studio
        </button>
        <button
          className={`segment-btn ${activeTab === 'benchmarks' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmarks')}
        >
          <span>📊</span>
          200+ Benchmarks
        </button>
        <button
          className={`segment-btn ${activeTab === 'knowledge' ? 'active' : ''}`}
          onClick={() => setActiveTab('knowledge')}
        >
          <span>🗄️</span>
          Knowledge Base
        </button>
        <button
          className={`segment-btn ${activeTab === 'concurrency' ? 'active' : ''}`}
          onClick={() => setActiveTab('concurrency')}
        >
          <span>⚡</span>
          Concurrency & SLA
        </button>
      </div>

      <div className="nav-actions">
        <div className="badge-pulse">
          <span>50+ Concurrent SLA</span>
        </div>
        <button className="icon-btn" onClick={toggleTheme} title="Toggle Dark/Light Mode">
          {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
        </button>
      </div>
    </nav>
  );
};
