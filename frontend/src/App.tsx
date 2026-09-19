import React, { useEffect, useState, Suspense } from 'react';
import { Navbar } from './components/Navbar';
import { TelemetryStrip } from './components/TelemetryStrip';
import { StudioTab } from './components/StudioTab';
import { BenchmarksTab } from './components/BenchmarksTab';
import { KnowledgeTab } from './components/KnowledgeTab';
import { ConcurrencyTab } from './components/ConcurrencyTab';
import { AnalystTab } from './components/AnalystTab';

const HeroScene = React.lazy(() =>
  import('./components/three/HeroScene').then((m) => ({ default: m.HeroScene })),
);

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('studio');
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [accuracy, setAccuracy] = useState<number | undefined>(undefined);
  const [latency, setLatency] = useState<number | undefined>(undefined);
  const [toastMessage, setToastMessage] = useState('');
  const [showToast, setShowToast] = useState(false);
  const [studioSeedQuery, setStudioSeedQuery] = useState<string | undefined>(undefined);

  useEffect(() => {
    const saved = localStorage.getItem('theme');
    if (saved === 'light' || saved === 'dark') {
      setTheme(saved);
      document.documentElement.setAttribute('data-theme', saved);
    }
  }, []);

  const triggerToast = (msg: string) => {
    setToastMessage(msg);
    setShowToast(true);
    setTimeout(() => setShowToast(false), 2800);
  };

  const updateMetrics = (acc: number, lat: number) => {
    setAccuracy(acc);
    setLatency(lat);
  };

  return (
    <>
      <div className="ambient-glow" />

      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        theme={theme}
        setTheme={setTheme}
      />

      <main className="container">
        {/* Editorial hero with ambient 3D scene */}
        <section className="anthropic-hero hero-with-3d">
          <Suspense fallback={null}>
            <HeroScene />
          </Suspense>
          <div className="hero-copy">
            <div className="hero-kicker">Multi-Agent AI Research Assistant \ 2026</div>
            <h1 className="hero-headline">
              Frontier research in retrieval-augmented multi-agent intelligence.
            </h1>
            <p className="hero-subtext">
              Seven specialized agents — Planner, Retriever, Analyzer, Writer,
              Verifier, Critic, and the multimodal Interactive Analyst — with
              every accuracy and latency figure measured from real pipeline
              runs, never asserted.
            </p>
            <div className="hero-stats">
              <div className="hero-stat">
                <span className="hero-stat-value">205/205</span>
                <span className="hero-stat-label">benchmark cases passed</span>
              </div>
              <div className="hero-stat-divider" />
              <div className="hero-stat">
                <span className="hero-stat-value">p95 &lt; 8s</span>
                <span className="hero-stat-label">at 50 concurrent users</span>
              </div>
              <div className="hero-stat-divider" />
              <div className="hero-stat">
                <span className="hero-stat-value">7 agents</span>
                <span className="hero-stat-label">incl. multimodal analyst</span>
              </div>
            </div>
          </div>
        </section>

        <TelemetryStrip accuracy={accuracy} latency={latency} />

        <div className="tab-pane" key={activeTab}>
          {activeTab === 'studio' && (
            <StudioTab
              onShowToast={triggerToast}
              onUpdateMetrics={updateMetrics}
              initialQuery={studioSeedQuery}
            />
          )}

          {activeTab === 'analyst' && (
            <AnalystTab
              onShowToast={triggerToast}
              onAskResearch={(q) => {
                setStudioSeedQuery(q);
                setActiveTab('studio');
              }}
            />
          )}

          {activeTab === 'benchmarks' && (
            <BenchmarksTab onShowToast={triggerToast} />
          )}

          {activeTab === 'knowledge' && (
            <KnowledgeTab onShowToast={triggerToast} />
          )}

          {activeTab === 'concurrency' && <ConcurrencyTab />}
        </div>
      </main>

      <div id="toast" className={showToast ? 'show' : ''}>
        {toastMessage}
      </div>
    </>
  );
};

export default App;
