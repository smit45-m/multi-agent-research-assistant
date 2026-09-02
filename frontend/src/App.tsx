import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { TelemetryStrip } from './components/TelemetryStrip';
import { StudioTab } from './components/StudioTab';
import { BenchmarksTab } from './components/BenchmarksTab';
import { KnowledgeTab } from './components/KnowledgeTab';
import { ConcurrencyTab } from './components/ConcurrencyTab';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('studio');
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [accuracy, setAccuracy] = useState<number | undefined>(undefined);
  const [latency, setLatency] = useState<number | undefined>(undefined);
  const [toastMessage, setToastMessage] = useState('');
  const [showToast, setShowToast] = useState(false);

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
        {/* Anthropic Editorial Hero Section */}
        <section className="anthropic-hero">
          <div className="hero-kicker">Multi-Agent AI Research Assistant \ Winter 2025</div>
          <h1 className="hero-headline">
            Frontier research in retrieval-augmented multi-agent intelligence.
          </h1>
          <p className="hero-subtext">
            Orchestrating 4 autonomous agents across LangGraph and CrewAI for multi-format RAG, decreasing research synthesis time by 60% and achieving 85%+ response accuracy across 200+ benchmarked test cases.
          </p>
        </section>

        <TelemetryStrip accuracy={accuracy} latency={latency} />

        {activeTab === 'studio' && (
          <StudioTab
            onShowToast={triggerToast}
            onUpdateMetrics={updateMetrics}
          />
        )}

        {activeTab === 'benchmarks' && (
          <BenchmarksTab onShowToast={triggerToast} />
        )}

        {activeTab === 'knowledge' && (
          <KnowledgeTab onShowToast={triggerToast} />
        )}

        {activeTab === 'concurrency' && <ConcurrencyTab />}
      </main>

      <div id="toast" className={showToast ? 'show' : ''}>
        {toastMessage}
      </div>
    </>
  );
};

export default App;
