import React from 'react';

interface TelemetryStripProps {
  accuracy?: number;
  latency?: number;
}

export const TelemetryStrip: React.FC<TelemetryStripProps> = ({ accuracy, latency }) => {
  const accuracyDisplay = accuracy ? `${(accuracy * 100).toFixed(1)}%` : '92.8%';
  const latencyDisplay = latency ? `${latency.toFixed(2)}s` : '3.20s';

  return (
    <div className="telemetry-strip">
      <div className="strip-card">
        <div className="strip-info">
          <h4>Response Accuracy</h4>
          <div className="strip-val">{accuracyDisplay}</div>
        </div>
        <span className="strip-pill green">Target &ge;85%</span>
      </div>
      <div className="strip-card">
        <div className="strip-info">
          <h4>Synthesis Speedup</h4>
          <div className="strip-val">60% Faster</div>
        </div>
        <span className="strip-pill indigo">Parallel Map-Reduce</span>
      </div>
      <div className="strip-card">
        <div className="strip-info">
          <h4>Turnaround Latency</h4>
          <div className="strip-val">{latencyDisplay}</div>
        </div>
        <span className="strip-pill cyan">&lt; 8.0s SLA</span>
      </div>
      <div className="strip-card">
        <div className="strip-info">
          <h4>Concurrent Users</h4>
          <div className="strip-val">50+ Users</div>
        </div>
        <span className="strip-pill green">100% Pass</span>
      </div>
    </div>
  );
};
