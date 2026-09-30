import { useRef, useState } from 'react';
import { ArrowRight, CheckCircle2, FlaskConical, LoaderCircle, Search, XCircle } from 'lucide-react';
import { api, errorMessage } from '../api';
import type { BenchmarkResponse } from '../types';

export function BenchmarksTab({ onShowToast }: { onShowToast: (message: string) => void }) {
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('all');
  const [limit, setLimit] = useState(30);
  const lock = useRef(false);
  async function run() {
    if (lock.current) return;
    lock.current = true; setLoading(true); setError('');
    try {
      const result = await api<BenchmarkResponse>('/api/v1/research/benchmark/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ max_cases: limit }) });
      setData(result); setCategory('all'); onShowToast(`Evaluation finished: ${result.passed_test_cases} of ${result.total_test_cases} cases passed.`);
    } catch (err) { setError(errorMessage(err)); }
    finally { setLoading(false); lock.current = false; }
  }
  const results = (data?.sample_results || []).filter(item => (category === 'all' || item.category === category) && `${item.query} ${item.test_id}`.toLowerCase().includes(search.toLowerCase()));
  const categories = [...new Set((data?.sample_results || []).map(item => item.category))];
  return <div className="secondary-page"><div className="page-heading"><div className="eyebrow">UNDERSTAND THE OUTPUT</div><h1>Put the research to the test.</h1><p>Explore the project's evaluation cases and see where the system can improve.</p></div><div className="alert"><FlaskConical size={19} /><div><strong>Development evaluation, not a production guarantee.</strong><p>The current backend uses estimated accuracy and simulated latency. These results are useful for development, not independent evidence of real-world performance.</p></div></div><section className="panel"><div className="panel-toolbar"><div><h2>Benchmark evaluation</h2><p>Run only when you need it. No evaluations start automatically.</p></div><div className="toolbar-actions"><label className="sr-only" htmlFor="case-limit">Number of test cases</label><select id="case-limit" value={limit} disabled={loading} onChange={event => setLimit(Number(event.target.value))}><option value={5}>5 cases</option><option value={30}>30 cases</option><option value={100}>100 cases</option><option value={300}>Full suite (up to 300)</option></select><button className="button primary" disabled={loading} onClick={() => void run()}>{loading ? <><LoaderCircle size={16} className="spin" /> Evaluating</> : <>Run evaluation <ArrowRight size={16} /></>}</button></div></div>
    {error && <div className="alert error inset" role="alert"><strong>Evaluation failed</strong><p>{error}</p></div>}
    {data && <div className="metrics-grid"><div><span>Cases evaluated</span><strong>{data.total_test_cases}</strong></div><div><span>Pass rate</span><strong>{data.pass_rate_percentage.toFixed(1)}<small>%</small></strong></div><div><span>Estimated accuracy</span><strong>{data.average_accuracy_percentage.toFixed(1)}<small>%</small></strong></div><div><span>Simulated latency</span><strong>{data.average_latency_seconds.toFixed(2)}<small>s</small></strong></div></div>}
    {data && <div className="filter-toolbar"><label className="search-field"><Search size={16} /><input aria-label="Search benchmark cases" placeholder="Search questions or test IDs..." value={search} onChange={event => setSearch(event.target.value)} /></label><label className="sr-only" htmlFor="category">Filter benchmark category</label><select id="category" value={category} onChange={event => setCategory(event.target.value)}><option value="all">All categories</option>{categories.map(item => <option key={item}>{item}</option>)}</select></div>}
    <div className="table-scroll"><table className="data-table"><thead><tr><th scope="col">Test case</th><th scope="col">Research question</th><th scope="col">Estimated accuracy</th><th scope="col">Simulated latency</th><th scope="col">Result</th></tr></thead><tbody>{results.length ? results.map(item => <tr key={item.test_id}><td><code>{item.test_id}</code></td><td className="query-cell"><strong>{item.query}</strong><small>{item.category}</small></td><td>{item.accuracy_score.toFixed(1)}%</td><td>{item.latency_seconds.toFixed(2)}s</td><td><span className={`status-pill ${item.passed ? 'success' : 'failure'}`}>{item.passed ? <CheckCircle2 size={12} /> : <XCircle size={12} />}{item.passed ? 'Passed' : 'Failed'}</span></td></tr>) : <tr><td colSpan={5}><div className="empty-state">{loading ? <LoaderCircle size={30} className="spin" /> : <FlaskConical size={30} />}<h3>{loading ? 'Evaluating your selected cases' : data ? 'No matching test cases' : 'A clearer picture starts with a test'}</h3><p>{loading ? 'Results will appear when the server finishes.' : data ? 'Try a different search or category.' : 'Choose a sample size, then run an evaluation.'}</p></div></td></tr>}</tbody></table></div>{data && <p className="panel-footnote">Showing {results.length} of {data.sample_results.length} returned sample cases. Aggregate metrics cover {data.total_test_cases} evaluated cases.</p>}</section></div>;
}
