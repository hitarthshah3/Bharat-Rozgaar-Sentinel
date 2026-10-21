import React, { useEffect, useState } from 'react';
import brandLogo from './assets/bharat-rozgaar-sentinel-mark.jpg';

const API = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '');
const EMPTY = { role_and_location: '', company_and_recruiter: '', work_and_qualifications: '', pay_and_application: '' };
const MODELS = { baseline: 'TF-IDF + Logistic Regression', bilstm: 'Word2Vec + BiLSTM', hybrid_bilstm: 'Hybrid BiLSTM', distilbert: 'DistilBERT', ensemble: 'Three-model ensemble' };
const METRIC_MODEL_NAMES = { tfidf_lr: 'TF-IDF + Logistic Regression', word2vec_bilstm: 'Word2Vec + BiLSTM', distilbert: 'DistilBERT', ensemble: 'Three-model ensemble', hybrid_bilstm: 'Hybrid BiLSTM', hybrid_bilstm_0_5: 'Hybrid BiLSTM (0.5 threshold)', distilbert_0_5: 'DistilBERT (0.5 threshold)', word2vec_bilstm_ablation: 'BiLSTM text-only ablation' };
const SHORT_REASONS = { 'upfront payment request': 'Upfront fees', 'unusually high pay claim': 'High pay', 'informal contact channel': 'Informal contact', 'urgent hiring language': 'Urgent hiring', 'sensitive information request': 'Sensitive data', 'company profile not provided': 'No profile', 'company logo not provided': 'No logo', 'no listed warning pattern matched': 'No rule match' };

function PageLink({ page, active, children, onNavigate }) {
  return <button className={active ? 'active' : ''} type="button" onClick={() => onNavigate(page)}>{children}</button>;
}

export default function App() {
  const initialPage = ['home', 'analyze', 'results'].includes(window.location.hash.slice(1)) ? window.location.hash.slice(1) : 'home';
  const [page, setPage] = useState(initialPage);
  const [posting, setPosting] = useState(EMPTY);
  const [api, setApi] = useState('checking');
  const [model, setModel] = useState('baseline');
  const [file, setFile] = useState(null);
  const [text, setText] = useState('');
  const [extract, setExtract] = useState(null);
  const [uploadError, setUploadError] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [showMetrics, setShowMetrics] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState('');
  const [busy, setBusy] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [drag, setDrag] = useState(false);

  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
        const h = await r.json();
        if (alive) { setApi(r.ok && h.status === 'ready' ? 'online' : 'offline'); setModel(h.model || 'baseline'); }
      } catch { if (alive) setApi('offline'); }
    };
    check(); const id = setInterval(check, 30000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  useEffect(() => {
    const sync = () => { const next = window.location.hash.slice(1); if (['home', 'analyze', 'results'].includes(next)) setPage(next); };
    window.addEventListener('hashchange', sync);
    return () => window.removeEventListener('hashchange', sync);
  }, []);

  function navigate(next) {
    setPage(next);
    if (window.location.hash !== `#${next}`) window.history.pushState(null, '', `#${next}`);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  async function toggleMetrics() {
    if (showMetrics) { setShowMetrics(false); return; }
    setShowMetrics(true);
    if (metrics !== null || metricsLoading) return;
    setMetricsLoading(true); setMetricsError('');
    try {
      const response = await fetch(`${API}/metrics`);
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Evaluation metrics could not be loaded.');
      setMetrics(body.models || []);
    } catch (e) { setMetricsError(e.message || 'Evaluation metrics could not be loaded.'); }
    finally { setMetricsLoading(false); }
  }

  function chooseFile(next) {
    setUploadError(''); setExtract(null); setText('');
    if (!next) { setFile(null); return; }
    if (!/\.(pdf|png|jpe?g)$/i.test(next.name)) { setFile(null); setUploadError('Choose a PDF, PNG, or JPEG file.'); return; }
    if (next.size > 15 * 1024 * 1024) { setFile(null); setUploadError('This file is over the 15 MB limit. Choose a smaller one.'); return; }
    setFile(next);
  }

  async function extractText() {
    if (!file) { setUploadError('Choose a PDF, PNG, or JPEG file first.'); return; }
    setExtracting(true); setUploadError('');
    try {
      const form = new FormData(); form.append('file', file);
      const r = await fetch(`${API}/extract`, { method: 'POST', body: form });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || 'The document could not be processed.');
      setText(body.text || ''); setExtract(body);
      if (!body.text?.trim()) setUploadError('No readable text was found. You can still enter the posting details manually.');
    } catch (e) { setUploadError(e.message || 'The extraction service is unavailable.'); }
    finally { setExtracting(false); }
  }

  async function assess(event) {
    event?.preventDefault(); setError(''); setResult(null);
    if (!Object.values(posting).some(v => v.trim()) && !text.trim()) { setError('Add posting text or extract it from a document first.'); return; }
    setBusy(true);
    const payload = { ...posting, description: text.trim() ? `Uploaded job posting text:\n\n${text.trim()}` : '' };
    try {
      const r = await fetch(`${API}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || 'The assessment service could not process this posting.');
      setResult(body);
      navigate('results');
    } catch (e) { setError(e.message || 'The local assessment API could not be reached.'); }
    finally { setBusy(false); }
  }

  function clearAll() {
    setPosting(EMPTY); setFile(null); setText(''); setExtract(null); setUploadError(''); setError(''); setResult(null);
    const input = document.getElementById('job-file'); if (input) input.value = '';
  }
  function sample() {
    setPosting({ role_and_location: 'Junior Data Analyst — Mumbai, India / hybrid', company_and_recruiter: 'BrightPath Analytics is a fictional company. Apply through its company website.', work_and_qualifications: 'Review datasets and prepare reports with the analytics team. Entry-level applicants with basic spreadsheet skills may apply.', pay_and_application: 'Illustrative salary: ₹3.5–5.0 LPA. Full-time role. No application fee requested.' });
    setFile(null); setText(''); setExtract(null); setUploadError(''); setError(''); setResult(null);
  }
  function drop(event) { event.preventDefault(); setDrag(false); chooseFile(event.dataTransfer.files?.[0]); }

  const fraud = result ? Math.max(0, Math.min(100, Number(result.fraud_probability || 0) * 100)) : 0;
  const genuine = result ? Math.max(0, Math.min(100, Number(result.genuine_probability ?? (1 - Number(result.fraud_probability || 0))) * 100)) : 0;
  const modelName = MODELS[result?.model_used || model] || result?.model_used || 'Model';
  const risk = ['low', 'moderate', 'high'].includes(result?.risk_level) ? result.risk_level : 'moderate';
  const label = result?.assessment === 'genuine' ? 'Genuine classification' : 'Potentially fraudulent';

  return <>
    <div className="topline"><i /> Local job-posting review · No account needed</div>
    <nav className="nav"><button className="brand" type="button" onClick={() => navigate('home')}><span className="brand-logo"><img src={brandLogo} alt="Bharat Rozgaar Sentinel logo" /></span><span className="brand-name">Bharat Rozgaar Sentinel</span></button><div className="nav-links"><PageLink page="home" active={page === 'home'} onNavigate={navigate}>About / Overview</PageLink><PageLink page="analyze" active={page === 'analyze'} onNavigate={navigate}>Analyze Posting</PageLink><PageLink page="results" active={page === 'results'} onNavigate={navigate}>Results &amp; Indicators</PageLink></div></nav>

    <main id="home">
      {page === 'home' && <>
        <section className="hero wrap"><div className="hero-copy"><div className="eyebrow">01 <span /> EMPLOYMENT SAFETY, WITH CONTEXT</div><h1>Before you say yes,<br /><em>look a little closer.</em></h1><p>Review a job posting for patterns associated with employment scams. Get an AI-based risk assessment and see which signals deserve a closer look.</p><button className="primary" type="button" onClick={() => navigate('analyze')}>Start job assessment <b>↗</b></button><div className="hero-points"><span>✓ No signup</span><span>✓ PDF and image input</span><span>✓ Explainable signals</span></div></div><div className="hero-art"><div className="orbit o1" /><div className="orbit o2" /><div className="art-card"><div className="art-top"><span>POSTING REVIEW</span><span className="live-dot">● MODEL READY</span></div><div className="shield">◇</div><small>AI-BASED RISK ASSESSMENT</small><h2>Signals to consider.</h2><div className="art-bars"><i /><i /><i /></div><div className="bar-labels"><span>Genuine indicators</span><span>Risk indicators</span></div><div className="art-foot">A useful check. Never a final verdict.</div></div><div className="float float-b">Read the details,<br />not just the promise.</div></div></section>
        <section className="home-info wrap"><div className="section-head"><div><div className="eyebrow">WHAT IS BHARAT ROZGAAR SENTINEL?</div><h2>A careful second look at a job posting.</h2><p>A practical way to review job details and spot signals worth checking.</p></div></div><div className="steps"><article><b>01</b><i>▤</i><h3>Bring the posting</h3><p>Paste details into four plain-language sections, or upload a PDF or screenshot.</p></article><article><b>02</b><i>✳</i><h3>Review the text</h3><p>Correct any mistakes in extracted text before sending it for analysis.</p></article><article><b>03</b><i>◇</i><h3>Check the signals</h3><p>The model estimates risk and highlights patterns worth investigating.</p></article><article><b>04</b><i>✓</i><h3>Verify independently</h3><p>Check the employer’s official website and contact details. Don’t pay to apply.</p></article></div></section>
        <section className="about"><div className="wrap about-grid"><div><div className="eyebrow">ABOUT THE PROJECT</div><h2>Research-led.<br /><em>People-first.</em></h2><p>Bharat Rozgaar Sentinel explores how natural language processing and deep learning can help people review suspicious employment postings.</p><p>It reviews job-posting text and offers an AI-based risk assessment to support independent checks.</p></div><div className="about-card"><span>ⓘ</span><h3>Keep a healthy amount of doubt.</h3><p>The model has not been validated on offer letters or every type of scam. A low risk score does not prove an offer is safe.</p><ul><li>Verify the company and recruiter independently.</li><li>Be cautious about requests for money or sensitive details.</li><li>Use the assessment as one signal among several.</li></ul></div></div></section>
      </>}

      {page === 'analyze' && <section className="analyze wrap page-section"><div className="section-head"><div><div className="eyebrow">02 <span /> POSTING ANALYZER</div><h2>What did you find?</h2><p>Paste the details you have, or upload a document and review the extracted text.</p></div><button className="link-button" type="button" onClick={sample}>Load sample posting ↗</button></div>
        <div className="analyze-grid"><form className="form-card" onSubmit={assess}><div className="form-top"><span>ADD POSTING TEXT</span><button type="button" onClick={clearAll}>Clear all</button></div>
          <div className={`dropzone ${drag ? 'drag' : ''} ${file ? 'selected' : ''}`} onDragOver={e => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={drop}><span className="upload-icon">↑</span><div className="file-copy"><b>{file ? file.name : 'Have a screenshot or PDF?'}</b><small>{file ? `${(file.size / 1048576).toFixed(1)} MB · ready to extract` : 'Drop it here or browse from your device'}</small></div><label htmlFor="job-file" className="choose">{file ? 'Change file' : 'Choose file'}</label><input id="job-file" type="file" accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg" onChange={e => chooseFile(e.target.files?.[0])} /></div>
          <div className="upload-row"><span>PDF, PNG or JPG · up to 15 MB</span><button type="button" onClick={extractText} disabled={!file || extracting}>{extracting ? 'Extracting…' : 'Extract text'}</button></div>
          {uploadError && <p className="error" role="alert">{uploadError}</p>}
          {extract && <div className="extracted"><div className="extract-title"><span>✓</span><b>Text extracted</b><small>{extract.extraction_method} · {extract.page_count} page{extract.page_count === 1 ? '' : 's'}</small></div><label htmlFor="extracted-text">Review and edit before assessment</label><textarea id="extracted-text" rows="7" value={text} onChange={e => setText(e.target.value)} /><p>{extract.warning || 'Text extraction complete. Review before continuing.'}</p></div>}
          <div className="divider"><span>OR ADD DETAILS BY SECTION</span></div>
          <div className="fields"><label>Job and location<textarea name="role_and_location" rows="3" value={posting.role_and_location} onChange={e => setPosting({ ...posting, [e.target.name]: e.target.value })} placeholder="Role, title, location, remote or on-site details…" /></label><label>Company and recruiter<textarea name="company_and_recruiter" rows="3" value={posting.company_and_recruiter} onChange={e => setPosting({ ...posting, [e.target.name]: e.target.value })} placeholder="Company background, recruiter, contact or website…" /></label><label className="wide">Work and qualifications<textarea name="work_and_qualifications" rows="4" value={posting.work_and_qualifications} onChange={e => setPosting({ ...posting, [e.target.name]: e.target.value })} placeholder="Responsibilities, experience, skills, education, work arrangement…" /></label><label className="wide">Pay, benefits and next steps<textarea name="pay_and_application" rows="4" value={posting.pay_and_application} onChange={e => setPosting({ ...posting, [e.target.name]: e.target.value })} placeholder="Salary, benefits, how to apply, interview steps, fees or payments…" /></label></div>
          {error && <p className="error form-error" role="alert">{error}</p>}<button className="primary submit" type="submit" disabled={busy || api !== 'online'}>{busy ? 'Reviewing posting…' : 'Assess this posting'} <b>{busy ? '◌' : '↗'}</b></button>{api !== 'online' && <p className="api-help">Start the local demo server, then refresh this page.</p>}<p className="privacy">Uploads are processed by this local app and are not saved as files. Review the extracted text before assessment.</p>
        </form>
        <aside className="method-card"><div className="eyebrow">EVALUATION METHODOLOGY</div><h3>Signals, not verdicts.</h3><p>We look at wording across the posting and show a risk assessment from the selected model.</p><div><b>01</b><span><strong>Contextual text review</strong><small>Language is combined across the details you provide.</small></span></div><div><b>02</b><span><strong>Transparent result</strong><small>Review model probabilities and available risk indicators.</small></span></div><div><b>03</b><span><strong>Your decision</strong><small>Verify the employer through independent, official channels.</small></span></div><p className="method-note">The model can be wrong. Its output is not proof that a job is genuine or fraudulent.</p></aside></div>
      </section>}

      {page === 'results' && <section className="review-page wrap page-section">{result ? <>
        <div className="review-kicker"><span>03 / POSTING ASSESSMENT</span><button className="print-button" type="button" onClick={() => window.print()}>Print this summary</button></div>
        <section className={`review-summary ${risk}`}><div><div className="eyebrow">AI-BASED RISK ASSESSMENT</div><h1>{risk === 'low' ? 'Fewer warning signals found' : risk === 'high' ? 'Several signals need attention' : 'Review these signals before deciding'}</h1><p>Summary first; model probabilities, model contributions, and review indicators below show how to interpret this estimate. It is not a definitive determination.</p></div><div className={`risk-badge ${risk}`}><small>ASSESSMENT</small><b>{risk} risk</b><span>{label}</span></div></section>
        <div className="review-layout"><section className="evidence-panel"><div className="panel-heading"><div><div className="eyebrow">PART I / MODEL OUTPUT</div><h2>Probability estimates</h2></div><span className="model-chip">{modelName}</span></div><div className="probability-grid"><article><small>POTENTIALLY FRAUDULENT</small><b>{fraud.toFixed(1)}%</b><i className="fraud-track"><em style={{ width: `${fraud}%` }} /></i></article><article><small>GENUINE</small><b>{genuine.toFixed(1)}%</b><i className="genuine-track"><em style={{ width: `${genuine}%` }} /></i></article></div><div className="panel-heading signal-heading"><div><div className="eyebrow">PART II / INDICATORS</div><h2>Signals to review</h2></div><span>{(result.risk_indicators || []).length} reported</span></div>{result.risk_indicators?.length ? <div className="review-signals">{result.risk_indicators.map((item, i) => <article key={i}><span className="signal-index">{String(i + 1).padStart(2, '0')}</span><div><h3><span className="reason-tag">{SHORT_REASONS[item.indicator?.toLowerCase()] || 'Review signal'}</span>{item.indicator}</h3><p>{item.detail}</p></div><span className="signal-mark">↗</span></article>)}</div> : <p className="no-signals">No rule-based warning indicators were triggered. This does not establish that the posting is safe.</p>}{result.model_token_evidence?.length > 0 && <div className="token-evidence"><h3>Model wording evidence</h3><p className="evidence-note">Learned word associations from the TF-IDF component; these do not prove fraud.</p><div className="tokens">{result.model_token_evidence.map((item, i) => <span key={i} className={item.direction}>{item.term}<small>{item.direction.includes('fraud') ? 'Fraud cue' : 'Genuine cue'}</small></span>)}</div></div>}</section>
          <aside className="verification-panel"><div className="eyebrow">PART III / VERIFY NEXT</div><h2>What to verify next</h2><p>Use independent sources to confirm details before sharing money or sensitive information.</p><ol><li><b>01</b><span><strong>Check the official employer site</strong><small>Find the role and recruiter contact using the company’s published channels.</small></span></li><li><b>02</b><span><strong>Verify the communication channel</strong><small>Look for a matching company domain and independently confirmed contact details.</small></span></li><li><b>03</b><span><strong>Pause before paying or sharing ID</strong><small>Be cautious if asked for fees or sensitive documents early in recruitment.</small></span></li></ol><div className="caution-note">This app does not verify company registration, recruiter identity, or offer-letter authenticity.</div></aside></div>
        <div className="metrics-area"><button className="metrics-toggle" type="button" onClick={toggleMetrics}>{showMetrics ? 'Hide model evaluation metrics' : 'View model evaluation metrics'} <span>{showMetrics ? '−' : '+'}</span></button>{showMetrics && <section className="metrics-panel"><div className="metrics-panel-head"><div><h2>Held-out model results</h2><p>Scores describe the dataset test split, not certainty for this posting.</p></div><span>TEST EVALUATION</span></div>{metricsLoading ? <p className="metrics-message">Loading saved evaluation reports…</p> : metricsError ? <p className="metrics-message error">{metricsError}</p> : metrics?.length ? <div className="metrics-table-wrap"><table><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th></tr></thead><tbody>{metrics.map(row => <tr className={row.model === 'ensemble' ? 'ensemble-row' : ''} key={row.model}><td>{METRIC_MODEL_NAMES[row.model] || row.model}</td>{['accuracy','precision','recall','f1','roc_auc','pr_auc'].map(key => <td key={key}>{row[key] == null ? '—' : `${(Number(row[key]) * 100).toFixed(2)}%`}</td>)}</tr>)}</tbody></table></div> : <p className="metrics-message">No saved evaluation reports were found.</p>}</section>}</div>
        <div className="review-actions"><button className="secondary-button" type="button" onClick={() => navigate('analyze')}>Back to posting</button><button className="primary" type="button" onClick={() => { clearAll(); navigate('analyze'); }}>Analyze another posting <b>↗</b></button></div>
      </> : <div className="empty-review"><div className="eyebrow">03 / RESULTS &amp; INDICATORS</div><h1>Your assessment will appear here.</h1><p>Submit a job posting first. You’ll see model probabilities, risk indicators and practical checks to consider.</p><button className="primary" type="button" onClick={() => navigate('analyze')}>Analyze a posting <b>↗</b></button></div>}</section>}
    </main>
    <footer className="footer"><div className="wrap footer-main"><button className="brand" type="button" onClick={() => navigate('home')}><span className="brand-logo"><img src={brandLogo} alt="Bharat Rozgaar Sentinel logo" /></span><span className="brand-name">Bharat Rozgaar Sentinel</span></button><p>AI-based risk assessment · Not proof of fraud</p><button className="footer-home" type="button" onClick={() => navigate('home')}>Back to overview ↑</button></div><div className="wrap footer-bottom"><span>© Bharat Rozgaar Sentinel</span></div></footer>
  </>;
}

