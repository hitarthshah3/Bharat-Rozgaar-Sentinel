import React, { useEffect, useState } from 'react';

const API = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '');
const EMPTY = { role_and_location: '', company_and_recruiter: '', work_and_qualifications: '', pay_and_application: '' };
const MODELS = { baseline: 'TF-IDF + Logistic Regression', bilstm: 'Word2Vec + BiLSTM', hybrid_bilstm: 'Hybrid BiLSTM', distilbert: 'DistilBERT' };
const SCORES = [
  ['TF-IDF + Logistic Regression', '98.99%', '89.14%', '90.17%', '89.66%', '99.34%', 'Current website model · threshold 0.50'],
  ['Word2Vec + BiLSTM', '93.18%', '34.89%', '47.40%', '40.20%', '85.58%', 'Text-only deep learning · threshold 0.50'],
  ['Hybrid BiLSTM', '95.55%', '56.25%', '36.42%', '44.21%', '90.81%', 'Text + metadata · validation threshold 0.6107'],
  ['DistilBERT', '98.24%', '90.44%', '71.10%', '79.61%', '98.06%', 'Transformer · validation threshold 0.9841'],
];

export default function App() {
  const [posting, setPosting] = useState(EMPTY);
  const [api, setApi] = useState('checking');
  const [model, setModel] = useState('baseline');
  const [file, setFile] = useState(null);
  const [text, setText] = useState('');
  const [extract, setExtract] = useState(null);
  const [uploadError, setUploadError] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
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
      setTimeout(() => document.getElementById('result-card')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60);
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

  const fraud = result ? Math.max(0, Math.min(100, result.fraud_probability * 100)) : 0;
  const modelName = MODELS[result?.model_used || model] || result?.model_used || 'Model';
  const risk = ['low', 'moderate', 'high'].includes(result?.risk_level) ? result.risk_level : 'moderate';

  return <>
    <div className="topline"><i /> Local job-posting review · No account needed</div>
    <nav className="nav"><a className="brand" href="#home"><span className="brand-mark">RS</span><span><b>Rozgaar</b><small>SENTINEL</small></span></a><div className="nav-links"><a href="#analyze">Analyze</a><a href="#how">How it works</a><a href="#models">Model results</a><a href="#about">About</a></div><span className={`api-status ${api}`}><i />{api === 'online' ? 'System online' : api === 'checking' ? 'Connecting' : 'API offline'}</span></nav>

    <main id="home">
      <section className="hero wrap"><div className="hero-copy"><div className="eyebrow">01 <span /> EMPLOYMENT SAFETY, WITH CONTEXT</div><h1>Before you say yes,<br /><em>look a little closer.</em></h1><p>Review a job posting for patterns associated with employment scams. Get an AI-based risk assessment and see which signals deserve a closer look.</p><a className="primary" href="#analyze">Check a job posting <b>↗</b></a><div className="hero-points"><span>✓ No signup</span><span>✓ PDF and image input</span><span>✓ Explainable signals</span></div></div><div className="hero-art"><div className="orbit o1" /><div className="orbit o2" /><div className="art-card"><div className="art-top"><span>POSTING REVIEW</span><span className="live-dot">● MODEL READY</span></div><div className="shield">◇</div><small>AI-BASED RISK ASSESSMENT</small><h2>Signals to consider.</h2><div className="art-bars"><i /><i /><i /></div><div className="bar-labels"><span>Genuine indicators</span><span>Risk indicators</span></div><div className="art-foot">A useful check. Never a final verdict.</div></div><div className="float float-a">17,880 <small>job postings in the research dataset</small></div><div className="float float-b">Read the details,<br />not just the promise.</div></div></section>

      <section className="stats"><div className="stats-inner wrap"><div><b>4</b><span>models compared</span></div><div><b>17,880</b><span>dataset postings</span></div><div><b>866</b><span>fraud-labelled records</span></div><p>Research metrics describe a held-out dataset split, not real-world guarantees.</p></div></section>

      <section className="analyze wrap" id="analyze"><div className="section-head"><div><div className="eyebrow">02 <span /> POSTING ANALYZER</div><h2>What did you find?</h2><p>Paste the details you have, or upload a document and review the extracted text.</p></div><button className="link-button" type="button" onClick={sample}>Load sample posting ↗</button></div>
        <div className="analyze-grid"><form className="form-card" onSubmit={assess}><div className="form-top"><span>ADD POSTING TEXT</span><button type="button" onClick={clearAll}>Clear all</button></div>
          <div className={`dropzone ${drag ? 'drag' : ''} ${file ? 'selected' : ''}`} onDragOver={e => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={drop}><span className="upload-icon">↑</span><div className="file-copy"><b>{file ? file.name : 'Have a screenshot or PDF?'}</b><small>{file ? `${(file.size / 1048576).toFixed(1)} MB · ready to extract` : 'Drop it here or browse from your device'}</small></div><label htmlFor="job-file" className="choose">{file ? 'Change file' : 'Choose file'}</label><input id="job-file" type="file" accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg" onChange={e => chooseFile(e.target.files?.[0])} /></div>
          <div className="upload-row"><span>PDF, PNG or JPG · up to 15 MB</span><button type="button" onClick={extractText} disabled={!file || extracting}>{extracting ? 'Extracting…' : 'Extract text'}</button></div>
          {uploadError && <p className="error" role="alert">{uploadError}</p>}
          {extract && <div className="extracted"><div className="extract-title"><span>✓</span><b>Text extracted</b><small>{extract.extraction_method} · {extract.page_count} page{extract.page_count === 1 ? '' : 's'}</small></div><label htmlFor="extracted-text">Review and edit before assessment</label><textarea id="extracted-text" rows="7" value={text} onChange={e => setText(e.target.value)} /><p>{extract.warning || 'Text extraction complete. Review before continuing.'}</p></div>}
          <div className="divider"><span>OR ADD DETAILS BY SECTION</span></div>
          <div className="fields"><label>Job and location<textarea name="role_and_location" rows="3" value={posting.role_and_location} onChange={e => setPosting({ ...posting, [e.target.name]: e.target.value })} placeholder="Role, title, location, remote or on-site details…" /></label><label>Company and recruiter<textarea name="company_and_recruiter" rows="3" value={posting.company_and_recruiter} onChange={e => setPosting({ ...posting, [e.target.name]: e.target.value })} placeholder="Company background, recruiter, contact or website…" /></label><label className="wide">Work and qualifications<textarea name="work_and_qualifications" rows="4" value={posting.work_and_qualifications} onChange={e => setPosting({ ...posting, [e.target.name]: e.target.value })} placeholder="Responsibilities, experience, skills, education, work arrangement…" /></label><label className="wide">Pay, benefits and next steps<textarea name="pay_and_application" rows="4" value={posting.pay_and_application} onChange={e => setPosting({ ...posting, [e.target.name]: e.target.value })} placeholder="Salary, benefits, how to apply, interview steps, fees or payments…" /></label></div>
          {error && <p className="error form-error" role="alert">{error}</p>}<button className="primary submit" type="submit" disabled={busy || api !== 'online'}>{busy ? 'Reviewing posting…' : 'Assess this posting'} <b>{busy ? '◌' : '↗'}</b></button>{api !== 'online' && <p className="api-help">Start the local demo server, then refresh this page.</p>}<p className="privacy">Uploads are processed by this local app and are not saved as files. Review the extracted text before assessment.</p>
        </form>

        <aside className="result-column" id="result-card" aria-live="polite">{result ? <div className={`result-card ${risk}`}><div className="result-head"><span>AI-BASED ASSESSMENT</span><b className={`risk-pill ${risk}`}>{risk} risk</b></div><div className="model-line"><span className={`result-icon ${risk}`}>{result.assessment === 'genuine' ? '✓' : '!'}</span><div><small>CURRENT MODEL</small><b>{modelName}</b></div></div><h3>{result.assessment === 'genuine' ? 'Fewer fraud signals found' : 'Some signals need a closer look'}</h3><p className="summary">This estimate reflects patterns in the supplied text. It is not a definitive determination.</p><div className="probability"><div><span>Potentially fraudulent</span><b>{fraud.toFixed(1)}%</b></div><i className="fraud-track"><em style={{ width: `${fraud}%` }} /></i><div><span>Genuine</span><b>{(100 - fraud).toFixed(1)}%</b></div><i className="genuine-track"><em style={{ width: `${100 - fraud}%` }} /></i><small>Decision threshold: {Number(result.decision_threshold ?? .5).toFixed(2)}</small></div><div className="signals"><h4>Signals to review <small>{(result.risk_indicators || []).length} found</small></h4>{result.risk_indicators?.length ? result.risk_indicators.map((item, i) => <div className="signal" key={i}><span>↗</span><div><b>{item.indicator}</b><p>{item.detail}</p></div></div>) : <p className="no-signal">No rule-based warning patterns were triggered. Keep verifying the employer.</p>}</div>{result.model_token_evidence?.length > 0 && <div className="signals"><h4>Model wording evidence</h4><div className="tokens">{result.model_token_evidence.map((item, i) => <span key={i} className={item.direction}>{item.term}<small>{item.direction}</small></span>)}</div></div>}<p className="disclaimer">{result.assessment_type}</p></div> : <div className="placeholder"><span className="placeholder-icon">✳</span><small>YOUR REVIEW, YOUR DECISION</small><h3>Signals, not verdicts.</h3><p>After an assessment, this panel will show the model probabilities, risk level, and indicators worth reviewing.</p><hr /><div><b>01</b> We look at the wording across the posting.</div><div><b>02</b> You see the score and contributing signals.</div><div><b>03</b> You decide what to verify next.</div><aside>The model can be wrong. Never treat its output as proof that a job is genuine or fraudulent.</aside></div>}</aside></div>
      </section>

      <section className="how" id="how"><div className="wrap"><div className="center-head"><div className="eyebrow">03 <span /> A CLEARER WAY TO CHECK</div><h2>From posting to perspective.</h2><p>You stay in control at every step.</p></div><div className="steps"><article><b>01</b><i>▤</i><h3>Bring the posting</h3><p>Paste details into four plain-language sections, or upload a PDF or screenshot.</p></article><article><b>02</b><i>✳</i><h3>Review the text</h3><p>Correct any mistakes in extracted text before sending it for analysis.</p></article><article><b>03</b><i>◇</i><h3>Check the signals</h3><p>The NLP model estimates risk and highlights patterns worth investigating.</p></article><article><b>04</b><i>✓</i><h3>Verify independently</h3><p>Check the employer’s official website and contact details. Don’t pay to apply.</p></article></div></div></section>

      <section className="models wrap" id="models"><div className="section-head"><div><div className="eyebrow">04 <span /> MODEL EVALUATION</div><h2>Built, trained and compared.</h2><p>Measured on a stratified held-out test split from the project dataset.</p></div><div className="test-tag"><small>TEST SPLIT</small><b>3,576 postings</b></div></div><div className="table-scroll"><table><thead><tr><th>MODEL</th><th>ACCURACY</th><th>PRECISION</th><th>FRAUD RECALL</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>{SCORES.map((row, i) => <tr key={row[0]} className={i === 0 ? 'best' : ''}><td><b>{row[0]}</b><small>{row[6]}</small></td>{row.slice(1, 6).map((value, j) => <td key={j}>{value}</td>)}</tr>)}</tbody></table></div><div className="metrics-note"><span>ⓘ Accuracy can look high on imbalanced data. Review fraud recall, precision and F1 together.</span><small>One held-out split · not deployment validation</small></div><div className="dataset"><span>DATA<br />SET</span><div><b>Employment Scam Aegean Dataset</b><small>17,880 historical postings · 866 fraudulent · 17,014 genuine</small></div><a href="https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction" target="_blank" rel="noreferrer">Dataset source ↗</a></div></section>

      <section className="about" id="about"><div className="wrap about-grid"><div><div className="eyebrow">05 <span /> ABOUT THE PROJECT</div><h2>Research-led.<br /><em>People-first.</em></h2><p>Bharat Rozgaar Sentinel is a college project exploring how natural language processing and deep learning can help people review suspicious employment postings.</p><p>It compares a TF-IDF baseline with Word2Vec/BiLSTM and DistilBERT models trained on the same dataset.</p></div><div className="about-card"><span>ⓘ</span><h3>Keep a healthy amount of doubt.</h3><p>The training data is historical and mostly contains online job postings. The model has not been validated on offer letters or every type of scam. A low risk score does not prove an offer is safe.</p><ul><li>Verify the company and recruiter independently.</li><li>Be cautious about requests for money or sensitive details.</li><li>Use the assessment as one signal among several.</li></ul></div></div></section>
    </main>
    <footer className="footer"><div className="wrap footer-main"><a className="brand" href="#home"><span className="brand-mark">RS</span><span><b>Rozgaar</b><small>SENTINEL</small></span></a><p>AI-based risk assessment · Not proof of fraud</p><a href="#home">Back to top ↑</a></div><div className="wrap footer-bottom"><span>© Bharat Rozgaar Sentinel · NLP + Deep Learning project</span><span>Built for more informed job searches.</span></div></footer>
  </>;
}
