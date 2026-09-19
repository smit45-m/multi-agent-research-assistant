import React, { useCallback, useEffect, useRef, useState } from 'react';
import { marked } from 'marked';
import {
  Camera,
  Mic,
  Square,
  UploadCloud,
  FileSpreadsheet,
  FileText,
  ImageIcon,
  AudioLines,
  Sparkles,
  X,
  Loader2,
  Database,
  RefreshCw,
} from 'lucide-react';
import { startWavRecording, type WavRecorderHandle } from '../lib/wavEncoder';

interface AnalystTabProps {
  onShowToast: (msg: string) => void;
  onAskResearch?: (question: string) => void;
}

interface Briefing {
  agent: string;
  kind: string;
  markdown: string;
  highlights: string[];
  followups: string[];
  chunks_indexed?: number;
}

type InputMode = 'idle' | 'camera' | 'mic';

const ACCEPTED =
  '.png,.jpg,.jpeg,.webp,.gif,.bmp,.wav,.csv,.tsv,.xlsx,.json,.txt,.md,.py,.js,.ts,.html,.yaml,.xml,.sh';

export const AnalystTab: React.FC<AnalystTabProps> = ({ onShowToast, onAskResearch }) => {
  const [mode, setMode] = useState<InputMode>('idle');
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState('');
  const [briefing, setBriefing] = useState<Briefing | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordSecs, setRecordSecs] = useState(0);
  const [micLevel, setMicLevel] = useState(0);
  const [typedMarkdown, setTypedMarkdown] = useState('');

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<WavRecorderHandle | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const levelTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const typewriterRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ---------------------------------------------------------------
  // Typewriter reveal of the briefing (interactive answer feel)
  // ---------------------------------------------------------------
  useEffect(() => {
    if (!briefing) {
      setTypedMarkdown('');
      return;
    }
    if (typewriterRef.current) clearInterval(typewriterRef.current);
    const full = briefing.markdown;
    // Reveal in line-batches so tables materialize row by row
    const lines = full.split('\n');
    let idx = 0;
    setTypedMarkdown('');
    typewriterRef.current = setInterval(() => {
      idx += 2;
      setTypedMarkdown(lines.slice(0, idx).join('\n'));
      if (idx >= lines.length && typewriterRef.current) {
        clearInterval(typewriterRef.current);
        typewriterRef.current = null;
      }
    }, 55);
    return () => {
      if (typewriterRef.current) clearInterval(typewriterRef.current);
    };
  }, [briefing]);

  // ---------------------------------------------------------------
  // Shared submit
  // ---------------------------------------------------------------
  const submit = useCallback(
    async (endpoint: string, blob: Blob, filename: string, label: string) => {
      setBusy(true);
      setBusyLabel(label);
      setBriefing(null);
      try {
        const form = new FormData();
        form.append('file', blob, filename);
        const res = await fetch(endpoint, { method: 'POST', body: form });
        const body = await res.json();
        if (!res.ok) {
          throw new Error(body?.detail || `HTTP ${res.status}`);
        }
        setBriefing(body as Briefing);
        onShowToast('🧪 Interactive Analyst finished measuring your input!');
      } catch (err: any) {
        onShowToast(`Analysis failed: ${err.message || err}`);
      } finally {
        setBusy(false);
        setBusyLabel('');
      }
    },
    [onShowToast],
  );

  // ---------------------------------------------------------------
  // Camera
  // ---------------------------------------------------------------
  const openCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      setMode('camera');
      // Wait a tick for the <video> to mount
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => undefined);
        }
      });
    } catch (err: any) {
      onShowToast(
        `📷 Camera unavailable: ${err?.name === 'NotAllowedError' ? 'permission denied' : err.message || err}`,
      );
    }
  };

  const closeCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setMode('idle');
  }, []);

  const capturePhoto = async () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) {
      onShowToast('Camera not ready yet — give it a second.');
      return;
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d')!.drawImage(video, 0, 0);
    const blob: Blob = await new Promise((resolve) =>
      canvas.toBlob((b) => resolve(b!), 'image/png'),
    );
    closeCamera();
    await submit(
      '/api/v1/media/capture/photo',
      blob,
      'live-photo.png',
      '📷 Measuring your photo pixels…',
    );
  };

  // ---------------------------------------------------------------
  // Microphone
  // ---------------------------------------------------------------
  const startMic = async () => {
    try {
      const handle = await startWavRecording();
      recorderRef.current = handle;
      setMode('mic');
      setRecording(true);
      setRecordSecs(0);
      timerRef.current = setInterval(() => setRecordSecs((s) => s + 1), 1000);
      levelTimerRef.current = setInterval(
        () => setMicLevel(handle.getLevel()),
        100,
      );
    } catch (err: any) {
      onShowToast(
        `🎙️ Microphone unavailable: ${err?.name === 'NotAllowedError' ? 'permission denied' : err.message || err}`,
      );
    }
  };

  const stopMic = async () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (levelTimerRef.current) clearInterval(levelTimerRef.current);
    setRecording(false);
    setMode('idle');
    const handle = recorderRef.current;
    recorderRef.current = null;
    if (!handle) return;
    const blob = await handle.stop();
    await submit(
      '/api/v1/media/capture/audio',
      blob,
      'live-audio.wav',
      '🎙️ Measuring your recording waveform…',
    );
  };

  const cancelMic = async () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (levelTimerRef.current) clearInterval(levelTimerRef.current);
    setRecording(false);
    setMode('idle');
    const handle = recorderRef.current;
    recorderRef.current = null;
    if (handle) await handle.stop(); // discard blob
  };

  // Cleanup on unmount
  useEffect(
    () => () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      if (timerRef.current) clearInterval(timerRef.current);
      if (levelTimerRef.current) clearInterval(levelTimerRef.current);
    },
    [],
  );

  // ---------------------------------------------------------------
  // Files
  // ---------------------------------------------------------------
  const handleFile = async (file: File, index = false) => {
    const label = file.name.match(/\.(csv|tsv|xlsx|json)$/i)
      ? '📊 Profiling your dataset…'
      : file.name.match(/\.(png|jpe?g|webp|gif|bmp)$/i)
        ? '🖼️ Measuring image pixels…'
        : file.name.match(/\.wav$/i)
          ? '🎵 Measuring audio samples…'
          : '📄 Reading your document…';
    const endpoint = index
      ? '/api/v1/media/analyze/index'
      : '/api/v1/media/analyze';
    await submit(endpoint, file, file.name, label);
  };

  const onDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) await handleFile(file);
  };

  const renderedHtml = typedMarkdown
    ? (marked.parse(typedMarkdown) as string)
    : '';

  return (
    <div className="analyst-layout">
      {/* ------------ INPUT SIDE ------------ */}
      <div className="analyst-inputs">
        <div className="analyst-header card-rise">
          <div className="analyst-agent-badge">
            <Sparkles size={14} />
            Agent 07 · Interactive Analyst
          </div>
          <h2>Show it anything. It measures everything.</h2>
          <p>
            Live camera frames, microphone audio, datasets, and documents —
            analyzed from raw bytes on the server, presented back with
            emojis, meters, and tables. No made-up numbers.
          </p>
        </div>

        {/* Live capture buttons */}
        <div className="capture-grid">
          <button
            className={`capture-tile ${mode === 'camera' ? 'active' : ''}`}
            onClick={mode === 'camera' ? closeCamera : openCamera}
            disabled={busy || recording}
          >
            <Camera size={22} />
            <span className="capture-title">Live Photo</span>
            <span className="capture-sub">Camera → pixel analysis</span>
          </button>
          <button
            className={`capture-tile ${recording ? 'recording' : ''}`}
            onClick={recording ? stopMic : startMic}
            disabled={busy || mode === 'camera'}
          >
            {recording ? <Square size={22} /> : <Mic size={22} />}
            <span className="capture-title">
              {recording ? `Stop (${recordSecs}s)` : 'Live Audio'}
            </span>
            <span className="capture-sub">
              {recording ? 'Recording 16-bit WAV…' : 'Mic → waveform analysis'}
            </span>
          </button>
        </div>

        {/* Camera viewfinder */}
        {mode === 'camera' && (
          <div className="viewfinder card-rise">
            <video ref={videoRef} playsInline muted />
            <div className="viewfinder-controls">
              <button className="btn-shutter" onClick={capturePhoto}>
                <Camera size={18} /> Capture & Analyze
              </button>
              <button className="btn-ghost-sm" onClick={closeCamera}>
                <X size={16} />
              </button>
            </div>
          </div>
        )}

        {/* Mic level meter */}
        {recording && (
          <div className="mic-meter-card card-rise">
            <AudioLines size={18} className="pulse-icon" />
            <div className="mic-meter-track">
              <div
                className="mic-meter-fill"
                style={{ width: `${Math.min(100, micLevel * 260)}%` }}
              />
            </div>
            <button className="btn-ghost-sm" onClick={cancelMic} title="Discard">
              <X size={16} />
            </button>
          </div>
        )}

        {/* Drop zone */}
        <div
          className={`dropzone ${dragOver ? 'drag-over' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
        >
          <UploadCloud size={26} />
          <strong>Drop any file — or click to browse</strong>
          <div className="dropzone-formats">
            <span>
              <ImageIcon size={13} /> Images
            </span>
            <span>
              <AudioLines size={13} /> WAV audio
            </span>
            <span>
              <FileSpreadsheet size={13} /> CSV · XLSX · JSON
            </span>
            <span>
              <FileText size={13} /> Text · Markdown · Code
            </span>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED}
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
              e.currentTarget.value = '';
            }}
          />
        </div>

        <button
          className="index-hint-btn"
          disabled={busy}
          onClick={() => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.csv,.tsv,.json,.txt,.md';
            input.onchange = () => {
              const f = input.files?.[0];
              if (f) handleFile(f, true);
            };
            input.click();
          }}
        >
          <Database size={15} />
          Analyze <em>and</em> index into the knowledge base
        </button>
      </div>

      {/* ------------ OUTPUT SIDE ------------ */}
      <div className="analyst-output">
        {busy && (
          <div className="analyst-busy card-rise">
            <Loader2 size={28} className="spin" />
            <p>{busyLabel}</p>
            <span>Measuring raw bytes server-side…</span>
          </div>
        )}

        {!busy && !briefing && (
          <div className="analyst-empty">
            <div className="empty-orbit">
              <span className="orbit-emoji e1">🖼️</span>
              <span className="orbit-emoji e2">🎙️</span>
              <span className="orbit-emoji e3">📊</span>
              <span className="orbit-emoji e4">📄</span>
              <div className="orbit-core">🧪</div>
            </div>
            <h3>The Analyst is ready</h3>
            <p>
              Capture a photo, record your voice, or drop a file. Every number
              in the briefing is computed from your actual input.
            </p>
          </div>
        )}

        {!busy && briefing && (
          <div className="briefing card-rise">
            <div className="briefing-topbar">
              <div className="briefing-kind">
                {briefing.kind === 'image' && <ImageIcon size={15} />}
                {briefing.kind === 'audio' && <AudioLines size={15} />}
                {briefing.kind === 'tabular' && <FileSpreadsheet size={15} />}
                {briefing.kind === 'text' && <FileText size={15} />}
                <span>{briefing.kind} briefing</span>
              </div>
              {typeof briefing.chunks_indexed === 'number' &&
                briefing.chunks_indexed > 0 && (
                  <span className="indexed-pill">
                    <Database size={12} /> {briefing.chunks_indexed} chunks
                    indexed
                  </span>
                )}
              <button
                className="btn-ghost-sm"
                onClick={() => setBriefing(null)}
                title="Clear"
              >
                <RefreshCw size={14} />
              </button>
            </div>

            <div className="highlight-row">
              {briefing.highlights.map((h, i) => (
                <div
                  className="highlight-chip"
                  key={i}
                  style={{ animationDelay: `${i * 120}ms` }}
                >
                  {h}
                </div>
              ))}
            </div>

            <div
              className="markdown-body briefing-markdown"
              dangerouslySetInnerHTML={{ __html: renderedHtml }}
            />

            {briefing.followups?.length > 0 && (
              <div className="followups">
                <span className="followups-label">💡 Try next</span>
                {briefing.followups.map((f, i) => (
                  <button
                    className="followup-chip followup-clickable"
                    key={i}
                    onClick={() => {
                      if (onAskResearch) {
                        onAskResearch(f);
                        onShowToast('🧠 Sent to Research Studio!');
                      } else {
                        navigator.clipboard?.writeText(f);
                        onShowToast('Copied suggestion to clipboard.');
                      }
                    }}
                  >
                    {f}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
