import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronRight,
  FlaskConical,
  Code2,
  Menu,
  Moon,
  Plus,
  Search,
  Sun,
  X,
  LogOut,
} from "lucide-react";
import { StudioTab } from "./components/StudioTab";
import { BenchmarksTab } from "./components/BenchmarksTab";
import { KnowledgeTab } from "./components/KnowledgeTab";
import { ConcurrencyTab } from "./components/ConcurrencyTab";
import { AuthModal } from "./components/AuthModal";
import { api, fetchMeApi, logoutApi, getAuthToken, type HealthInfo } from "./api";
import type { ResearchResponse, User } from "./types";

const pages = [
  { id: "studio", label: "Research studio", icon: Search },
  { id: "knowledge", label: "Knowledge library", icon: BookOpen },
  { id: "benchmarks", label: "Benchmarks", icon: FlaskConical },
  { id: "concurrency", label: "System status", icon: Activity },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("studio");
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    try {
      return localStorage.getItem("research-theme") === "light"
        ? "light"
        : "dark";
    } catch {
      return "dark";
    }
  });
  const [mobileOpen, setMobileOpen] = useState(false);
  const sidebarRef = useRef<HTMLElement>(null);
  const [toast, setToast] = useState("");
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [connection, setConnection] = useState("checking");
  const [history, setHistory] = useState<ResearchResponse[]>([]);
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [resetKey, setResetKey] = useState(0);
  const [busy, setBusy] = useState(false);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [authModalOpen, setAuthModalOpen] = useState(false);

  const notify = useCallback((message: string) => {
    if (toastTimer.current) clearTimeout(toastTimer.current);
    setToast(message);
    toastTimer.current = setTimeout(() => setToast(""), 5000);
  }, []);

  useEffect(() => {
    const token = getAuthToken();
    if (token) {
      fetchMeApi()
        .then((user) => setCurrentUser(user))
        .catch(() => setCurrentUser(null));
    }
  }, []);

  const handleLogout = useCallback(async () => {
    try {
      await logoutApi();
    } catch {
      // Ignored
    }
    setCurrentUser(null);
    notify("Signed out successfully.");
  }, [notify]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("research-theme", theme);
    } catch {
      /* Storage is optional. */
    }
  }, [theme]);

  const checkHealth = useCallback(async () => {
    setConnection("checking");
    try {
      const data = await api<HealthInfo>("/health/ready", {
        signal: AbortSignal.timeout(6000),
      });
      setHealth(data);
      setConnection(data.status === "ok" ? "connected" : "degraded");
    } catch {
      setHealth(null);
      setConnection("offline");
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    api<HealthInfo>("/health/ready", {
      signal: AbortSignal.any([controller.signal, AbortSignal.timeout(6000)]),
    })
      .then((data) => {
        setHealth(data);
        setConnection(data.status === "ok" ? "connected" : "degraded");
      })
      .catch(() => {
        if (!controller.signal.aborted) setConnection("offline");
      });
    return () => {
      controller.abort();
      if (toastTimer.current) clearTimeout(toastTimer.current);
    };
  }, []);
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileOpen(false);
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, []);

  useEffect(() => {
    if (!mobileOpen) return;
    const previous = document.activeElement as HTMLElement | null;
    const sidebar = sidebarRef.current;
    const focusable = () => Array.from(sidebar?.querySelectorAll<HTMLElement>('a[href], button:not(:disabled)') || []);
    focusable()[0]?.focus();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const trapFocus = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') return;
      const items = focusable();
      const first = items[0];
      const last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener('keydown', trapFocus);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener('keydown', trapFocus);
      previous?.focus();
    };
  }, [mobileOpen]);

  function navigate(id: string) {
    setActiveTab(id);
    setMobileOpen(false);
  }
  function newResearch() {
    if (busy) {
      notify("Stop waiting for the current request before starting a new one.");
      navigate("studio");
      return;
    }
    setResult(null);
    setResetKey((key) => key + 1);
    navigate("studio");
  }
  function complete(data: ResearchResponse) {
    setResult(data);
    setHistory((items) =>
      [data, ...items.filter((item) => item.task_id !== data.task_id)].slice(
        0,
        8,
      ),
    );
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      {mobileOpen && (
        <button
          className="sidebar-overlay"
          aria-label="Close navigation"
          onClick={() => setMobileOpen(false)}
        />
      )}
      <aside
        ref={sidebarRef}
        className={`app-sidebar ${mobileOpen ? "is-open" : ""}`}
        aria-label="Workspace navigation"
      >
        <a
          className="brand"
          href="#studio"
          onClick={(event) => {
            event.preventDefault();
            navigate("studio");
          }}
        >
          <span className="brand-symbol">
            <span />
            <span />
            <span />
            <span />
          </span>
          <span>
            Research<span className="brand-dot">.</span>
            <small>MULTI-AGENT WORKSPACE</small>
          </span>
        </a>
        <button className="new-research" onClick={newResearch}>
          <Plus size={17} /> New research <span className="tiny-plus">+</span>
        </button>
        <div className="nav-label">WORKSPACE</div>
        <nav className="sidebar-nav">
          {pages.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${activeTab === id ? "active" : ""}`}
              onClick={() => navigate(id)}
              aria-current={activeTab === id ? "page" : undefined}
            >
              <Icon size={18} />
              <span>{label}</span>
              {activeTab === id && <span className="nav-active-dot" />}
            </button>
          ))}
        </nav>
        <div className="history-heading">
          <span className="nav-label">THIS SESSION</span>
          <span className="count">{history.length}</span>
        </div>
        <div className="sidebar-history">
          {history.length ? (
            history.map((item) => (
              <button
                key={item.task_id}
                className="history-item"
                disabled={busy}
                onClick={() => {
                  setResult(item);
                  setResetKey((key) => key + 1);
                  navigate("studio");
                }}
                title={item.query}
              >
                <span className="history-dot" />
                <span>{item.query}</span>
              </button>
            ))
          ) : (
            <p>
              Your next discovery starts
              <br />
              with a question.
            </p>
          )}
        </div>
        <div className="sidebar-bottom">
          <div className="workspace-note">
            <span className="note-icon">
              <BookOpen size={17} />
            </span>
            <strong>Bring your own context</strong>
            <p>Add documents for more focused, grounded research.</p>
            <button
              className="text-button"
              onClick={() => navigate("knowledge")}
            >
              Open library <ArrowUpRight size={14} />
            </button>
          </div>
          <a
            className="repo-link"
            href="https://github.com/smit45-m/multi-agent-research-assistant"
            target="_blank"
            rel="noreferrer"
          >
            <Code2 size={16} /> View project <ArrowUpRight size={13} />
          </a>
          <div
            className="workspace-profile"
            onClick={() => {
              if (!currentUser) setAuthModalOpen(true);
            }}
            style={{ cursor: currentUser ? "default" : "pointer" }}
            title={currentUser ? `Signed in as ${currentUser.email}` : "Click to sign in"}
          >
            <span className="profile-avatar">
              {currentUser?.full_name ? currentUser.full_name.charAt(0).toUpperCase() : "R"}
            </span>
            <div>
              {currentUser?.full_name || "Personal workspace"}
              <small>
                {currentUser
                  ? `${currentUser.role.toUpperCase()} • ${currentUser.email}`
                  : "Sign in to save research"}
              </small>
            </div>
          </div>
        </div>
      </aside>
      <div className="app-body" inert={mobileOpen}>
        <header className="topbar">
          <div className="breadcrumbs">
            <button
              className="icon-button mobile-menu"
              aria-label={mobileOpen ? "Close navigation" : "Open navigation"}
              aria-expanded={mobileOpen}
              onClick={() => setMobileOpen(!mobileOpen)}
            >
              <Menu size={19} />
            </button>
            <span>Workspace</span>
            <ChevronRight size={13} />
            <strong>
              {pages.find((page) => page.id === activeTab)?.label}
            </strong>
          </div>
          <div className="topbar-actions">
            <button
              className={`connection ${connection}`}
              onClick={() => {
                navigate("concurrency");
                void checkHealth();
              }}
              title="Check backend connection"
            >
              <span />
              {connection === "connected"
                ? "Backend connected"
                : connection === "checking"
                  ? "Checking connection"
                  : connection === "degraded"
                    ? "Setup needed"
                    : "Backend offline"}
            </button>
            <span className="topbar-divider" />
            <button
              className="icon-button"
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
            </button>
            <span className="topbar-divider" />
            {currentUser ? (
              <div
                className="auth-user-pill"
                title={`${currentUser.email} (${currentUser.role})`}
              >
                <span className="auth-user-avatar">
                  {currentUser.full_name ? currentUser.full_name.charAt(0).toUpperCase() : "U"}
                </span>
                <span className="auth-user-name">{currentUser.full_name}</span>
                <span className={`auth-role-tag ${currentUser.role}`}>
                  {currentUser.role}
                </span>
                <button
                  className="auth-signout-btn"
                  title="Sign out"
                  onClick={handleLogout}
                  aria-label="Sign out"
                >
                  <LogOut size={13} />
                </button>
              </div>
            ) : (
              <button
                className="auth-topbar-btn"
                onClick={() => setAuthModalOpen(true)}
              >
                Sign In
              </button>
            )}
          </div>
        </header>
        <main id="main-content" className="main-content" tabIndex={-1}>
          <div hidden={activeTab !== "studio"}>
            <StudioTab
              key={resetKey}
              onShowToast={notify}
              result={result}
              onResult={complete}
              resetKey={resetKey}
              onBusyChange={setBusy}
              onOpenLibrary={() => navigate("knowledge")}
            />
          </div>
          {activeTab === "knowledge" && <KnowledgeTab onShowToast={notify} />}
          {activeTab === "benchmarks" && <BenchmarksTab onShowToast={notify} />}
          {activeTab === "concurrency" && (
            <ConcurrencyTab
              health={health}
              connection={connection}
              onRefresh={checkHealth}
            />
          )}
          <footer className="page-footer">
            <span>Built for curious minds.</span>
            <span>
              AI-generated research can be imperfect. Always check the sources.
            </span>
          </footer>
        </main>
      </div>
      <div className="toast-region" role="status" aria-live="polite">
        {toast && (
          <div className="toast">
            <Check size={17} />
            <span>{toast}</span>
            <button
              className="icon-button"
              aria-label="Dismiss notification"
              onClick={() => setToast("")}
            >
              <X size={15} />
            </button>
          </div>
        )}
      </div>
      <AuthModal
        isOpen={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        onSuccess={(user, message) => {
          setCurrentUser(user);
          notify(message);
        }}
      />
    </div>
  );
}
