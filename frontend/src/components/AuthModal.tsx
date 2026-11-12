import { useState, type FormEvent } from "react";
import {
  X,
  Mail,
  Lock,
  User as UserIcon,
  Eye,
  EyeOff,
  ShieldCheck,
  Sparkles,
  ArrowRight,
  CheckCircle2,
  AlertCircle
} from "lucide-react";
import { loginApi, signupApi, errorMessage } from "../api";
import type { User } from "../types";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (user: User, message: string) => void;
}

export function AuthModal({ isOpen, onClose, onSuccess }: Props) {
  const [tab, setTab] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  if (!isOpen) return null;

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      if (tab === "login") {
        if (!email.trim() || !password) {
          throw new Error("Please provide both email and password.");
        }
        const data = await loginApi(email.trim(), password);
        onSuccess(data.user, `Welcome back, ${data.user.full_name}! Your private database is ready.`);
        onClose();
      } else {
        if (!fullName.trim()) {
          throw new Error("Please provide your full name.");
        }
        if (!email.trim() || !password) {
          throw new Error("Please provide email and password.");
        }
        if (password.length < 6) {
          throw new Error("Password must be at least 6 characters long.");
        }
        const data = await signupApi(email.trim(), password, fullName.trim(), "user");
        onSuccess(data.user, `Private database provisioned! Welcome, ${data.user.full_name}.`);
        onClose();
      }
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  function fillDemoAdmin() {
    setEmail("admin@research.com");
    setPassword("Admin@1234");
    setTab("login");
    setError("");
  }

  return (
    <div className="auth-modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="auth-modal-card" onClick={(e) => e.stopPropagation()}>
        <button className="auth-modal-close" onClick={onClose} aria-label="Close modal">
          <X size={18} />
        </button>

        <div className="auth-modal-header">
          <div className="auth-modal-badge">
            <ShieldCheck size={16} />
            <span>Core JWT Authentication & RBAC</span>
          </div>
          <h2>{tab === "login" ? "Sign in to Research Assistant" : "Create your account"}</h2>
          <p>
            {tab === "login"
              ? "Access your coordinated research graph, saved documents, and session telemetry."
              : "Set up your multi-agent workspace with end-to-end evidence tracking."}
          </p>
        </div>

        <div className="auth-tabs">
          <button
            type="button"
            className={`auth-tab ${tab === "login" ? "active" : ""}`}
            onClick={() => { setTab("login"); setError(""); }}
          >
            Sign in
          </button>
          <button
            type="button"
            className={`auth-tab ${tab === "signup" ? "active" : ""}`}
            onClick={() => { setTab("signup"); setError(""); }}
          >
            Create account
          </button>
        </div>

        {error && (
          <div className="auth-error-banner">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form">
          {tab === "signup" && (
            <div className="auth-field">
              <label htmlFor="auth-name">Full name</label>
              <div className="auth-input-wrapper">
                <UserIcon size={16} className="auth-input-icon" />
                <input
                  id="auth-name"
                  type="text"
                  placeholder="e.g. Dr. Jane Doe"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  disabled={loading}
                  required
                />
              </div>
            </div>
          )}

          <div className="auth-field">
            <label htmlFor="auth-email">Email address</label>
            <div className="auth-input-wrapper">
              <Mail size={16} className="auth-input-icon" />
              <input
                id="auth-email"
                type="email"
                placeholder="you@domain.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={loading}
                required
              />
            </div>
          </div>

          <div className="auth-field">
            <label htmlFor="auth-password">Password</label>
            <div className="auth-input-wrapper">
              <Lock size={16} className="auth-input-icon" />
              <input
                id="auth-password"
                type={showPassword ? "text" : "password"}
                placeholder={tab === "signup" ? "At least 6 characters" : "Enter password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={loading}
                required
              />
              <button
                type="button"
                className="auth-eye-btn"
                onClick={() => setShowPassword(!showPassword)}
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {tab === "signup" && (
            <div className="auth-field">
              <label>Private Workspace Provisioning</label>
              <div className="auth-role-note">
                <CheckCircle2 size={15} />
                <span>Creates a dedicated personal MongoDB database for your chats, uploaded files &amp; persistent memory (standard Researcher role).</span>
              </div>
            </div>
          )}

          <button type="submit" className="auth-submit-btn" disabled={loading}>
            {loading ? (
              <span className="auth-spinner" />
            ) : (
              <>
                <span>{tab === "login" ? "Sign in" : "Create account"}</span>
                <ArrowRight size={16} />
              </>
            )}
          </button>
        </form>

        <div className="auth-modal-footer">
          <div className="auth-quick-actions">
            <span>Quick fill:</span>
            <button type="button" className="auth-quick-btn" onClick={fillDemoAdmin}>
              <Sparkles size={13} />
              <span>Demo Admin (admin@research.com)</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
