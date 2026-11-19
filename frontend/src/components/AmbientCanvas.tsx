import React, { useEffect, useRef } from 'react';
import { COLOR_PALETTES, type ColorPalette } from '../palettes';

interface AmbientCanvasProps {
  theme: 'dark' | 'light';
  palette: ColorPalette;
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  alpha: number;
  tint: 0 | 1 | 2;
}

/**
 * Lightweight interactive ambient background:
 * 1. Animated GPU-composited aurora mesh blobs
 * 2. Smoothly lerped cursor-reactive radial spotlight
 * 3. Subtle floating micro-constellation particles (<canvas>)
 * Automatically pauses on hidden tabs and respects prefers-reduced-motion.
 */
export const AmbientCanvas: React.FC<AmbientCanvasProps> = ({ theme, palette }) => {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const themeRef = useRef<'dark' | 'light'>(theme);
  const paletteRef = useRef<ColorPalette>(palette);

  useEffect(() => {
    themeRef.current = theme;
  }, [theme]);

  useEffect(() => {
    paletteRef.current = palette;
  }, [palette]);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    const canvas = canvasRef.current;
    if (!wrapper || !canvas) return;

    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    let reducedMotion = motionQuery.matches;

    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    let width = window.innerWidth;
    let height = window.innerHeight;
    let dpr = Math.min(window.devicePixelRatio || 1, 1.5);

    const pointer = {
      tx: width * 0.62,
      ty: height * 0.24,
      cx: width * 0.62,
      cy: height * 0.24,
      active: false,
    };

    const particles: Particle[] = [];

    function seedParticles() {
      particles.length = 0;
      const count = width < 640 ? 14 : width < 1100 ? 22 : 30;
      for (let i = 0; i < count; i++) {
        particles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.24,
          vy: (Math.random() - 0.5) * 0.24,
          radius: 1.1 + Math.random() * 1.5,
          alpha: 0.16 + Math.random() * 0.28,
          tint: (i % 3) as 0 | 1 | 2,
        });
      }
    }

    function resize() {
      width = window.innerWidth;
      height = window.innerHeight;
      dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      canvas!.width = Math.floor(width * dpr);
      canvas!.height = Math.floor(height * dpr);
      canvas!.style.width = `${width}px`;
      canvas!.style.height = `${height}px`;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      seedParticles();
    }

    resize();

    const onPointerMove = (e: PointerEvent) => {
      pointer.tx = e.clientX;
      pointer.ty = e.clientY;
      pointer.active = true;
    };

    const onPointerLeave = () => {
      pointer.active = false;
    };

    const onMotionChange = (e: MediaQueryListEvent) => {
      reducedMotion = e.matches;
      if (reducedMotion) {
        ctx.clearRect(0, 0, width, height);
      }
    };

    window.addEventListener('resize', resize, { passive: true });
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    document.addEventListener('mouseleave', onPointerLeave, { passive: true });
    motionQuery.addEventListener?.('change', onMotionChange);

    let rafId = 0;

    const tick = () => {
      rafId = window.requestAnimationFrame(tick);
      if (document.hidden || reducedMotion) return;

      // Smooth exponential lerp for cursor spotlight
      pointer.cx += (pointer.tx - pointer.cx) * 0.065;
      pointer.cy += (pointer.ty - pointer.cy) * 0.065;
      wrapper.style.setProperty('--mx', `${pointer.cx.toFixed(1)}px`);
      wrapper.style.setProperty('--my', `${pointer.cy.toFixed(1)}px`);

      ctx.clearRect(0, 0, width, height);

      const isLight = themeRef.current === 'light';
      const activeOption =
        COLOR_PALETTES.find((item) => item.id === paletteRef.current) || COLOR_PALETTES[0];
      const rgbTriad = isLight ? activeOption.lightParticles : activeOption.darkParticles;
      const linkColor = rgbTriad[0];
      const baseAlphaScale = isLight ? 0.65 : 0.9;

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;

        // Subtle interactive drift away from/around pointer
        if (pointer.active) {
          const dx = p.x - pointer.cx;
          const dy = p.y - pointer.cy;
          const distSq = dx * dx + dy * dy;
          if (distSq > 4 && distSq < 25600) {
            const dist = Math.sqrt(distSq);
            const push = (1 - dist / 160) * 0.045;
            p.x += (dx / dist) * push;
            p.y += (dy / dist) * push;
          }
        }

        if (p.x < -12) p.x = width + 12;
        if (p.x > width + 12) p.x = -12;
        if (p.y < -12) p.y = height + 12;
        if (p.y > height + 12) p.y = -12;

        // Draw node
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${rgbTriad[p.tint]}, ${(p.alpha * baseAlphaScale).toFixed(3)})`;
        ctx.fill();

        // Delicate nearby constellation filaments
        for (let j = i + 1; j < particles.length; j++) {
          const q = particles[j];
          const dx = p.x - q.x;
          const dy = p.y - q.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < 16900) {
            const dist = Math.sqrt(d2);
            const lineAlpha = (1 - dist / 130) * (isLight ? 0.08 : 0.11);
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(q.x, q.y);
            ctx.strokeStyle = `rgba(${linkColor}, ${lineAlpha.toFixed(3)})`;
            ctx.lineWidth = 0.75;
            ctx.stroke();
          }
        }

        // Whisper-soft cursor tether when close
        if (pointer.active) {
          const dx = p.x - pointer.cx;
          const dy = p.y - pointer.cy;
          const d2 = dx * dx + dy * dy;
          if (d2 < 22500) {
            const dist = Math.sqrt(d2);
            const lineAlpha = (1 - dist / 150) * (isLight ? 0.11 : 0.15);
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(pointer.cx, pointer.cy);
            ctx.strokeStyle = `rgba(${rgbTriad[p.tint]}, ${lineAlpha.toFixed(3)})`;
            ctx.lineWidth = 0.7;
            ctx.stroke();
          }
        }
      }
    };

    if (!reducedMotion) {
      rafId = window.requestAnimationFrame(tick);
    }

    return () => {
      window.cancelAnimationFrame(rafId);
      window.removeEventListener('resize', resize);
      window.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('mouseleave', onPointerLeave);
      motionQuery.removeEventListener?.('change', onMotionChange);
    };
  }, []);

  return (
    <div ref={wrapperRef} className="ambient-backdrop" aria-hidden="true">
      <div className="aurora-orb aurora-orb-one" />
      <div className="aurora-orb aurora-orb-two" />
      <div className="aurora-orb aurora-orb-three" />
      <div className="ambient-spotlight" />
      <div className="ambient-grid-veil" />
      <canvas ref={canvasRef} className="ambient-particles" />
    </div>
  );
};
