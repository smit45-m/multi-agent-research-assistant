import { useEffect } from 'react';

/**
 * Progressive scroll-triggered reveal hook.
 * Elements already inside the viewport animate cleanly via CSS entrance keyframes
 * and are never hidden. Elements strictly below the initial fold receive
 * `.reveal-pending` and smoothly reveal (`.is-visible`) via IntersectionObserver
 * with a scroll fallback so content can never remain stuck invisible.
 */
export function useScrollReveal(triggerKey: string | number): void {
  useEffect(() => {
    const prefersReduced =
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (prefersReduced || typeof IntersectionObserver === 'undefined') {
      document.querySelectorAll<HTMLElement>('[data-reveal]').forEach((el) => {
        el.classList.remove('reveal-pending');
        el.classList.add('is-visible');
      });
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.remove('reveal-pending');
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
          }
        }
      },
      { threshold: 0.02, rootMargin: '60px 0px 60px 0px' },
    );

    const scan = () => {
      const vh = window.innerHeight || 900;
      document.querySelectorAll<HTMLElement>('[data-reveal]:not(.is-visible)').forEach((el) => {
        const rect = el.getBoundingClientRect();
        if (rect.top <= vh + 40) {
          el.classList.remove('reveal-pending');
          el.classList.add('is-visible');
        } else {
          el.classList.add('reveal-pending');
          observer.observe(el);
        }
      });
    };

    scan();

    const onScroll = () => scan();
    window.addEventListener('scroll', onScroll, { passive: true });

    const mutationObserver = new MutationObserver(() => scan());
    const mainEl = document.getElementById('main-content') || document.body;
    mutationObserver.observe(mainEl, { childList: true, subtree: true });

    return () => {
      observer.disconnect();
      mutationObserver.disconnect();
      window.removeEventListener('scroll', onScroll);
    };
  }, [triggerKey]);
}
