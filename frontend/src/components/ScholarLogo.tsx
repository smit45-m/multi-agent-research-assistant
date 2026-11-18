import React, { useId } from 'react';

interface ScholarLogoProps {
  size?: number;
  className?: string;
  showOrbit?: boolean;
  decorative?: boolean;
}

/**
 * Bespoke ScholarAgent emblem: opposing precision orbital arcs with two active
 * agent constellation nodes framing a faceted four-point synthesis prism.
 */
export const ScholarLogo: React.FC<ScholarLogoProps> = ({
  size = 26,
  className = '',
  showOrbit = true,
  decorative = true,
}) => {
  const rawId = useId();
  const uid = rawId.replace(/[^a-zA-Z0-9_-]/g, '');

  const starGradId = `sa-star-${uid}`;
  const arcAGradId = `sa-arc-a-${uid}`;
  const arcBGradId = `sa-arc-b-${uid}`;
  const haloGradId = `sa-halo-${uid}`;

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 40 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={`sa-logo-svg ${className}`.trim()}
      aria-hidden={decorative ? 'true' : undefined}
      role={decorative ? undefined : 'img'}
      aria-label={decorative ? undefined : 'ScholarAgent logo'}
    >
      <defs>
        <linearGradient id={starGradId} x1="10" y1="10" x2="30" y2="30" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="var(--sa-star-start, #eef2ff)" />
          <stop offset="52%" stopColor="var(--sa-star-mid, #a5b4fc)" />
          <stop offset="100%" stopColor="var(--sa-star-end, #7c3aed)" />
        </linearGradient>

        <linearGradient id={arcAGradId} x1="6" y1="24" x2="26" y2="7" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="var(--sa-arc-indigo, #818cf8)" stopOpacity="0.45" />
          <stop offset="55%" stopColor="var(--sa-arc-violet, #c084fc)" stopOpacity="0.95" />
          <stop offset="100%" stopColor="var(--sa-arc-cyan, #38bdf8)" stopOpacity="1" />
        </linearGradient>

        <linearGradient id={arcBGradId} x1="34" y1="16" x2="14" y2="33" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="var(--sa-arc-cyan, #38bdf8)" stopOpacity="0.9" />
          <stop offset="55%" stopColor="var(--sa-arc-indigo, #818cf8)" stopOpacity="0.95" />
          <stop offset="100%" stopColor="var(--sa-arc-violet, #c084fc)" stopOpacity="0.4" />
        </linearGradient>

        <radialGradient id={haloGradId} cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="var(--sa-node-emerald, #34d399)" stopOpacity="0.5" />
          <stop offset="100%" stopColor="var(--sa-node-emerald, #34d399)" stopOpacity="0" />
        </radialGradient>
      </defs>

      {showOrbit && (
        <g className="sa-logo-orbit">
          <circle
            cx="20"
            cy="20"
            r="13.5"
            stroke="var(--sa-arc-indigo, #818cf8)"
            strokeOpacity="0.16"
            strokeWidth="0.9"
            strokeDasharray="1.8 2.8"
          />

          <path
            d="M 6.96 23.49 A 13.5 13.5 0 0 1 25.71 7.76"
            stroke={`url(#${arcAGradId})`}
            strokeWidth="2.1"
            strokeLinecap="round"
          />

          <path
            d="M 33.04 16.51 A 13.5 13.5 0 0 1 14.29 32.24"
            stroke={`url(#${arcBGradId})`}
            strokeWidth="2.1"
            strokeLinecap="round"
          />

          {/* Ascendant discovery satellite node */}
          <circle cx="30.57" cy="11.13" r="4.1" fill={`url(#${haloGradId})`} />
          <circle cx="30.57" cy="11.13" r="2.15" fill="var(--sa-node-emerald, #34d399)" />
          <circle cx="30.05" cy="10.6" r="0.65" fill="#ffffff" fillOpacity="0.85" />

          {/* Grounding corpus satellite node */}
          <circle cx="9.43" cy="28.87" r="1.85" fill="var(--sa-node-violet, #a78bfa)" />
        </g>
      )}

      <g className="sa-logo-core">
        {/* Four-point faceted synthesis prism */}
        <path
          d="M 20 10.2 C 20.85 16.2 23.8 19.15 29.8 20 C 23.8 20.85 20.85 23.8 20 29.8 C 19.15 23.8 16.2 20.85 10.2 20 C 16.2 19.15 19.15 16.2 20 10.2 Z"
          fill={`url(#${starGradId})`}
        />
        {/* Crisp upper-left crystalline bevel highlight */}
        <path
          d="M 20 10.2 L 20 20 L 10.2 20 C 16.2 19.15 19.15 16.2 20 10.2 Z"
          fill="#ffffff"
          fillOpacity="0.28"
        />
        {/* Inner precision diamond core */}
        <path
          d="M 20 16.9 L 23.1 20 L 20 23.1 L 16.9 20 Z"
          fill="var(--sa-prism-inner, #ffffff)"
          fillOpacity="0.72"
        />
      </g>
    </svg>
  );
};
