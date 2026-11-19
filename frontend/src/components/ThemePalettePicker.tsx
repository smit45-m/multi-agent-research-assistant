import React, { useEffect, useRef, useState } from 'react';
import { Check, Palette } from 'lucide-react';
import { COLOR_PALETTES, type ColorPalette } from '../palettes';

interface ThemePalettePickerProps {
  palette: ColorPalette;
  onSelect: (palette: ColorPalette) => void;
}

export const ThemePalettePicker: React.FC<ThemePalettePickerProps> = ({
  palette,
  onSelect,
}) => {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  const active = COLOR_PALETTES.find((p) => p.id === palette) || COLOR_PALETTES[0];

  return (
    <div className="palette-picker" ref={containerRef}>
      <button
        type="button"
        className={`palette-trigger ${open ? 'is-open' : ''}`}
        aria-label={`Color theme: ${active.name}. Change color palette`}
        aria-expanded={open}
        aria-haspopup="menu"
        title={`Color theme: ${active.name}`}
        onClick={() => setOpen((prev) => !prev)}
      >
        <Palette size={15} />
        <span
          className="palette-active-swatch"
          style={{
            background: `linear-gradient(135deg, ${active.swatches[0]}, ${active.swatches[1]}, ${active.swatches[2]})`,
          }}
        />
      </button>

      {open && (
        <div className="palette-popover" role="menu" aria-label="Color combinations">
          <div className="palette-popover-header">
            <span>Color Palette</span>
            <small>Works in Dark &amp; Light</small>
          </div>
          <div className="palette-options">
            {COLOR_PALETTES.map((option) => {
              const selected = option.id === palette;
              return (
                <button
                  key={option.id}
                  type="button"
                  role="menuitemradio"
                  aria-checked={selected}
                  className={`palette-option ${selected ? 'selected' : ''}`}
                  onClick={() => {
                    onSelect(option.id);
                    setOpen(false);
                  }}
                >
                  <span
                    className="palette-swatch-tri"
                    style={{
                      background: `linear-gradient(135deg, ${option.swatches[0]} 0%, ${option.swatches[1]} 52%, ${option.swatches[2]} 100%)`,
                    }}
                  />
                  <span className="palette-option-meta">
                    <strong>{option.name}</strong>
                    <small>{option.subtitle}</small>
                  </span>
                  {selected && <Check size={14} className="palette-check" />}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
