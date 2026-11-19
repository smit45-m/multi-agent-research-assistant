export type ColorPalette = 'aurora' | 'emerald' | 'amber' | 'rose' | 'ocean';

export interface PaletteOption {
  id: ColorPalette;
  name: string;
  subtitle: string;
  swatches: [string, string, string];
  darkParticles: [string, string, string];
  lightParticles: [string, string, string];
}

export const COLOR_PALETTES: PaletteOption[] = [
  {
    id: 'aurora',
    name: 'Aurora Indigo',
    subtitle: 'Indigo · Violet · Cyan',
    swatches: ['#818cf8', '#a78bfa', '#38bdf8'],
    darkParticles: ['129, 140, 248', '167, 139, 250', '56, 189, 248'],
    lightParticles: ['79, 70, 229', '124, 58, 237', '2, 132, 199'],
  },
  {
    id: 'emerald',
    name: 'Botanical Jade',
    subtitle: 'Emerald · Teal · Mint',
    swatches: ['#34d399', '#2dd4bf', '#38bdf8'],
    darkParticles: ['52, 211, 153', '45, 212, 191', '56, 189, 248'],
    lightParticles: ['5, 150, 105', '15, 118, 110', '2, 132, 199'],
  },
  {
    id: 'amber',
    name: 'Solar Amber',
    subtitle: 'Gold · Copper · Coral',
    swatches: ['#f59e0b', '#fb923c', '#fb7185'],
    darkParticles: ['245, 158, 11', '251, 146, 60', '251, 113, 133'],
    lightParticles: ['180, 83, 9', '194, 65, 12', '225, 29, 72'],
  },
  {
    id: 'rose',
    name: 'Velvet Crimson',
    subtitle: 'Rose · Orchid · Berry',
    swatches: ['#fb7185', '#e879f9', '#a78bfa'],
    darkParticles: ['251, 113, 133', '232, 121, 249', '167, 139, 250'],
    lightParticles: ['190, 18, 60', '162, 28, 175', '109, 40, 217'],
  },
  {
    id: 'ocean',
    name: 'Arctic Sapphire',
    subtitle: 'Sapphire · Cobalt · Aqua',
    swatches: ['#38bdf8', '#60a5fa', '#22d3ee'],
    darkParticles: ['56, 189, 248', '96, 165, 250', '34, 211, 238'],
    lightParticles: ['3, 105, 161', '29, 78, 216', '14, 116, 144'],
  },
];

export function isColorPalette(value: string | null): value is ColorPalette {
  return COLOR_PALETTES.some((p) => p.id === value);
}
