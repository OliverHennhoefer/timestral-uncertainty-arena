/** Seeded synthetic series for streaming benchmarks, scaled to the 0-100 drawing range. */

export interface Preset {
  id: string;
  label: string;
  /** Dropdown group; synthetic presets leave it unset. */
  group?: string;
  /** Provenance shown as the option's tooltip. */
  source?: string;
  /** Up to `n` points; finite datasets may return fewer, and the stream ends with them. */
  make(n: number, seed: number): number[];
}

function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 2 ** 32;
  };
}

function gauss(r: () => number) {
  return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
}

const clamp = (v: number) => Math.max(1, Math.min(99, v));

export const PRESETS: Preset[] = [
  {
    id: "sine",
    label: "Seasonal + noise",
    make: (n, seed) => {
      const r = rng(seed);
      return Array.from({ length: n }, (_, t) => clamp(50 + 22 * Math.sin((2 * Math.PI * t) / 48) + 3 * gauss(r)));
    },
  },
  {
    id: "trend",
    label: "Trend + two seasons",
    make: (n, seed) => {
      const r = rng(seed);
      return Array.from({ length: n }, (_, t) => clamp(20 + (50 * t) / n + 10 * Math.sin((2 * Math.PI * t) / 24) + 6 * Math.sin((2 * Math.PI * t) / 96) + 2 * gauss(r)));
    },
  },
  {
    id: "volshift",
    label: "Volatility shift (calm → 4× noise)",
    make: (n, seed) => {
      const r = rng(seed);
      return Array.from({ length: n }, (_, t) => clamp(50 + 15 * Math.sin((2 * Math.PI * t) / 40) + (t < n / 2 ? 1.5 : 6) * gauss(r)));
    },
  },
  {
    id: "levels",
    label: "Level shifts",
    make: (n, seed) => {
      const r = rng(seed);
      let level = 40;
      return Array.from({ length: n }, (_, t) => {
        if (t > 0 && t % Math.floor(n / 5) === 0) level = 25 + 50 * r();
        return clamp(level + 2.5 * gauss(r));
      });
    },
  },
  {
    id: "walk",
    label: "Random walk",
    make: (n, seed) => {
      const r = rng(seed);
      let v = 50;
      return Array.from({ length: n }, () => (v = clamp(v + 1.5 * gauss(r))));
    },
  },
  {
    id: "heavy",
    label: "Heavy-tailed noise (Student t₂)",
    make: (n, seed) => {
      const r = rng(seed);
      return Array.from({ length: n }, (_, t) => {
        const z = gauss(r);
        const chi = (gauss(r) ** 2 + gauss(r) ** 2) / 2;
        return clamp(50 + 12 * Math.sin((2 * Math.PI * t) / 30) + (2.5 * z) / Math.sqrt(chi + 1e-9));
      });
    },
  },
  {
    id: "regime",
    label: "Regime change (period 60 → 20)",
    make: (n, seed) => {
      const r = rng(seed);
      let phase = 0;
      return Array.from({ length: n }, (_, t) => {
        phase += (2 * Math.PI) / (t < n / 2 ? 60 : 20);
        return clamp(50 + 25 * Math.sin(phase) + 2 * gauss(r));
      });
    },
  },
];
