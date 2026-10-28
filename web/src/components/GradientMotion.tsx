"use client";
/* Animated "aurora" background: soft blurred colour blobs drifting over a base gradient, with optional film grain.
   React port of the Framer GradientMotionBackground idea, without the Framer runtime. The composition is seeded, so
   server and client render the same first frame; motion runs in requestAnimationFrame and writes transforms straight
   to the DOM (no re-render per frame). It pauses off-screen and in hidden tabs, and stays still for reduced motion. */
import { useEffect, useMemo, useRef } from "react";

type Props = {
  colors: string[];
  base?: string; // any CSS background, drawn under the blobs
  blobs?: number;
  blur?: number; // px
  sizeMin?: number; // % of the container
  sizeMax?: number;
  speed?: number; // 100 = one 6 s loop
  range?: number; // px of drift
  seed?: number;
  opacity?: number; // 0..1
  blend?: React.CSSProperties["mixBlendMode"];
  grain?: number; // 0..1, 0 = off
  className?: string;
};

// mulberry32: small deterministic PRNG, so a seed always gives the same layout
function prng(seed: number) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const GRAIN = `url("data:image/svg+xml;utf8,${encodeURIComponent(
  "<svg xmlns='http://www.w3.org/2000/svg' width='140' height='140'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 1 0'/></filter><rect width='100%' height='100%' filter='url(#n)'/></svg>",
)}")`;
const SHAPES = ["50% 45% 55% 50% / 50% 55% 45% 50%", "60% 40% 30% 70% / 60% 30% 70% 40%", "45% 55% 50% 50% / 55% 45% 55% 45%"];

export default function GradientMotion({
  colors, base = "transparent", blobs = 4, blur = 110, sizeMin = 45, sizeMax = 75, speed = 40, range = 70,
  seed = 7, opacity = 1, blend = "screen", grain = 0, className = "",
}: Props) {
  const root = useRef<HTMLDivElement>(null);
  const els = useRef<(HTMLDivElement | null)[]>([]);

  const layout = useMemo(() => {
    const r = prng(Math.floor(seed * 1e5) + 1);
    return Array.from({ length: Math.max(1, Math.min(10, blobs)) }, (_, i) => {
      const angle = r() * Math.PI * 2;
      return {
        x: Math.min(100, Math.max(0, ((i + 0.5) / blobs) * 100 + (r() - 0.5) * 30)),
        y: 30 + r() * 40,
        size: sizeMin + r() * (sizeMax - sizeMin),
        color: colors[i % colors.length],
        rot: Math.round(r() * 360),
        phase: r(),
        dx: Math.cos(angle),
        dy: Math.sin(angle),
        shape: SHAPES[i % SHAPES.length],
      };
    });
  }, [seed, blobs, sizeMin, sizeMax, colors]);

  useEffect(() => {
    const node = root.current;
    if (!node || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    let raf = 0;
    let visible = true;
    let start: number | null = null;
    const frame = (t: number) => {
      start ??= t;
      const tick = ((t - start) / 1000) * (speed / 100);
      layout.forEach((b, i) => {
        const el = els.current[i];
        if (!el) return;
        // looping ease-in-out drift along the blob's own direction, phase-shifted per blob
        const p = (Math.sin((((tick + b.phase * 6) % 6) / 6 - 0.25) * Math.PI * 2) + 1) / 2;
        const s = Math.sin(p * p * (3 - 2 * p) * Math.PI * 2);
        el.style.transform = `translate(-50%, -50%) translate(${(b.dx * range * s).toFixed(1)}px, ${(b.dy * range * s).toFixed(1)}px) rotate(${b.rot}deg)`;
      });
      raf = requestAnimationFrame(frame);
    };
    const play = () => { if (!raf && visible && !document.hidden) raf = requestAnimationFrame(frame); };
    const pause = () => { cancelAnimationFrame(raf); raf = 0; };
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) play(); else pause(); });
    io.observe(node);
    const onVis = () => (document.hidden ? pause() : play());
    document.addEventListener("visibilitychange", onVis);
    play();
    return () => { pause(); io.disconnect(); document.removeEventListener("visibilitychange", onVis); };
  }, [layout, speed, range]);

  return (
    <div ref={root} aria-hidden className={`pointer-events-none absolute inset-0 overflow-hidden ${className}`} style={{ background: base }}>
      <div className="absolute inset-0" style={{ opacity }}>
        {layout.map((b, i) => (
          <div key={i} ref={(el) => { els.current[i] = el; }}
            style={{
              position: "absolute", left: `${b.x}%`, top: `${b.y}%`, width: `${b.size}%`, height: `${b.size}%`,
              background: b.color, borderRadius: b.shape, filter: `blur(${blur}px)`, mixBlendMode: blend,
              transform: `translate(-50%, -50%) rotate(${b.rot}deg)`, willChange: "transform",
            }} />
        ))}
      </div>
      {grain > 0 && (
        <div className="absolute inset-0" style={{ backgroundImage: GRAIN, backgroundSize: "140px 140px", opacity: grain, mixBlendMode: "overlay" }} />
      )}
    </div>
  );
}

// site palette: indigo / violet / cobalt with a touch of coral, over the black-to-indigo night sky
const SKY_BASE = "linear-gradient(180deg, #030112 0%, #0b0730 45%, #1a0f6e 100%)";
const AURORA = ["#4c2edc", "#2f4bf0", "#7b5cf0", "#b8336f", "#3a1fb8"];
const AURORA_SOFT = ["#4c2edc", "#2f4bf0", "#8b7cf6", "#c0397a"];

/* hero of the home page */
export function HeroAurora() {
  return <GradientMotion className="-z-10" base={SKY_BASE} colors={AURORA} blobs={5} blur={130} sizeMin={40} sizeMax={70} opacity={0.5} range={90} seed={3} grain={0.08} />;
}

/* inner-page header band: quieter, and it settles into the page colour at the bottom so overlapping cards sit on a clean edge */
export function HeaderAurora() {
  return (
    <>
      <GradientMotion className="-z-10" base={SKY_BASE} colors={AURORA_SOFT} blobs={4} blur={100} sizeMin={45} sizeMax={70} opacity={0.5} range={70} seed={11} grain={0.06} />
      <div aria-hidden className="pointer-events-none absolute inset-x-0 bottom-0 -z-10 h-2/5 bg-gradient-to-b from-transparent to-night" />
    </>
  );
}
