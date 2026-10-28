import { HeaderAurora } from "./GradientMotion";

export default function PageHead({ eyebrow, title, sub }: { eyebrow: string; title: string; sub: string }) {
  return (
    <div className="relative isolate overflow-hidden px-4 pb-16 pt-32 sm:px-8 sm:pt-36">
      <HeaderAurora />
      <div className="mx-auto max-w-6xl">
        <p className="text-sm uppercase tracking-[0.2em] text-mist/80">{eyebrow}</p>
        <h1 className="glow-text mt-3 text-4xl font-medium tracking-tight sm:text-5xl">{title}</h1>
        <p className="mt-3 max-w-2xl text-white/70">{sub}</p>
      </div>
    </div>
  );
}
