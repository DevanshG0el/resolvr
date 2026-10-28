import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { HeaderAurora } from "@/components/GradientMotion";

export default function NotFound() {
  return (
    <main className="relative isolate grid min-h-[80dvh] place-items-center overflow-hidden px-4 pt-24 text-center">
      <HeaderAurora />
      <div>
        <p className="text-sm uppercase tracking-[0.2em] text-primary/80">404</p>
        <h1 className="glow-text mt-3 text-4xl font-medium tracking-tight sm:text-5xl">No record matches this page</h1>
        <p className="mx-auto mt-3 max-w-md text-fg/75">The address may be mistyped, or the page has moved.</p>
        <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <Link href="/" className="btn-solid w-full sm:w-auto">Back to home</Link>
          <Link href="/match" className="btn-ghost w-full sm:w-auto">Try live matching <ArrowRight size={16} /></Link>
        </div>
      </div>
    </main>
  );
}
