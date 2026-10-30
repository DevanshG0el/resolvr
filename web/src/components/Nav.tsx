"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Menu, X } from "lucide-react";
import Logo from "./Logo";

const LINKS = [
  { href: "/match", label: "Live match" },
  { href: "/explore", label: "Explore" },
  { href: "/data", label: "Dataset" },
  { href: "/models", label: "Models" },
  { href: "/progress", label: "Progress" },
  { href: "/pipeline", label: "Pipeline" },
];

export default function Nav() {
  const path = usePathname();
  const [open, setOpen] = useState(false);

  // close the phone menu on navigation (including back/forward) and on Escape
  useEffect(() => setOpen(false), [path]);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <header className="absolute inset-x-0 top-0 z-30">
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-4 py-5 sm:px-8" aria-label="Main">
        <Link href="/" className="font-display flex items-center gap-2.5 text-xl font-semibold tracking-tight">
          <Logo /> Resolvr
        </Link>
        <ul className="hidden items-center gap-6 text-sm md:flex lg:gap-9">
          {LINKS.map((l) => (
            <li key={l.href}>
              <Link href={l.href} aria-current={path === l.href ? "page" : undefined}
                className={`relative py-1 transition hover:text-fg ${path === l.href
                  ? "text-fg after:absolute after:inset-x-0 after:-bottom-1 after:h-0.5 after:rounded-full after:bg-accent"
                  : "text-fg/75"}`}>
                {l.label}
              </Link>
            </li>
          ))}
        </ul>
        <div className="hidden items-center gap-6 md:flex">
          <a href="/api/health" target="_blank" rel="noreferrer" className="hidden text-sm text-fg/75 hover:text-fg lg:inline">API</a>
          <Link href="/match" className="rounded-xl border border-fg/80 px-5 py-2.5 text-sm transition hover:bg-fg/[0.06]">Get started</Link>
        </div>
        <button className="-mr-2 rounded-lg p-2 md:hidden" aria-label={open ? "Close menu" : "Open menu"} aria-expanded={open}
          aria-controls="mobile-menu" onClick={() => setOpen(!open)}>
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>
      </nav>
      {open && (
        <>
          <div className="fixed inset-0 -z-10 bg-page/60 md:hidden" onClick={() => setOpen(false)} aria-hidden />
          <div id="mobile-menu" className="mx-4 rounded-2xl border border-fg/15 bg-surface p-2 shadow-2xl shadow-primary/15 md:hidden">
            {LINKS.map((l) => (
              <Link key={l.href} href={l.href} aria-current={path === l.href ? "page" : undefined}
                className={`block rounded-xl px-4 py-3 hover:bg-fg/[0.06] ${path === l.href ? "bg-fg/[0.06] text-fg" : "text-fg/90"}`}>
                {l.label}
              </Link>
            ))}
            <a href="/api/health" target="_blank" rel="noreferrer" className="block rounded-xl px-4 py-3 text-fg/70 hover:bg-fg/[0.06]">API status</a>
          </div>
        </>
      )}
    </header>
  );
}
