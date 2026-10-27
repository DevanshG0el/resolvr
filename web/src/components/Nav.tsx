"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Menu, X } from "lucide-react";
import Logo from "./Logo";

const LINKS = [
  { href: "/match", label: "Live match" },
  { href: "/explore", label: "Explore" },
  { href: "/models", label: "Models" },
  { href: "/pipeline", label: "Pipeline" },
];

export default function Nav() {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  return (
    <header className="absolute inset-x-0 top-0 z-30">
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-4 py-5 sm:px-8">
        <Link href="/" className="flex items-center gap-2.5 text-lg font-semibold tracking-tight" onClick={() => setOpen(false)}>
          <Logo /> Resolvr
        </Link>
        <ul className="hidden items-center gap-9 text-sm md:flex">
          {LINKS.map((l) => (
            <li key={l.href}>
              <Link href={l.href} className={`transition hover:text-white ${path === l.href ? "text-white" : "text-white/70"}`}>
                {l.label}
              </Link>
            </li>
          ))}
        </ul>
        <div className="hidden items-center gap-6 md:flex">
          <a href="/api/health" className="text-sm text-white/70 hover:text-white">API</a>
          <Link href="/match" className="rounded-xl border border-white/80 px-5 py-2.5 text-sm hover:bg-white/10">Get started</Link>
        </div>
        <button className="rounded-lg p-2 md:hidden" aria-label="Menu" onClick={() => setOpen(!open)}>
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>
      </nav>
      {open && (
        <div className="glass mx-4 rounded-2xl p-2 md:hidden">
          {LINKS.map((l) => (
            <Link key={l.href} href={l.href} onClick={() => setOpen(false)}
              className="block rounded-xl px-4 py-3 text-white/85 hover:bg-white/10">
              {l.label}
            </Link>
          ))}
        </div>
      )}
    </header>
  );
}
