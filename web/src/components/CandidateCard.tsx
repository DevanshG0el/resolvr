"use client";
import { useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import { tidy, type Candidate } from "@/lib/api";
import ScoreRing from "./ScoreRing";

export default function CandidateCard({ c }: { c: Candidate }) {
  const [open, setOpen] = useState(c.matched);
  const maxImp = Math.max(...c.reasons.map((r) => Math.abs(r.impact)), 0.01);
  return (
    <div className={`glass rounded-2xl p-4 sm:p-5 ${c.matched ? "ring-1 ring-accent/60" : ""}`}>
      <button className="flex w-full items-center gap-4 text-left" onClick={() => setOpen(!open)} aria-expanded={open}>
        <ScoreRing p={c.p} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full bg-fg/[0.06] px-2 py-0.5 text-xs text-primary">{c.source}</span>
            <span className="font-mono text-xs text-fg/55">{c.id}</span>
            {c.matched && (
              <span className="inline-flex items-center gap-1 rounded-full bg-accent/20 px-2 py-0.5 text-xs text-accent">
                <Check size={12} /> match
              </span>
            )}
          </div>
          <div className="mt-1 truncate font-medium">{tidy(c.name) || <em className="text-fg/55">no name</em>}</div>
          <div className="truncate text-sm text-fg/65">{tidy(c.address) || "no address"}</div>
        </div>
        <ChevronDown size={18} className={`shrink-0 text-fg/60 transition ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="mt-4 border-t border-fg/10 pt-4">
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            {[["Name", c.name_sim / 100], ["Address", c.addr_sim / 100], ["Semantic", c.cos]].map(([k, v]) => (
              <div key={k as string} className="rounded-xl bg-fg/[0.04] p-2">
                <div className="text-fg/60">{k}</div>
                <div className="text-base font-semibold">{(v as number) < 0 ? "–" : `${((v as number) * 100).toFixed(0)}%`}</div>
              </div>
            ))}
          </div>
          <p className="mb-2 mt-4 text-xs uppercase tracking-wider text-fg/55">Why this score (SHAP)</p>
          <ul className="space-y-1.5">
            {c.reasons.map((r) => (
              <li key={r.feature} className="flex items-center gap-3 text-sm">
                <span className="w-40 shrink-0 truncate text-fg/80 sm:w-52">{r.label}</span>
                <div className="relative h-2 flex-1 rounded-full bg-fg/[0.04]">
                  <div className={`absolute top-0 h-2 rounded-full ${r.impact >= 0 ? "left-1/2 bg-accent" : "right-1/2 bg-primary"}`}
                    style={{ width: `${(Math.abs(r.impact) / maxImp) * 50}%` }} />
                </div>
                <span className={`w-12 text-right font-mono text-xs ${r.impact >= 0 ? "text-accent" : "text-primary"}`}>
                  {r.impact >= 0 ? "+" : ""}{r.impact.toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
