export default function Logo({ className = "h-7 w-9" }: { className?: string }) {
  // two record cards converging into one: the matching idea in a mark
  return (
    <svg viewBox="0 0 36 28" className={className} aria-hidden>
      <defs>
        <linearGradient id="lg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#8b7cf6" />
          <stop offset="1" stopColor="#e5486b" />
        </linearGradient>
      </defs>
      <rect x="2" y="6" width="14" height="17" rx="3" fill="#5b4bd8" opacity="0.28" transform="rotate(-12 9 14)" />
      <rect x="20" y="6" width="14" height="17" rx="3" fill="#5b4bd8" opacity="0.28" transform="rotate(12 27 14)" />
      <rect x="10" y="3" width="16" height="20" rx="3.5" fill="url(#lg)" />
      <path d="M14 13.5l3 3 5-6" stroke="#ffffff" strokeWidth="2.2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
