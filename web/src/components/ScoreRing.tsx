export default function ScoreRing({ p, size = 56 }: { p: number; size?: number }) {
  const r = size / 2 - 5;
  const c = 2 * Math.PI * r;
  const color = p >= 0.5 ? "#ff5f7e" : "#8b7cf6";
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`match probability ${(p * 100).toFixed(0)} percent`}>
      <circle cx={size / 2} cy={size / 2} r={r} stroke="rgba(255,255,255,.12)" strokeWidth="5" fill="none" />
      <circle cx={size / 2} cy={size / 2} r={r} stroke={color} strokeWidth="5" fill="none" strokeLinecap="round"
        strokeDasharray={`${c * p} ${c}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} />
      <text x="50%" y="54%" textAnchor="middle" dominantBaseline="middle" fontSize={size * 0.26} fill="#fff" fontWeight="600" className="font-display tabular-nums">
        {(p * 100).toFixed(0)}
      </text>
    </svg>
  );
}
