/* Hero illustration: records from Source 2 and Source 3 float as glass cards above glowing pillars,
   the coral platform in the middle is the Source 1 "core" they resolve into. Pure SVG, no assets. */

type CylProps = { x: number; y: number; w: number; h: number; body: string; top: string; rim?: string };

function Cylinder({ x, y, w, h, body, top, rim }: CylProps) {
  const ry = w * 0.13;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} fill={body} />
      <ellipse cx={x + w / 2} cy={y + h} rx={w / 2} ry={ry} fill={body} />
      <ellipse cx={x + w / 2} cy={y} rx={w / 2} ry={ry} fill={top} />
      {rim && <ellipse cx={x + w / 2} cy={y} rx={w / 2 - 10} ry={ry - 4} fill={rim} />}
    </g>
  );
}

function Card({ x, y, r, src, line1, line2, accent }: { x: number; y: number; r: number; src: string; line1: string; line2: string; accent: string }) {
  return (
    <g className="float" style={{ ["--r" as string]: `${r}deg` }}>
      <g transform={`translate(${x} ${y}) rotate(${r})`}>
        <rect width="148" height="96" rx="14" fill="url(#cardFill)" stroke="rgba(255,255,255,.55)" strokeWidth="1.2" />
        <rect x="12" y="12" width="34" height="18" rx="9" fill={accent} opacity=".9" />
        <text x="29" y="25" textAnchor="middle" fontSize="11" fontWeight="600" fill="#0b0730">{src}</text>
        <text x="12" y="52" fontSize="12.5" fontWeight="600" fill="#fff">{line1}</text>
        <text x="12" y="70" fontSize="10.5" fill="rgba(255,255,255,.7)">{line2}</text>
        <rect x="12" y="80" width="90" height="4" rx="2" fill="rgba(255,255,255,.35)" />
        <ellipse cx="74" cy="48" rx="86" ry="20" fill="none" stroke="rgba(255,255,255,.45)" strokeWidth="1" transform="rotate(-14 74 48)" />
      </g>
    </g>
  );
}

const STARS = [[120, 200], [300, 330], [330, 380], [520, 170], [690, 120], [870, 300], [905, 345], [1060, 190], [770, 260], [450, 300]];

export default function Hero3D({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 1200 560" className={className} preserveAspectRatio="xMidYMax slice" aria-hidden>
      <defs>
        <linearGradient id="violetBody" x1="0" x2="1">
          <stop offset="0" stopColor="#b3a7f7" />
          <stop offset=".55" stopColor="#7d6ef0" />
          <stop offset="1" stopColor="#3b2aa8" />
        </linearGradient>
        <linearGradient id="blueBody" x1="0" x2="1">
          <stop offset="0" stopColor="#3657f5" />
          <stop offset=".6" stopColor="#2a3fd6" />
          <stop offset="1" stopColor="#1a1f8f" />
        </linearGradient>
        <linearGradient id="coralBody" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#ff8fc3" />
          <stop offset=".5" stopColor="#ff6b8a" />
          <stop offset="1" stopColor="#f2475a" />
        </linearGradient>
        <linearGradient id="coralTop" x1="0" x2="1">
          <stop offset="0" stopColor="#ffb3d6" />
          <stop offset="1" stopColor="#ff7a93" />
        </linearGradient>
        <linearGradient id="indigoBody" x1="0" x2="1">
          <stop offset="0" stopColor="#5b4bd8" />
          <stop offset="1" stopColor="#24157a" />
        </linearGradient>
        <linearGradient id="beamG" x1="0" x2="0" y1="1" y2="0">
          <stop offset="0" stopColor="#fff" stopOpacity=".95" />
          <stop offset="1" stopColor="#fff" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="cardFill" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#ffffff" stopOpacity=".38" />
          <stop offset="1" stopColor="#a99ff8" stopOpacity=".18" />
        </linearGradient>
        <radialGradient id="dome" cx=".4" cy=".3">
          <stop offset="0" stopColor="#fff" />
          <stop offset="1" stopColor="#f6a8d4" />
        </radialGradient>
        <linearGradient id="fade" x1="0" x2="0" y1="0" y2="1">
          <stop offset=".75" stopColor="#1a0f6e" stopOpacity="0" />
          <stop offset="1" stopColor="#1a0f6e" stopOpacity=".9" />
        </linearGradient>
      </defs>

      {STARS.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={i % 3 ? 2.2 : 3.4} fill="#fff" className="twinkle" style={{ animationDelay: `${i * 0.4}s` }} />
      ))}

      {/* left: Source 2 pillar */}
      <polygon points="120,150 300,150 245,262 175,262" fill="url(#beamG)" className="beam" opacity=".7" />
      <Cylinder x={160} y={262} w={100} h={300} body="url(#violetBody)" top="#e9e4ff" rim="#ffffff" />
      <rect x={120} y={300} width={175} height={280} fill="url(#violetBody)" opacity=".92" />
      <ellipse cx={207} cy={300} rx={88} ry={13} fill="#d9d2ff" />
      <ellipse cx={240} cy={390} rx={26} ry={16} fill="#24157a" transform="rotate(-25 240 390)" />
      <path d="M140 470 q30 -60 70 -55 q40 5 20 -60" stroke="#fff" strokeOpacity=".7" fill="none" strokeWidth="1.3" />
      <circle cx={210} cy={415} r={4} fill="#fff" />
      <Card x={140} y={118} r={-10} src="S2" line1="Sharma Traders Pvt" line2="MG Rd, near SBI ATM" accent="#c9c4f5" />

      {/* small left drum */}
      <Cylinder x={260} y={445} w={130} h={140} body="url(#blueBody)" top="#2a3fd6" rim="#14106b" />
      <circle cx={300} cy={452} r={4} fill="#fff" />
      <circle cx={335} cy={458} r={4} fill="#fff" />

      {/* centre: Source 1 core */}
      <Cylinder x={400} y={455} w={400} h={140} body="url(#coralBody)" top="#ff9ab5" />
      <Cylinder x={440} y={385} w={320} h={80} body="url(#coralBody)" top="url(#coralTop)" />
      <g stroke="#fff" strokeOpacity=".75" strokeWidth="1.2" fill="none">
        <path d="M600 380 L470 400 M600 380 L520 418 M600 380 L600 428 M600 380 L680 418 M600 380 L730 400" />
        <path d="M520 418 L520 470 M600 428 L600 480 M680 418 L680 470" />
      </g>
      <path d="M548 384 a52 40 0 0 1 104 0 z" fill="url(#dome)" />
      <circle cx={470} cy={400} r={3.5} fill="#fff" />
      <circle cx={730} cy={400} r={3.5} fill="#fff" />
      <ellipse cx={720} cy={520} rx={30} ry={18} fill="#3b1a8f" transform="rotate(-20 720 520)" />
      <text x="600" y="520" textAnchor="middle" fontSize="15" fontWeight="600" fill="#fff" opacity=".9" letterSpacing="3">SOURCE 1</text>

      {/* right: Source 3 pillar */}
      <polygon points="905,120 1095,120 1035,230 965,230" fill="url(#beamG)" className="beam" style={{ animationDelay: "1.5s" }} opacity=".7" />
      <Cylinder x={950} y={230} w={100} h={360} body="url(#blueBody)" top="#dfe6ff" rim="#ffffff" />
      <rect x={910} y={275} width={185} height={300} fill="url(#blueBody)" />
      <ellipse cx={1002} cy={275} rx={93} ry={13} fill="#4a66ff" />
      <ellipse cx={955} cy={370} rx={28} ry={15} fill="#14106b" transform="rotate(20 955 370)" />
      <path d="M1000 330 q60 10 40 90 q-20 60 30 100" stroke="#fff" strokeOpacity=".7" fill="none" strokeWidth="1.3" />
      <circle cx={1012} cy={330} r={4} fill="#fff" />
      <Card x={925} y={88} r={9} src="S3" line1="शर्मा ट्रेडर्स" line2="M.G. Road, Pune 411001" accent="#9fb0ff" />

      {/* small right drum */}
      <Cylinder x={810} y={470} w={110} h={120} body="url(#indigoBody)" top="#3a2aa8" rim="#1a0f6e" />
      <circle cx={850} cy={476} r={4} fill="#fff" />
      <circle cx={885} cy={480} r={4} fill="#fff" />

      <rect width="1200" height="560" fill="url(#fade)" />
    </svg>
  );
}
