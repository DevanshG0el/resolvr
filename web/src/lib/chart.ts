// Recharts styling for the light theme, shared by the Models and Progress pages
export const INK = "#15123a";
export const AXIS = { stroke: "rgba(21,18,58,.35)", fill: "rgba(21,18,58,.6)", fontSize: 12 };
export const CAT_TICK = { ...AXIS, fill: "rgba(21,18,58,.8)" }; // category labels next to bars
export const GRID = "rgba(21,18,58,.08)";
export const CURSOR = { fill: "rgba(91,75,216,.06)" };
export const BAR_LABEL = "rgba(21,18,58,.75)";
export const TIP = {
  contentStyle: { background: "#fff", border: "1px solid rgba(21,18,58,.1)", borderRadius: 12, boxShadow: "0 8px 24px -12px rgba(76,46,220,.3)" },
  labelStyle: { color: INK, fontWeight: 600 }, itemStyle: { color: "rgba(21,18,58,.8)" },
};
