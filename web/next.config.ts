import type { NextConfig } from "next";

const API = process.env.BER_API ?? "http://127.0.0.1:7400";

const nextConfig: NextConfig = {
  // the browser only talks to Next; /api/* is proxied to the FastAPI service
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;
