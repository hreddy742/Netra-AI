import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  async rewrites() {
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'
    return [
      {
        source: '/api/auth/:path*',
        destination: `${backendUrl}/api/auth/:path*`,
      },
      {
        source: '/api/cameras/:path*',
        destination: `${backendUrl}/api/cameras/:path*`,
      },
      {
        source: '/api/cameras',
        destination: `${backendUrl}/api/cameras`,
      },
      {
        source: '/api/incidents/:path*',
        destination: `${backendUrl}/api/incidents/:path*`,
      },
      {
        source: '/api/incidents',
        destination: `${backendUrl}/api/incidents`,
      },
      {
        source: '/api/health',
        destination: `${backendUrl}/api/health`,
      },
      {
        source: '/api/stream/:path*',
        destination: `${backendUrl}/api/stream/:path*`,
      },
    ]
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
