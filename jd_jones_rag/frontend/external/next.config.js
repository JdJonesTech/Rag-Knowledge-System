/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  
  // API rewrites for development - proxies /api/* to backend
  async rewrites() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/:path*`,
      },
    ];
  },
  
  // Environment variables
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  
  // Output configuration for Docker
  output: 'standalone',
};

module.exports = nextConfig;
