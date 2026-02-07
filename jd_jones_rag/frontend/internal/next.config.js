/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Environment variables
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://api:8000',
  },

  // Output configuration for Docker
  output: 'standalone',
};

module.exports = nextConfig;


