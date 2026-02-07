import { NextRequest, NextResponse } from 'next/server';

// Disable Next.js caching for this dynamic data route
export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET(request: NextRequest) {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://api:8000';

    try {
        const response = await fetch(`${API_URL}/demo/quotations`, {
            headers: {
                'Content-Type': 'application/json',
            },
            cache: 'no-store',  // Disable fetch caching
            next: { revalidate: 0 }  // Disable Next.js revalidation cache
        });

        if (!response.ok) {
            return NextResponse.json(
                { error: 'Failed to fetch quotations' },
                {
                    status: response.status,
                    headers: {
                        'Cache-Control': 'no-store, no-cache, must-revalidate',
                    }
                }
            );
        }

        const data = await response.json();
        return NextResponse.json(data, {
            headers: {
                'Cache-Control': 'no-store, no-cache, must-revalidate',
                'Pragma': 'no-cache',
            }
        });
    } catch (error) {
        console.error('API proxy error:', error);
        return NextResponse.json(
            { error: 'Failed to connect to API' },
            { status: 500 }
        );
    }
}
