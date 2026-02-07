import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://api:8000';

    try {
        const response = await fetch(`${API_URL}/demo/enquiries`, {
            cache: 'no-store',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            return NextResponse.json(
                { error: 'Failed to fetch enquiries' },
                { status: response.status }
            );
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('API proxy error:', error);
        return NextResponse.json(
            { error: 'Failed to connect to API' },
            { status: 500 }
        );
    }
}
