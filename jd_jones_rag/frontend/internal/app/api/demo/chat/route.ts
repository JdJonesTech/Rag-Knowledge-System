import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://api:8000';

    try {
        const body = await request.json();

        const response = await fetch(`${API_URL}/demo/chat`, {
            method: 'POST',
            cache: 'no-store',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
            return NextResponse.json(
                { error: errorData.detail || 'Failed to chat' },
                { status: response.status }
            );
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Chat proxy error:', error);
        return NextResponse.json(
            { error: 'Failed to connect to API' },
            { status: 500 }
        );
    }
}
