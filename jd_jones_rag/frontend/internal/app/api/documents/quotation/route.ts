import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://api:8000';

    try {
        const body = await request.json();

        const response = await fetch(`${API_URL}/documents/quotation`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        const data = await response.json();

        if (!response.ok) {
            return NextResponse.json(
                data,
                { status: response.status }
            );
        }

        return NextResponse.json(data);
    } catch (error) {
        console.error('Documents quotation proxy error:', error);
        return NextResponse.json(
            { detail: 'Failed to connect to API' },
            { status: 500 }
        );
    }
}
