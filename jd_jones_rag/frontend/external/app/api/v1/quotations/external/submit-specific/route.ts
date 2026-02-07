import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://api:8000';

    try {
        const body = await request.json();

        const response = await fetch(`${API_URL}/v1/quotations/external/submit-specific`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        const data = await response.json();

        if (!response.ok) {
            return NextResponse.json(
                { error: data.detail || 'Failed to submit quotation' },
                { status: response.status }
            );
        }

        return NextResponse.json(data);
    } catch (error) {
        console.error('API proxy error:', error);
        return NextResponse.json(
            { error: 'Failed to connect to API' },
            { status: 500 }
        );
    }
}
