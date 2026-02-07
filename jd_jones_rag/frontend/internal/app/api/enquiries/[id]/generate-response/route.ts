import { NextRequest, NextResponse } from 'next/server';

export async function POST(
    request: NextRequest,
    context: { params: Promise<{ id: string }> }
) {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://api:8000';
    const { id: enquiryId } = await context.params;

    try {
        const body = await request.json();

        const response = await fetch(`${API_URL}/demo/enquiries/${enquiryId}/generate-response`, {
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
                { success: false, error: errorData.detail || 'Failed to generate response' },
                { status: response.status }
            );
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Generate response proxy error:', error);
        return NextResponse.json(
            { success: false, error: 'Failed to connect to API' },
            { status: 500 }
        );
    }
}
