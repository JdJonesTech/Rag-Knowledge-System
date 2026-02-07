import { NextRequest, NextResponse } from 'next/server';

export async function POST(
    request: NextRequest,
    context: { params: Promise<{ id: string }> }
) {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://api:8000';
    const { id: enquiryId } = await context.params;

    try {
        const response = await fetch(`${API_URL}/demo/enquiries/${enquiryId}/mark-sent`, {
            method: 'POST',
            cache: 'no-store',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
            return NextResponse.json(
                { success: false, error: errorData.detail || 'Failed to mark as sent' },
                { status: response.status }
            );
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Mark sent proxy error:', error);
        return NextResponse.json(
            { success: false, error: 'Failed to connect to API' },
            { status: 500 }
        );
    }
}
