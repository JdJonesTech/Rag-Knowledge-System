import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET(
    request: NextRequest,
    { params }: { params: { docId: string } }
) {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://api:8000';

    try {
        const response = await fetch(`${API_URL}/documents/download/${params.docId}`, {
            cache: 'no-store',
        });

        if (!response.ok) {
            return NextResponse.json(
                { detail: 'Document not found' },
                { status: response.status }
            );
        }

        // Stream the file response back
        const contentType = response.headers.get('content-type') || 'application/octet-stream';
        const contentDisposition = response.headers.get('content-disposition') || '';
        const blob = await response.blob();

        return new NextResponse(blob, {
            headers: {
                'Content-Type': contentType,
                ...(contentDisposition && { 'Content-Disposition': contentDisposition }),
            },
        });
    } catch (error) {
        console.error('Document download proxy error:', error);
        return NextResponse.json(
            { detail: 'Failed to connect to API' },
            { status: 500 }
        );
    }
}
