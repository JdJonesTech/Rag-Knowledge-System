import { NextResponse } from 'next/server';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://api:8000';

export async function GET() {
    try {
        const response = await fetch(`${API_URL}/demo/products`, {
            cache: 'no-store',
            headers: { 'Content-Type': 'application/json' },
        });
        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Products API proxy error:', error);
        return NextResponse.json(
            { success: false, products: [], error: 'Failed to fetch products' },
            { status: 500 }
        );
    }
}
