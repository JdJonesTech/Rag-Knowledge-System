'use client';

import { useState, useEffect } from 'react';

interface LineItem {
    id: string;
    product_code: string;
    product_name: string;
    size?: string;
    size_od?: number | string;
    size_id?: number | string;
    size_th?: number | string;
    dimension_unit?: string;  // "mm" or "inch"
    style?: string;
    dimensions?: {
        od?: string;
        id?: string;
        th?: string;
        standard_sizes?: Array<{ size: string; suitable_for: string }>;
    };
    material_grade?: string;
    material_code?: string;
    colour?: string;
    quantity: number;
    unit?: string;
    rings_per_set?: number | string;
    specific_requirements?: string;
    is_ai_suggested?: boolean;
    ai_confidence?: number;
    unit_price?: number;
    total_price?: number;
    notes?: string;
}

interface Quotation {
    id: string;
    quotation_number: string;
    customer: {
        name: string;
        company?: string;
        email: string;
    };
    status: string;
    priority: string;
    created_at: string;
    assigned_to?: string;
    line_items: LineItem[];
    is_generic?: boolean;  // True if this was a generic quotation (AI filled fields)
    ai_analysis?: {
        one_liner: string;
        requirements_summary: string;
        recommended_products: string[];
        pricing_confidence: string;
        delivery_urgency: string;
        value_estimate: string;
        estimated_value?: number;
        specifications_recommendations?: any;
    };
    total_value?: number;
    pdf_generated?: boolean;
    // AI Processing fields
    requires_ai_processing?: boolean;  // True if customer submitted generic request
    ai_processed?: boolean;  // True if AI has analyzed this request
    original_message?: string;  // Customer's original free-text message (generic requests)
    notes?: string;
}

interface QuotationStats {
    total: number;
    by_status: Record<string, number>;
    total_value_pending: number;
}

export default function QuotationsDashboard() {
    const [quotations, setQuotations] = useState<Quotation[]>([]);
    const [stats, setStats] = useState<QuotationStats | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedQuotation, setSelectedQuotation] = useState<Quotation | null>(null);
    const [filterStatus, setFilterStatus] = useState<string>('all');
    const [generatingPDF, setGeneratingPDF] = useState<string | null>(null);
    const [editingPrices, setEditingPrices] = useState<Record<string, number>>({});
    const [editingLineItems, setEditingLineItems] = useState<Record<string, Record<string, any>>>({});

    // For browser-side fetch calls, we must use localhost:8000
    // The NEXT_PUBLIC_API_URL (http://api:8000) only works for server-side rendering within Docker
    // Client-side JavaScript runs in the user's browser which can't resolve 'api' hostname
    const API_BASE = typeof window !== 'undefined' ? 'http://localhost:8000' : (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');

    useEffect(() => {
        fetchQuotations();
    }, []);

    const fetchQuotations = async () => {
        setIsLoading(true);
        try {
            // Use Next.js API route to proxy to backend (server-side fetch)
            const response = await fetch('/api/quotations');
            if (!response.ok) {
                throw new Error('Failed to fetch quotations');
            }
            const data = await response.json();
            setQuotations(data.quotations || []);
            setStats(data.stats || null);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load quotations');
            // Use mock data for demo
            setQuotations([
                {
                    id: 'QR-20260205-001',
                    quotation_number: 'QUO-2026-0001',
                    customer: { name: 'John Smith', company: 'ABC Industries', email: 'john@abc.com' },
                    status: 'pending_pricing',
                    priority: 'high',
                    created_at: new Date().toISOString(),
                    line_items: [
                        { id: 'li-1', product_code: 'NA 701', product_name: 'Graphite Packing', quantity: 50, size: '12mm × 12mm', style: 'Braided', material_grade: 'Standard', is_ai_suggested: false },
                        { id: 'li-2', product_code: 'NA 715', product_name: 'PTFE Packing', quantity: 25, size: '10mm × 10mm', style: 'Die-formed', material_grade: 'High Purity', is_ai_suggested: false }
                    ],
                    ai_analysis: {
                        one_liner: 'Standard valve packing order for refinery customer',
                        requirements_summary: 'Customer needs valve packing for high-temp steam application in refinery',
                        recommended_products: ['NA 701', 'NA 715', 'NA 750'],
                        pricing_confidence: 'high',
                        delivery_urgency: '2-3 weeks',
                        value_estimate: '₹50,000 - ₹75,000'
                    }
                },
                {
                    id: 'QR-20260205-002',
                    quotation_number: 'QUO-2026-0002',
                    customer: { name: 'Sarah Lee', company: 'PetroChem Ltd', email: 'sarah@petrochem.com' },
                    status: 'ready_to_send',
                    priority: 'medium',
                    created_at: new Date(Date.now() - 86400000).toISOString(),
                    assigned_to: 'Sales Team',
                    line_items: [
                        { id: 'li-3', product_code: 'NA 750', product_name: 'Aramid Packing', quantity: 100, unit_price: 450, size: '8mm × 8mm', style: 'Braided', material_grade: 'Standard', is_ai_suggested: false }
                    ],
                    ai_analysis: {
                        one_liner: 'Fugitive emission sealing for petrochemical plant',
                        requirements_summary: 'API 622 certified packing for valve emission control',
                        recommended_products: ['NA 750'],
                        pricing_confidence: 'high',
                        delivery_urgency: '1 week',
                        value_estimate: '₹45,000'
                    },
                    total_value: 45000,
                    pdf_generated: true
                }
            ]);
            setStats({
                total: 2,
                by_status: { pending_pricing: 1, ready_to_send: 1 },
                total_value_pending: 50000
            });
        } finally {
            setIsLoading(false);
        }
    };

    const generatePDF = async (quotationId: string, includePricing: boolean) => {
        setGeneratingPDF(quotationId);
        try {
            // Use demo endpoint (follows datasheet generation pattern)
            // API_BASE should be http://localhost:8000 for browser access
            console.log('Generating PDF for quotation:', quotationId);
            console.log('API_BASE:', API_BASE);
            const generateUrl = `${API_BASE}/demo/quotations/${quotationId}/generate-pdf?include_pricing=${includePricing}`;
            console.log('Generate URL:', generateUrl);

            const response = await fetch(generateUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            console.log('Generate response status:', response.status);

            if (response.ok) {
                // Response is JSON with download URL (like datasheet endpoint)
                const data = await response.json();
                console.log('Generate response data:', data);

                if (data.download_url) {
                    // Fetch the actual PDF from the download URL
                    const downloadUrl = `${API_BASE}${data.download_url}`;
                    console.log('Download URL:', downloadUrl);

                    const pdfResponse = await fetch(downloadUrl);
                    console.log('Download response status:', pdfResponse.status);

                    if (pdfResponse.ok) {
                        const blob = await pdfResponse.blob();
                        console.log('PDF blob size:', blob.size);
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = data.filename || `quotation_${quotationId}.pdf`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        window.URL.revokeObjectURL(url);
                        console.log('PDF download complete');
                    } else {
                        const errorText = await pdfResponse.text();
                        console.error('PDF download failed:', pdfResponse.status, errorText);
                        alert(`Failed to download PDF file: ${pdfResponse.status}`);
                    }
                } else {
                    console.error('No download_url in response:', data);
                    alert('PDF generated but no download URL provided');
                }
            } else {
                const errorText = await response.text();
                console.error('PDF generation failed:', response.status, errorText);
                alert(`Failed to generate PDF: ${response.status} - ${errorText}`);
            }
        } catch (err) {
            console.error('PDF generation exception:', err);
            alert(`Failed to generate PDF: ${err instanceof Error ? err.message : String(err)}`);
        } finally {
            setGeneratingPDF(null);
        }
    };

    const filteredQuotations = quotations.filter(q =>
        filterStatus === 'all' || q.status === filterStatus
    );

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'pending_pricing': return { bg: '#fff3e0', color: '#e65100' };
            case 'ready_to_send': return { bg: '#e8f5e9', color: '#2e7d32' };
            case 'sent': return { bg: '#e3f2fd', color: '#1565c0' };
            case 'accepted': return { bg: '#c8e6c9', color: '#1b5e20' };
            case 'rejected': return { bg: '#ffcdd2', color: '#c62828' };
            default: return { bg: '#f5f5f5', color: '#666' };
        }
    };

    const getPriorityColor = (priority: string) => {
        switch (priority) {
            case 'high': return { bg: '#ffebee', color: '#c62828' };
            case 'medium': return { bg: '#fff8e1', color: '#f9a825' };
            case 'low': return { bg: '#e8f5e9', color: '#2e7d32' };
            default: return { bg: '#f5f5f5', color: '#666' };
        }
    };

    const formatCurrency = (amount: number) => {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            minimumFractionDigits: 0
        }).format(amount);
    };

    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        return date.toLocaleString('en-IN', {
            day: '2-digit',
            month: 'short',
            year: 'numeric'
        });
    };

    if (isLoading) {
        return (
            <div style={{ padding: '2rem', textAlign: 'center' }}>
                <p style={{ color: '#666' }}>Loading quotations...</p>
            </div>
        );
    }

    return (
        <div style={{ padding: '1.5rem' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <div>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#2d2d2d', margin: 0 }}>
                        Quotation Management
                    </h2>
                    <p style={{ fontSize: '0.8rem', color: '#666', margin: '0.25rem 0 0' }}>
                        AI-powered analysis • Smart pricing • PDF generation
                    </p>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <select
                        value={filterStatus}
                        onChange={(e) => setFilterStatus(e.target.value)}
                        style={{ padding: '0.5rem', border: '1px solid #ddd', borderRadius: '4px', fontSize: '0.85rem' }}
                    >
                        <option value="all">All Status</option>
                        <option value="pending">Pending</option>
                        <option value="pending_pricing">Pending Pricing</option>
                        <option value="ready_to_send">Ready to Send</option>
                        <option value="sent">Sent</option>
                        <option value="accepted">Accepted</option>
                        <option value="rejected">Rejected</option>
                    </select>
                    <button
                        onClick={fetchQuotations}
                        style={{
                            padding: '0.5rem 1rem',
                            backgroundColor: '#952825',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '0.85rem'
                        }}
                    >
                        Refresh
                    </button>
                </div>
            </div>

            {/* Stats Cards - computed from quotations */}
            {quotations.length > 0 && (() => {
                // Compute stats from actual quotations data
                const pendingCount = quotations.filter((q: Quotation) => q.status === 'pending' || q.status === 'pending_pricing').length;
                const readyToSendCount = quotations.filter((q: Quotation) => q.status === 'ready_to_send' || q.status === 'quoted').length;
                const sentCount = quotations.filter((q: Quotation) => q.status === 'sent').length;
                const totalValue = quotations.reduce((sum: number, q: Quotation) => {
                    // Calculate value from line items or estimated value
                    const itemsValue = q.line_items?.reduce((acc: number, item: LineItem) =>
                        acc + ((item.unit_price || 0) * (item.quantity || 1)), 0) || 0;
                    const estimatedValue = q.ai_analysis?.estimated_value || 0;
                    return sum + (itemsValue > 0 ? itemsValue : estimatedValue);
                }, 0);

                return (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                        <div style={{ padding: '1rem', backgroundColor: '#fff3e0', borderRadius: '8px', textAlign: 'center' }}>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#e65100' }}>
                                {pendingCount}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: '#bf360c' }}>Pending Pricing</div>
                        </div>
                        <div style={{ padding: '1rem', backgroundColor: '#e8f5e9', borderRadius: '8px', textAlign: 'center' }}>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#2e7d32' }}>
                                {readyToSendCount}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: '#1b5e20' }}>Ready to Send</div>
                        </div>
                        <div style={{ padding: '1rem', backgroundColor: '#e3f2fd', borderRadius: '8px', textAlign: 'center' }}>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#1565c0' }}>
                                {sentCount}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: '#0d47a1' }}>Sent</div>
                        </div>
                        <div style={{ padding: '1rem', backgroundColor: '#f3e5f5', borderRadius: '8px', textAlign: 'center' }}>
                            <div style={{ fontSize: '1rem', fontWeight: '700', color: '#7b1fa2' }}>
                                {formatCurrency(totalValue)}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: '#4a148c' }}>Pipeline Value</div>
                        </div>
                    </div>
                );
            })()}

            {error && (
                <div style={{
                    padding: '0.75rem',
                    backgroundColor: '#fff3e0',
                    border: '1px solid #ffcc80',
                    borderRadius: '6px',
                    color: '#e65100',
                    marginBottom: '1rem',
                    fontSize: '0.875rem'
                }}>
                    Note: Showing demo data. {error}
                </div>
            )}

            {/* Quotation List */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {filteredQuotations.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
                        No quotations found
                    </div>
                ) : (
                    filteredQuotations.map((quotation) => {
                        const isSelected = selectedQuotation?.id === quotation.id;
                        const ai = quotation.ai_analysis;

                        return (
                            <div
                                key={quotation.id}
                                onClick={() => setSelectedQuotation(isSelected ? null : quotation)}
                                style={{
                                    padding: '1rem',
                                    backgroundColor: 'white',
                                    border: isSelected ? '2px solid #952825' : '1px solid #e5e5e5',
                                    borderRadius: '8px',
                                    cursor: 'pointer',
                                    transition: 'all 0.2s'
                                }}
                            >
                                {/* Header Row */}
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                                    <div>
                                        <span style={{ fontWeight: '600', color: '#952825', fontSize: '0.9rem' }}>
                                            {quotation.quotation_number}
                                        </span>
                                        <span style={{ color: '#666', fontSize: '0.875rem', marginLeft: '0.75rem' }}>
                                            {quotation.customer.name}
                                        </span>
                                        {quotation.customer.company && (
                                            <span style={{ color: '#888', fontSize: '0.8rem' }}> • {quotation.customer.company}</span>
                                        )}
                                    </div>
                                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                                        <span style={{
                                            padding: '0.25rem 0.5rem',
                                            borderRadius: '4px',
                                            fontSize: '0.7rem',
                                            fontWeight: '600',
                                            ...getPriorityColor(quotation.priority)
                                        }}>
                                            {quotation.priority.toUpperCase()}
                                        </span>
                                        <span style={{
                                            padding: '0.25rem 0.5rem',
                                            borderRadius: '4px',
                                            fontSize: '0.7rem',
                                            fontWeight: '500',
                                            ...getStatusColor(quotation.status)
                                        }}>
                                            {quotation.status.replace('_', ' ').toUpperCase()}
                                        </span>
                                    </div>
                                </div>

                                {/* AI Summary */}
                                {ai && (
                                    <div style={{
                                        padding: '0.5rem 0.75rem',
                                        backgroundColor: '#f8f9fa',
                                        borderRadius: '4px',
                                        borderLeft: '3px solid #952825',
                                        marginBottom: '0.5rem'
                                    }}>
                                        <span style={{ fontSize: '0.75rem', color: '#952825', fontWeight: '500' }}>AI ANALYSIS: </span>
                                        <span style={{ fontSize: '0.875rem', color: '#333' }}>{ai.one_liner}</span>
                                    </div>
                                )}

                                {/* Line Items Summary */}
                                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
                                    {quotation.line_items.slice(0, 3).map((item, idx) => (
                                        <span key={idx} style={{
                                            padding: '0.25rem 0.5rem',
                                            backgroundColor: '#e3f2fd',
                                            color: '#1565c0',
                                            borderRadius: '4px',
                                            fontSize: '0.75rem'
                                        }}>
                                            {item.product_code} × {item.quantity}
                                        </span>
                                    ))}
                                    {quotation.line_items.length > 3 && (
                                        <span style={{ fontSize: '0.75rem', color: '#666' }}>
                                            +{quotation.line_items.length - 3} more
                                        </span>
                                    )}
                                </div>

                                {/* Footer Row */}
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#888' }}>
                                    <span>{formatDate(quotation.created_at)}</span>
                                    <div style={{ display: 'flex', gap: '1rem' }}>
                                        {ai?.delivery_urgency && (
                                            <span>Delivery: {ai.delivery_urgency}</span>
                                        )}
                                        {ai?.value_estimate && (
                                            <span style={{ color: '#2e7d32', fontWeight: '500' }}>
                                                {ai.value_estimate}
                                            </span>
                                        )}
                                        {quotation.pdf_generated && (
                                            <span style={{ color: '#1565c0' }}>PDF Ready</span>
                                        )}
                                    </div>
                                </div>

                                {/* Expanded Details */}
                                {isSelected && (
                                    <div style={{
                                        marginTop: '1rem',
                                        paddingTop: '1rem',
                                        borderTop: '1px solid #e5e5e5'
                                    }}>
                                        {/* Describe Your Application (for generic quotations) */}
                                        {quotation.is_generic && quotation.original_message && (
                                            <div style={{
                                                marginBottom: '1rem',
                                                padding: '0.75rem',
                                                backgroundColor: '#fff8e1',
                                                borderRadius: '6px',
                                                border: '1px solid #ffe082'
                                            }}>
                                                <strong style={{ fontSize: '0.8rem', color: '#f57f17' }}>Describe Your Application (Customer Input):</strong>
                                                <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.875rem', color: '#444', whiteSpace: 'pre-wrap' }}>
                                                    {quotation.original_message}
                                                </p>
                                            </div>
                                        )}

                                        {/* Special Requirements & Notes (for specific quotations) */}
                                        {!quotation.is_generic && quotation.notes && (
                                            <div style={{
                                                marginBottom: '1rem',
                                                padding: '0.75rem',
                                                backgroundColor: '#f3e5f5',
                                                borderRadius: '6px',
                                                border: '1px solid #ce93d8'
                                            }}>
                                                <strong style={{ fontSize: '0.8rem', color: '#7b1fa2' }}>Special Requirements:</strong>
                                                <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.875rem', color: '#444', whiteSpace: 'pre-wrap' }}>
                                                    {quotation.notes}
                                                </p>
                                            </div>
                                        )}

                                        {/* Requirements Summary */}
                                        {ai?.requirements_summary && (
                                            <div style={{ marginBottom: '1rem' }}>
                                                <strong style={{ fontSize: '0.8rem', color: '#666' }}>Requirements:</strong>
                                                <p style={{ margin: '0.25rem 0', fontSize: '0.875rem', color: '#444' }}>
                                                    {ai.requirements_summary}
                                                </p>
                                            </div>
                                        )}

                                        {/* AI Suggested Notice */}
                                        {quotation.is_generic && (
                                            <div style={{
                                                marginBottom: '0.75rem',
                                                padding: '0.5rem 0.75rem',
                                                backgroundColor: '#e3f2fd',
                                                borderRadius: '4px',
                                                borderLeft: '3px solid #1565c0',
                                                fontSize: '0.8rem',
                                                color: '#1565c0',
                                                fontWeight: '500'
                                            }}>
                                                AI SUGGESTED SPECIFICATIONS - The fields below were auto-filled by AI analysis. Review and edit as needed before sending.
                                            </div>
                                        )}

                                        {/* Line Items Table - Editable */}
                                        <div style={{ marginBottom: '1rem' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                                <strong style={{ fontSize: '0.8rem', color: '#666' }}>Line Items:</strong>
                                                {quotation.is_generic && (
                                                    <span style={{
                                                        backgroundColor: '#e3f2fd',
                                                        color: '#1565c0',
                                                        padding: '2px 8px',
                                                        borderRadius: '4px',
                                                        fontSize: '0.7rem',
                                                        fontWeight: 'bold'
                                                    }}>
                                                        AI Suggested
                                                    </span>
                                                )}
                                            </div>
                                            <table style={{ width: '100%', marginTop: '0.5rem', fontSize: '0.8rem', borderCollapse: 'collapse' }}>
                                                <thead>
                                                    <tr style={{ backgroundColor: '#4472C4', color: 'white' }}>
                                                        <th style={{ padding: '0.4rem', textAlign: 'center', borderBottom: '1px solid #2a5292', borderRight: '1px solid #5a8ad4', fontSize: '0.7rem' }} rowSpan={2}>Sl.<br />No.</th>
                                                        <th style={{ padding: '0.4rem', textAlign: 'center', borderBottom: '1px solid #2a5292', borderRight: '1px solid #5a8ad4', fontSize: '0.7rem' }} rowSpan={2}>JDJ<br />Style</th>
                                                        <th style={{ padding: '0.4rem', textAlign: 'center', borderBottom: '1px solid #2a5292', borderRight: '1px solid #5a8ad4', fontSize: '0.7rem' }} rowSpan={2}>Material<br />Code/Grade</th>
                                                        <th style={{ padding: '0.4rem', textAlign: 'center', borderBottom: '1px solid #5a8ad4', borderRight: '1px solid #5a8ad4', fontSize: '0.7rem' }} colSpan={3}>{(() => { const units = Array.from(new Set(quotation.line_items.map((li: LineItem) => li.dimension_unit || 'mm'))); return units.length > 1 ? 'Size (mm/inch)' : `Size (${units[0]})`; })()}</th>
                                                        <th style={{ padding: '0.4rem', textAlign: 'center', borderBottom: '1px solid #2a5292', borderRight: '1px solid #5a8ad4', fontSize: '0.7rem' }} rowSpan={2}>Rings/<br />Set</th>
                                                        <th style={{ padding: '0.4rem', textAlign: 'center', borderBottom: '1px solid #2a5292', borderRight: '1px solid #5a8ad4', fontSize: '0.7rem' }} rowSpan={2}>Qty</th>
                                                        <th style={{ padding: '0.4rem', textAlign: 'center', borderBottom: '1px solid #2a5292', borderRight: '1px solid #5a8ad4', fontSize: '0.7rem' }} rowSpan={2}>Unit</th>
                                                        <th style={{ padding: '0.4rem', textAlign: 'center', borderBottom: '1px solid #2a5292', borderRight: '1px solid #5a8ad4', fontSize: '0.7rem' }} rowSpan={2}>Price/UOM<br />(Rs.)</th>
                                                        <th style={{ padding: '0.4rem', textAlign: 'center', borderBottom: '1px solid #2a5292', fontSize: '0.7rem' }} rowSpan={2}>Amount<br />Excl. GST<br />(Rs.)</th>
                                                    </tr>
                                                    <tr style={{ backgroundColor: '#4472C4', color: 'white' }}>
                                                        <th style={{ padding: '0.3rem', textAlign: 'center', borderBottom: '1px solid #2a5292', borderRight: '1px solid #5a8ad4', fontSize: '0.7rem' }}>OD</th>
                                                        <th style={{ padding: '0.3rem', textAlign: 'center', borderBottom: '1px solid #2a5292', borderRight: '1px solid #5a8ad4', fontSize: '0.7rem' }}>ID</th>
                                                        <th style={{ padding: '0.3rem', textAlign: 'center', borderBottom: '1px solid #2a5292', borderRight: '1px solid #5a8ad4', fontSize: '0.7rem' }}>TH</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {quotation.line_items.map((item: LineItem, idx: number) => {
                                                        const editKey = `${quotation.id}-${idx}`;
                                                        const itemEdits = editingLineItems[editKey] || {};
                                                        const unitPrice = item.unit_price || 0;
                                                        const amount = unitPrice * item.quantity;
                                                        // Parse size: could be "OD x ID x TH" string or separate fields
                                                        let od = item.size_od || item.dimensions?.od || '-';
                                                        let id_val = item.size_id || item.dimensions?.id || '-';
                                                        let th = item.size_th || item.dimensions?.th || '-';
                                                        if (od === '-' && item.size && typeof item.size === 'string' && item.size.includes('x')) {
                                                            const parts = item.size.split('x').map((s: string) => s.trim());
                                                            if (parts.length >= 1) od = parts[0];
                                                            if (parts.length >= 2) id_val = parts[1];
                                                            if (parts.length >= 3) th = parts[2];
                                                        }
                                                        const cellStyle = { padding: '0.35rem', borderBottom: '1px solid #ddd', borderRight: '1px solid #eee', textAlign: 'center' as const, fontSize: '0.8rem' };
                                                        const inputStyle = { width: '60px', padding: '0.2rem', border: '1px solid #ddd', borderRadius: '3px', fontSize: '0.75rem', textAlign: 'center' as const, backgroundColor: item.is_ai_suggested ? '#e8f0fe' : 'white' };
                                                        return (
                                                            <tr key={item.id || idx} style={{ backgroundColor: item.is_ai_suggested ? '#f3f8ff' : (idx % 2 === 0 ? '#fff' : '#f9fbff') }}>
                                                                <td style={cellStyle}>
                                                                    {String(idx + 1).padStart(2, '0')}
                                                                    {item.is_ai_suggested && <span style={{ display: 'block', backgroundColor: '#1565c0', color: 'white', padding: '0px 2px', borderRadius: '2px', fontSize: '0.55rem', fontWeight: 'bold', marginTop: '2px' }}>AI</span>}
                                                                </td>
                                                                <td style={cellStyle}>
                                                                    <input type="text" defaultValue={item.product_code} placeholder="NA XXX"
                                                                        onClick={(e: React.MouseEvent) => e.stopPropagation()}
                                                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditingLineItems((prev: Record<string, Record<string, string>>) => ({ ...prev, [editKey]: { ...prev[editKey], product_code: e.target.value } }))}
                                                                        style={{ ...inputStyle, width: '65px' }} />
                                                                </td>
                                                                <td style={cellStyle}>
                                                                    <input type="text" defaultValue={item.material_code || ''} placeholder="Mat. code"
                                                                        onClick={(e: React.MouseEvent) => e.stopPropagation()}
                                                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditingLineItems((prev: Record<string, Record<string, string>>) => ({ ...prev, [editKey]: { ...prev[editKey], material_code: e.target.value } }))}
                                                                        style={{ ...inputStyle, width: '90px' }} />
                                                                    <input type="text" defaultValue={item.material_grade || ''} placeholder="Grade"
                                                                        onClick={(e: React.MouseEvent) => e.stopPropagation()}
                                                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditingLineItems((prev: Record<string, Record<string, string>>) => ({ ...prev, [editKey]: { ...prev[editKey], material_grade: e.target.value } }))}
                                                                        style={{ ...inputStyle, width: '90px', marginTop: '2px', fontSize: '0.65rem', color: '#555' }} />
                                                                </td>
                                                                <td style={cellStyle}>
                                                                    <input type="number" defaultValue={od !== '-' ? od : ''} placeholder="OD"
                                                                        onClick={(e: React.MouseEvent) => e.stopPropagation()}
                                                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditingLineItems((prev: Record<string, Record<string, string>>) => ({ ...prev, [editKey]: { ...prev[editKey], size_od: e.target.value } }))}
                                                                        step="0.01" min="0"
                                                                        style={{ ...inputStyle, width: '50px' }} />
                                                                </td>
                                                                <td style={cellStyle}>
                                                                    <input type="number" defaultValue={id_val !== '-' ? id_val : ''} placeholder="ID"
                                                                        onClick={(e: React.MouseEvent) => e.stopPropagation()}
                                                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditingLineItems((prev: Record<string, Record<string, string>>) => ({ ...prev, [editKey]: { ...prev[editKey], size_id: e.target.value } }))}
                                                                        step="0.01" min="0"
                                                                        style={{ ...inputStyle, width: '50px' }} />
                                                                </td>
                                                                <td style={cellStyle}>
                                                                    <input type="number" defaultValue={th !== '-' ? th : ''} placeholder="TH"
                                                                        onClick={(e: React.MouseEvent) => e.stopPropagation()}
                                                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditingLineItems((prev: Record<string, Record<string, string>>) => ({ ...prev, [editKey]: { ...prev[editKey], size_th: e.target.value } }))}
                                                                        step="0.01" min="0"
                                                                        style={{ ...inputStyle, width: '45px' }} />
                                                                </td>
                                                                <td style={cellStyle}>
                                                                    <input type="number" defaultValue={item.rings_per_set && item.rings_per_set !== '-' ? item.rings_per_set : ''} placeholder="-"
                                                                        onClick={(e: React.MouseEvent) => e.stopPropagation()}
                                                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditingLineItems((prev: Record<string, Record<string, string>>) => ({ ...prev, [editKey]: { ...prev[editKey], rings_per_set: e.target.value } }))}
                                                                        min="1"
                                                                        style={{ ...inputStyle, width: '40px' }} />
                                                                </td>
                                                                <td style={cellStyle}>
                                                                    <input type="number" defaultValue={item.quantity} min={1}
                                                                        onClick={(e: React.MouseEvent) => e.stopPropagation()}
                                                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditingLineItems((prev: Record<string, Record<string, string>>) => ({ ...prev, [editKey]: { ...prev[editKey], quantity: e.target.value } }))}
                                                                        style={{ ...inputStyle, width: '50px' }} />
                                                                </td>
                                                                <td style={cellStyle}>
                                                                    <select defaultValue={item.unit || 'Nos.'}
                                                                        onClick={(e: React.MouseEvent) => e.stopPropagation()}
                                                                        onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setEditingLineItems((prev: Record<string, Record<string, string>>) => ({ ...prev, [editKey]: { ...prev[editKey], unit: e.target.value } }))}
                                                                        style={{ ...inputStyle, width: '55px' }}>
                                                                        <option value="Nos.">Nos.</option>
                                                                        <option value="Set">Set</option>
                                                                        <option value="Mtr.">Mtr.</option>
                                                                        <option value="Kg.">Kg.</option>
                                                                    </select>
                                                                </td>
                                                                <td style={cellStyle}>
                                                                    <input type="number" defaultValue={unitPrice || ''} placeholder="Price"
                                                                        onClick={(e: React.MouseEvent) => e.stopPropagation()}
                                                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditingPrices({ ...editingPrices, [`${quotation.id}-${idx}`]: parseFloat(e.target.value) })}
                                                                        style={{ ...inputStyle, width: '65px' }} />
                                                                </td>
                                                                <td style={{ ...cellStyle, fontWeight: '600' }}>
                                                                    {amount > 0 ? amount.toLocaleString('en-IN') : '-'}
                                                                </td>
                                                            </tr>
                                                        );
                                                    })}
                                                    {/* Total Row */}
                                                    <tr style={{ backgroundColor: '#D9E2F3', fontWeight: 'bold' }}>
                                                        <td colSpan={10} style={{ padding: '0.4rem', textAlign: 'right', borderTop: '2px solid #4472C4', fontSize: '0.8rem' }}>
                                                            Total Amount Excl GST (In Rs.)
                                                        </td>
                                                        <td style={{ padding: '0.4rem', textAlign: 'center', borderTop: '2px solid #4472C4', fontSize: '0.8rem', fontWeight: 'bold' }}>
                                                            {quotation.line_items.reduce((sum: number, item: LineItem) => sum + ((item.unit_price || 0) * item.quantity), 0).toLocaleString('en-IN')}
                                                        </td>
                                                    </tr>
                                                </tbody>
                                            </table>
                                        </div>

                                        {/* Actions */}
                                        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    generatePDF(quotation.id, true);
                                                }}
                                                disabled={generatingPDF === quotation.id}
                                                style={{
                                                    padding: '0.5rem 1rem',
                                                    backgroundColor: generatingPDF === quotation.id ? '#ccc' : '#952825',
                                                    color: 'white',
                                                    border: 'none',
                                                    borderRadius: '4px',
                                                    cursor: generatingPDF === quotation.id ? 'wait' : 'pointer',
                                                    fontSize: '0.8rem'
                                                }}
                                            >
                                                {generatingPDF === quotation.id ? 'Generating...' : 'Generate PDF'}
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    window.location.href = `mailto:${quotation.customer.email}?subject=Quotation ${quotation.quotation_number}`;
                                                }}
                                                style={{
                                                    padding: '0.5rem 1rem',
                                                    backgroundColor: '#1565c0',
                                                    color: 'white',
                                                    border: 'none',
                                                    borderRadius: '4px',
                                                    cursor: 'pointer',
                                                    fontSize: '0.8rem'
                                                }}
                                            >
                                                Email Customer
                                            </button>
                                            <button
                                                onClick={async (e) => {
                                                    e.stopPropagation();
                                                    // Collect prices from editingPrices state
                                                    const prices: Record<string, number> = {};
                                                    quotation.line_items.forEach((item, idx) => {
                                                        const editedPrice = editingPrices[`${quotation.id}-${idx}`];
                                                        if (editedPrice && editedPrice > 0) {
                                                            prices[String(idx)] = editedPrice;
                                                        } else if (item.unit_price && item.unit_price > 0) {
                                                            prices[String(idx)] = item.unit_price;
                                                        }
                                                    });

                                                    // Collect line item field updates
                                                    const lineItemUpdates: Record<string, Record<string, any>> = {};
                                                    let hasAnyUpdate = false;
                                                    quotation.line_items.forEach((_item, idx) => {
                                                        const editKey = `${quotation.id}-${idx}`;
                                                        const edits = editingLineItems[editKey];
                                                        if (edits && Object.keys(edits).length > 0) {
                                                            lineItemUpdates[String(idx)] = edits;
                                                            hasAnyUpdate = true;
                                                        }
                                                    });

                                                    if (!hasAnyUpdate && Object.keys(prices).length === 0) {
                                                        alert('Please make at least one change before saving.');
                                                        return;
                                                    }

                                                    try {
                                                        const response = await fetch(`${API_BASE}/demo/quotations/${quotation.id}/save-prices`, {
                                                            method: 'POST',
                                                            headers: { 'Content-Type': 'application/json' },
                                                            body: JSON.stringify({ prices, line_item_updates: lineItemUpdates })
                                                        });
                                                        if (response.ok) {
                                                            const data = await response.json();
                                                            alert(`Changes saved! ${data.message}`);
                                                            setEditingPrices({});
                                                            setEditingLineItems({});
                                                            fetchQuotations();
                                                        } else {
                                                            const errorText = await response.text();
                                                            alert(`Failed to save: ${errorText}`);
                                                        }
                                                    } catch (err) {
                                                        alert(`Error saving: ${err instanceof Error ? err.message : String(err)}`);
                                                    }
                                                }}
                                                style={{
                                                    padding: '0.5rem 1rem',
                                                    backgroundColor: '#4caf50',
                                                    color: 'white',
                                                    border: 'none',
                                                    borderRadius: '4px',
                                                    cursor: 'pointer',
                                                    fontSize: '0.8rem'
                                                }}
                                            >
                                                Save Changes
                                            </button>
                                            <button
                                                onClick={async (e) => {
                                                    e.stopPropagation();
                                                    if (!confirm('Mark this quotation as sent to customer?')) return;
                                                    try {
                                                        const response = await fetch(`${API_BASE}/demo/quotations/${quotation.id}/mark-sent`, {
                                                            method: 'POST',
                                                            headers: { 'Content-Type': 'application/json' }
                                                        });
                                                        if (response.ok) {
                                                            alert('Quotation marked as sent!');
                                                            fetchQuotations();
                                                        } else {
                                                            const errorText = await response.text();
                                                            alert(`Failed to mark as sent: ${errorText}`);
                                                        }
                                                    } catch (err) {
                                                        alert(`Error: ${err instanceof Error ? err.message : String(err)}`);
                                                    }
                                                }}
                                                style={{
                                                    padding: '0.5rem 1rem',
                                                    backgroundColor: '#f5f5f5',
                                                    color: '#666',
                                                    border: '1px solid #ddd',
                                                    borderRadius: '4px',
                                                    cursor: 'pointer',
                                                    fontSize: '0.8rem'
                                                }}
                                            >
                                                Mark as Sent
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
}
