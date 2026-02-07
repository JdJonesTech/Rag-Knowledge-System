'use client';

import { useState, useEffect } from 'react';

interface AIQuickView {
    one_liner: string;
    type: string;
    priority: string;
    urgency: number;
    main_ask: string;
    products: string[];
    actions_needed: {
        technical_review: boolean;
        pricing: boolean;
        samples: boolean;
    };
}

interface Requirements {
    industry?: string;
    application?: string;
    operating_temperature?: string;
    operating_pressure?: string;
    media_handled?: string;
    quantity?: string;
    delivery_urgency?: string;
}

interface AIAnalysis {
    summary: string;
    one_liner: string;
    detected_type: string;
    detected_priority: string;
    confidence_score: number;
    customer_intent: string;
    main_ask: string;
    secondary_asks: string[];
    requirements?: Requirements;
    key_points: string[];
    suggested_products: string[];
    product_match_confidence: string;
    sentiment: string;
    urgency_score: number;
    recommended_actions: string[];
    requires_technical_review: boolean;
    requires_pricing: boolean;
    requires_samples: boolean;
}

interface Enquiry {
    id: string;
    customer: {
        name: string;
        company?: string;
        email: string;
    };
    subject?: string;
    raw_message?: string;
    status: string;
    created_at: string;
    assigned_to?: string;
    ai_quick_view?: AIQuickView;
    ai_analysis?: AIAnalysis;
    latest_suggested_response?: {
        response_text?: string;
        [key: string]: any;
    };
}

interface DashboardData {
    enquiries: Enquiry[];
    stats: {
        total: number;
        by_status: Record<string, number>;
        by_priority: Record<string, number>;
        by_type: Record<string, number>;
    };
}

export default function EnquiriesDashboard() {
    const [enquiries, setEnquiries] = useState<Enquiry[]>([]);
    const [stats, setStats] = useState<DashboardData['stats'] | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedEnquiry, setSelectedEnquiry] = useState<Enquiry | null>(null);
    const [filterStatus, setFilterStatus] = useState<string>('all');
    const [filterPriority, setFilterPriority] = useState<string>('all');
    const [generatingResponse, setGeneratingResponse] = useState<string | null>(null);
    const [reanalyzing, setReanalyzing] = useState<string | null>(null);
    const [markingSent, setMarkingSent] = useState<string | null>(null);
    const [suggestedResponses, setSuggestedResponses] = useState<Record<string, string>>({});
    const [showRawMessage, setShowRawMessage] = useState(false);

    const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    useEffect(() => {
        fetchDashboard();
    }, []);

    // Auto-poll when there are enquiries with 'new' status (analysis pending)
    useEffect(() => {
        const hasPendingAnalysis = enquiries.some(e => e.status === 'new');
        if (!hasPendingAnalysis || enquiries.length === 0) return;

        const interval = setInterval(async () => {
            try {
                const response = await fetch('/api/enquiries');
                if (response.ok) {
                    const data: DashboardData = await response.json();
                    setEnquiries(data.enquiries || []);
                    setStats(data.stats || null);
                }
            } catch (err) {
                // Silently fail polling - user can still manually refresh
            }
        }, 5000); // Poll every 5 seconds

        return () => clearInterval(interval);
    }, [enquiries]);

    const fetchDashboard = async () => {
        setIsLoading(true);
        try {
            // Use Next.js API route to proxy to backend (server-side fetch)
            const response = await fetch('/api/enquiries');
            if (!response.ok) {
                throw new Error('Failed to fetch enquiries');
            }
            const data: DashboardData = await response.json();
            const fetchedEnquiries = data.enquiries || [];
            setEnquiries(fetchedEnquiries);
            setStats(data.stats || null);

            // Populate suggestedResponses from backend-stored responses
            // This ensures each enquiry shows its own correct AI response
            setSuggestedResponses(prev => {
                const updated = { ...prev };
                for (const enq of fetchedEnquiries) {
                    if (enq.latest_suggested_response?.response_text && !updated[enq.id]) {
                        updated[enq.id] = enq.latest_suggested_response.response_text;
                    }
                }
                return updated;
            });
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load enquiries');
            // Use mock data for demo
            setEnquiries([
                {
                    id: 'ENQ-20260205-001',
                    customer: { name: 'John Smith', company: 'ABC Industries', email: 'john@abc.com' },
                    subject: 'Valve packing for high temperature application',
                    status: 'under_review',
                    created_at: new Date().toISOString(),
                    ai_quick_view: {
                        one_liner: 'Needs valve packing for 350°C steam application',
                        type: 'product_selection',
                        priority: 'high',
                        urgency: 4,
                        main_ask: 'Product recommendation for high-temp steam valve',
                        products: ['NA 701', 'NA 750'],
                        actions_needed: { technical_review: true, pricing: true, samples: false }
                    }
                },
                {
                    id: 'ENQ-20260205-002',
                    customer: { name: 'Sarah Lee', company: 'PetroChem Ltd', email: 'sarah@petrochem.com' },
                    subject: 'Quote request for pump seals',
                    status: 'ai_response_generated',
                    created_at: new Date(Date.now() - 3600000).toISOString(),
                    assigned_to: 'Sales Team',
                    ai_quick_view: {
                        one_liner: 'Requesting quote for 50 pump seal sets',
                        type: 'quotation',
                        priority: 'medium',
                        urgency: 3,
                        main_ask: 'Formal quotation for pump seals',
                        products: ['NA 715'],
                        actions_needed: { technical_review: false, pricing: true, samples: false }
                    }
                }
            ]);
            setStats({
                total: 2,
                by_status: { new: 0, under_review: 1, ai_response_generated: 1 },
                by_priority: { high: 1, medium: 1 },
                by_type: { product_selection: 1, quotation: 1 }
            });
        } finally {
            setIsLoading(false);
        }
    };

    const generateAIResponse = async (enquiryId: string) => {
        setGeneratingResponse(enquiryId);
        try {
            // Use Next.js API proxy to avoid CORS issues
            const response = await fetch(`/api/enquiries/${enquiryId}/generate-response`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tone: 'professional', include_products: true })
            });
            const data = await response.json();
            if (response.ok && data.success) {
                setSuggestedResponses(prev => ({ ...prev, [enquiryId]: data.suggested_response?.response_text || '' }));
                // Refresh to get updated data
                await fetchDashboard();
            } else {
                console.error('Generate response failed:', data.error);
                setSuggestedResponses(prev => ({ ...prev, [enquiryId]: 'Failed to generate AI response: ' + (data.error || 'Unknown error') }));
            }
        } catch (err) {
            console.error('Failed to generate response:', err);
            setSuggestedResponses(prev => ({ ...prev, [enquiryId]: 'Error generating response. Please try again later.' }));
        } finally {
            setGeneratingResponse(null);
        }
    };

    const reanalyzeEnquiry = async (enquiryId: string) => {
        setReanalyzing(enquiryId);
        try {
            // Use Next.js API proxy to avoid CORS issues
            const response = await fetch(`/api/enquiries/${enquiryId}/re-analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (response.ok && data.success) {
                // Refresh dashboard to show updated analysis
                await fetchDashboard();
                // Update selected enquiry if it's the one being re-analyzed
                if (selectedEnquiry?.id === enquiryId) {
                    setSelectedEnquiry(null); // Force re-select to refresh view
                }
                alert('AI analysis refreshed successfully!');
            } else {
                console.error('Re-analyze failed:', data.error);
                alert('Failed to re-analyze: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            console.error('Failed to re-analyze:', err);
            alert('Error re-analyzing enquiry. Please try again.');
        } finally {
            setReanalyzing(null);
        }
    };

    const markAsSent = async (enquiryId: string) => {
        setMarkingSent(enquiryId);
        try {
            const response = await fetch(`/api/enquiries/${enquiryId}/mark-sent`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (response.ok && data.success) {
                await fetchDashboard();
                if (selectedEnquiry?.id === enquiryId) {
                    setSelectedEnquiry(null);
                }
                alert('Enquiry marked as sent!');
            } else {
                alert('Failed to mark as sent: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            console.error('Failed to mark as sent:', err);
            alert('Error marking as sent. Please try again.');
        } finally {
            setMarkingSent(null);
        }
    };

    const filteredEnquiries = enquiries.filter(e => {
        const statusMatch = filterStatus === 'all' || e.status === filterStatus;
        const priorityMatch = filterPriority === 'all' || e.ai_quick_view?.priority === filterPriority;
        return statusMatch && priorityMatch;
    });

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'new': return { bg: '#e3f2fd', color: '#1565c0' };
            case 'under_review': return { bg: '#fff3e0', color: '#e65100' };
            case 'ai_response_generated': return { bg: '#f3e5f5', color: '#7b1fa2' };
            case 'response_sent': return { bg: '#e8f5e9', color: '#2e7d32' };
            case 'escalated': return { bg: '#ffebee', color: '#c62828' };
            case 'resolved': return { bg: '#f5f5f5', color: '#666' };
            default: return { bg: '#f5f5f5', color: '#666' };
        }
    };

    const getPriorityColor = (priority: string) => {
        switch (priority) {
            case 'urgent': return { bg: '#b71c1c', color: 'white' };
            case 'high': return { bg: '#ffebee', color: '#c62828' };
            case 'medium': return { bg: '#fff8e1', color: '#f9a825' };
            case 'low': return { bg: '#e8f5e9', color: '#2e7d32' };
            default: return { bg: '#f5f5f5', color: '#666' };
        }
    };

    const getUrgencyBar = (score: number) => {
        const colors = ['#4caf50', '#8bc34a', '#ffeb3b', '#ff9800', '#f44336'];
        return (
            <div style={{ display: 'flex', gap: '2px' }}>
                {[1, 2, 3, 4, 5].map(i => (
                    <div
                        key={i}
                        style={{
                            width: '8px',
                            height: '12px',
                            borderRadius: '2px',
                            backgroundColor: i <= score ? colors[score - 1] : '#e0e0e0'
                        }}
                    />
                ))}
            </div>
        );
    };

    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        return date.toLocaleString('en-IN', {
            day: '2-digit',
            month: 'short',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    if (isLoading) {
        return (
            <div style={{ padding: '2rem', textAlign: 'center' }}>
                <div style={{
                    width: '40px',
                    height: '40px',
                    border: '3px solid #f3f3f3',
                    borderTop: '3px solid #952825',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite',
                    margin: '0 auto 1rem'
                }} />
                <p style={{ color: '#666' }}>Loading enquiries with AI analysis...</p>
            </div>
        );
    }

    return (
        <div style={{ padding: '1.5rem' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <div>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#2d2d2d', margin: 0 }}>
                        AI-Powered Enquiry Dashboard
                    </h2>
                    <p style={{ fontSize: '0.8rem', color: '#666', margin: '0.25rem 0 0' }}>
                        Multi-agent analysis • Structured summaries • Quick actions
                    </p>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <select
                        value={filterStatus}
                        onChange={(e) => setFilterStatus(e.target.value)}
                        style={{ padding: '0.5rem', border: '1px solid #ddd', borderRadius: '4px', fontSize: '0.85rem' }}
                    >
                        <option value="all">All Status</option>
                        <option value="new">New</option>
                        <option value="under_review">Under Review</option>
                        <option value="ai_response_generated">AI Response Ready</option>
                        <option value="response_sent">Response Sent</option>
                        <option value="escalated">Escalated</option>
                    </select>
                    <select
                        value={filterPriority}
                        onChange={(e) => setFilterPriority(e.target.value)}
                        style={{ padding: '0.5rem', border: '1px solid #ddd', borderRadius: '4px', fontSize: '0.85rem' }}
                    >
                        <option value="all">All Priority</option>
                        <option value="urgent">Urgent</option>
                        <option value="high">High</option>
                        <option value="medium">Medium</option>
                        <option value="low">Low</option>
                    </select>
                    <button
                        onClick={fetchDashboard}
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

            {/* Stats Cards */}
            {stats && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem', marginBottom: '1.5rem' }}>
                    {/* Status row */}
                    <div style={{ padding: '0.75rem', backgroundColor: '#e3f2fd', borderRadius: '8px', textAlign: 'center' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#1565c0' }}>
                            {stats.by_status?.new || 0}
                        </div>
                        <div style={{ fontSize: '0.7rem', color: '#0d47a1' }}>New</div>
                    </div>
                    <div style={{ padding: '0.75rem', backgroundColor: '#fff3e0', borderRadius: '8px', textAlign: 'center' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#e65100' }}>
                            {stats.by_status?.under_review || 0}
                        </div>
                        <div style={{ fontSize: '0.7rem', color: '#bf360c' }}>Under Review</div>
                    </div>
                    <div style={{ padding: '0.75rem', backgroundColor: '#f3e5f5', borderRadius: '8px', textAlign: 'center' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#7b1fa2' }}>
                            {stats.by_status?.ai_response_generated || 0}
                        </div>
                        <div style={{ fontSize: '0.7rem', color: '#4a148c' }}>AI Ready</div>
                    </div>
                    <div style={{ padding: '0.75rem', backgroundColor: '#e8f5e9', borderRadius: '8px', textAlign: 'center' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#2e7d32' }}>
                            {stats.by_status?.response_sent || 0}
                        </div>
                        <div style={{ fontSize: '0.7rem', color: '#1b5e20' }}>Sent</div>
                    </div>
                    {/* Priority row */}
                    <div style={{ padding: '0.75rem', backgroundColor: '#ffebee', borderRadius: '8px', textAlign: 'center' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#c62828' }}>
                            {stats.by_priority?.high || 0}
                        </div>
                        <div style={{ fontSize: '0.7rem', color: '#b71c1c' }}>High Priority</div>
                    </div>
                    <div style={{ padding: '0.75rem', backgroundColor: '#fff8e1', borderRadius: '8px', textAlign: 'center' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#f9a825' }}>
                            {stats.by_priority?.medium || 0}
                        </div>
                        <div style={{ fontSize: '0.7rem', color: '#f57f17' }}>Medium Priority</div>
                    </div>
                    <div style={{ padding: '0.75rem', backgroundColor: '#e8f5e9', borderRadius: '8px', textAlign: 'center' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#388e3c' }}>
                            {stats.by_priority?.low || 0}
                        </div>
                        <div style={{ fontSize: '0.7rem', color: '#2e7d32' }}>Low Priority</div>
                    </div>
                    <div style={{ padding: '0.75rem', backgroundColor: '#f5f5f5', borderRadius: '8px', textAlign: 'center' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#333' }}>
                            {stats.total}
                        </div>
                        <div style={{ fontSize: '0.7rem', color: '#666' }}>Total</div>
                    </div>
                </div>
            )}

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

            {/* Enquiry List with AI Quick View */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {filteredEnquiries.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
                        No enquiries found
                    </div>
                ) : (
                    filteredEnquiries.map((enquiry) => {
                        const qv = enquiry.ai_quick_view;
                        const isSelected = selectedEnquiry?.id === enquiry.id;

                        return (
                            <div
                                key={enquiry.id}
                                onClick={() => setSelectedEnquiry(isSelected ? null : enquiry)}
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
                                        <span style={{ fontWeight: '600', color: '#2d2d2d' }}>
                                            {enquiry.customer.name}
                                        </span>
                                        {enquiry.customer.company && (
                                            <span style={{ color: '#666', fontSize: '0.875rem' }}> • {enquiry.customer.company}</span>
                                        )}
                                    </div>
                                    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                                        {qv && getUrgencyBar(qv.urgency)}
                                        {qv && (
                                            <span style={{
                                                padding: '0.25rem 0.5rem',
                                                borderRadius: '4px',
                                                fontSize: '0.7rem',
                                                fontWeight: '500',
                                                backgroundColor: '#e3f2fd',
                                                color: '#1565c0'
                                            }}>
                                                {qv.type.toUpperCase().replace('_', ' ')}
                                            </span>
                                        )}
                                        {qv && (
                                            <span style={{
                                                padding: '0.25rem 0.5rem',
                                                borderRadius: '4px',
                                                fontSize: '0.7rem',
                                                fontWeight: '600',
                                                ...getPriorityColor(qv.priority)
                                            }}>
                                                {qv.priority.toUpperCase()}
                                            </span>
                                        )}
                                        <span style={{
                                            padding: '0.25rem 0.5rem',
                                            borderRadius: '4px',
                                            fontSize: '0.7rem',
                                            fontWeight: '500',
                                            ...getStatusColor(enquiry.status)
                                        }}>
                                            {enquiry.status.replace('_', ' ').toUpperCase()}
                                        </span>
                                    </div>
                                </div>

                                {/* AI Summary One-Liner */}
                                <div style={{
                                    padding: '0.5rem 0.75rem',
                                    backgroundColor: '#f8f9fa',
                                    borderRadius: '4px',
                                    borderLeft: '3px solid #952825',
                                    marginBottom: '0.5rem'
                                }}>
                                    <span style={{ fontSize: '0.75rem', color: '#952825', fontWeight: '500' }}>AI SUMMARY: </span>
                                    <span style={{ fontSize: '0.875rem', color: '#333' }}>
                                        {qv?.one_liner || enquiry.subject || 'Pending analysis...'}
                                    </span>
                                </div>

                                {/* Quick Info Row */}
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', color: '#888' }}>
                                    <div style={{ display: 'flex', gap: '1rem' }}>
                                        <span>{enquiry.id}</span>
                                        <span>{formatDate(enquiry.created_at)}</span>
                                        {enquiry.assigned_to && <span>→ {enquiry.assigned_to}</span>}
                                    </div>
                                    {qv?.products && qv.products.length > 0 && (
                                        <div style={{ display: 'flex', gap: '0.25rem' }}>
                                            {qv.products.slice(0, 3).map(p => (
                                                <span key={p} style={{
                                                    padding: '0.15rem 0.5rem',
                                                    backgroundColor: '#e8f5e9',
                                                    color: '#2e7d32',
                                                    borderRadius: '10px',
                                                    fontSize: '0.7rem',
                                                    fontWeight: '500'
                                                }}>
                                                    {p}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                </div>

                                {/* Actions Needed Badges */}
                                {qv?.actions_needed && (
                                    <div style={{ marginTop: '0.5rem', display: 'flex', gap: '0.5rem' }}>
                                        {qv.actions_needed.technical_review && (
                                            <span style={{
                                                padding: '0.2rem 0.5rem',
                                                backgroundColor: '#fff3e0',
                                                color: '#e65100',
                                                borderRadius: '4px',
                                                fontSize: '0.65rem',
                                                fontWeight: '500'
                                            }}>
                                                TECHNICAL REVIEW
                                            </span>
                                        )}
                                        {qv.actions_needed.pricing && (
                                            <span style={{
                                                padding: '0.2rem 0.5rem',
                                                backgroundColor: '#e8f5e9',
                                                color: '#2e7d32',
                                                borderRadius: '4px',
                                                fontSize: '0.65rem',
                                                fontWeight: '500'
                                            }}>
                                                PRICING NEEDED
                                            </span>
                                        )}
                                        {qv.actions_needed.samples && (
                                            <span style={{
                                                padding: '0.2rem 0.5rem',
                                                backgroundColor: '#f3e5f5',
                                                color: '#7b1fa2',
                                                borderRadius: '4px',
                                                fontSize: '0.65rem',
                                                fontWeight: '500'
                                            }}>
                                                SAMPLES REQUESTED
                                            </span>
                                        )}
                                    </div>
                                )}

                                {/* Expanded Details */}
                                {isSelected && (
                                    <div style={{
                                        marginTop: '1rem',
                                        paddingTop: '1rem',
                                        borderTop: '1px solid #e5e5e5'
                                    }}>
                                        {/* Toggle Raw Message */}
                                        <div style={{ marginBottom: '1rem' }}>
                                            <button
                                                onClick={(e) => { e.stopPropagation(); setShowRawMessage(!showRawMessage); }}
                                                style={{
                                                    padding: '0.35rem 0.75rem',
                                                    backgroundColor: showRawMessage ? '#952825' : '#f5f5f5',
                                                    color: showRawMessage ? 'white' : '#666',
                                                    border: 'none',
                                                    borderRadius: '4px',
                                                    cursor: 'pointer',
                                                    fontSize: '0.75rem'
                                                }}
                                            >
                                                {showRawMessage ? 'Hide' : 'Show'} Raw Message
                                            </button>
                                        </div>

                                        {showRawMessage && enquiry.raw_message && (
                                            <div style={{
                                                padding: '0.75rem',
                                                backgroundColor: '#fafafa',
                                                borderRadius: '4px',
                                                marginBottom: '1rem',
                                                fontSize: '0.85rem',
                                                color: '#444',
                                                whiteSpace: 'pre-wrap'
                                            }}>
                                                {enquiry.raw_message}
                                            </div>
                                        )}

                                        {/* Main Ask */}
                                        {qv?.main_ask && (
                                            <div style={{ marginBottom: '0.75rem' }}>
                                                <strong style={{ fontSize: '0.8rem', color: '#666' }}>Main Request:</strong>
                                                <p style={{ margin: '0.25rem 0 0', fontSize: '0.875rem', color: '#333' }}>
                                                    {qv.main_ask}
                                                </p>
                                            </div>
                                        )}

                                        {/* AI Analysis Details */}
                                        {enquiry.ai_analysis && (
                                            <div style={{
                                                marginBottom: '0.75rem',
                                                padding: '0.75rem',
                                                backgroundColor: '#f3e5f5',
                                                borderRadius: '6px',
                                                border: '1px solid #ce93d8'
                                            }}>
                                                <strong style={{ fontSize: '0.8rem', color: '#7b1fa2' }}>AI Analysis:</strong>
                                                <p style={{ margin: '0.25rem 0 0.5rem', fontSize: '0.85rem', color: '#333' }}>
                                                    {enquiry.ai_analysis.summary || enquiry.ai_analysis.one_liner}
                                                </p>
                                                {enquiry.ai_analysis.customer_intent && (
                                                    <p style={{ margin: '0 0 0.25rem', fontSize: '0.8rem', color: '#555' }}>
                                                        <strong>Intent:</strong> {enquiry.ai_analysis.customer_intent}
                                                    </p>
                                                )}
                                                {enquiry.ai_analysis.requirements && (
                                                    <div style={{ fontSize: '0.75rem', color: '#555', marginTop: '0.25rem' }}>
                                                        {enquiry.ai_analysis.requirements.industry && <div>Industry: {enquiry.ai_analysis.requirements.industry}</div>}
                                                        {enquiry.ai_analysis.requirements.application && <div>Application: {enquiry.ai_analysis.requirements.application}</div>}
                                                        {enquiry.ai_analysis.requirements.operating_temperature && <div>Temp: {enquiry.ai_analysis.requirements.operating_temperature}</div>}
                                                        {enquiry.ai_analysis.requirements.operating_pressure && <div>Pressure: {enquiry.ai_analysis.requirements.operating_pressure}</div>}
                                                        {enquiry.ai_analysis.requirements.media_handled && <div>Media: {enquiry.ai_analysis.requirements.media_handled}</div>}
                                                        {enquiry.ai_analysis.requirements.quantity && <div>Qty: {enquiry.ai_analysis.requirements.quantity}</div>}
                                                        {enquiry.ai_analysis.requirements.delivery_urgency && <div>Delivery: {enquiry.ai_analysis.requirements.delivery_urgency}</div>}
                                                    </div>
                                                )}
                                                {enquiry.ai_analysis.key_points && enquiry.ai_analysis.key_points.length > 0 && (
                                                    <div style={{ marginTop: '0.4rem' }}>
                                                        <strong style={{ fontSize: '0.75rem', color: '#7b1fa2' }}>Key Points:</strong>
                                                        <ul style={{ margin: '0.15rem 0 0', paddingLeft: '1.2rem', fontSize: '0.75rem', color: '#555' }}>
                                                            {enquiry.ai_analysis.key_points.map((pt: string, i: number) => (
                                                                <li key={i}>{pt}</li>
                                                            ))}
                                                        </ul>
                                                    </div>
                                                )}
                                                {enquiry.ai_analysis.recommended_actions && enquiry.ai_analysis.recommended_actions.length > 0 && (
                                                    <div style={{ marginTop: '0.4rem' }}>
                                                        <strong style={{ fontSize: '0.75rem', color: '#7b1fa2' }}>Recommended Actions:</strong>
                                                        <ul style={{ margin: '0.15rem 0 0', paddingLeft: '1.2rem', fontSize: '0.75rem', color: '#555' }}>
                                                            {enquiry.ai_analysis.recommended_actions.map((act: string, i: number) => (
                                                                <li key={i}>{act}</li>
                                                            ))}
                                                        </ul>
                                                    </div>
                                                )}
                                                {enquiry.ai_analysis.suggested_products && enquiry.ai_analysis.suggested_products.length > 0 && (
                                                    <div style={{ marginTop: '0.4rem', display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
                                                        <strong style={{ fontSize: '0.75rem', color: '#7b1fa2', marginRight: '0.25rem' }}>Products:</strong>
                                                        {enquiry.ai_analysis.suggested_products.map((p: string) => (
                                                            <span key={p} style={{
                                                                padding: '0.1rem 0.4rem',
                                                                backgroundColor: '#e8f5e9',
                                                                color: '#2e7d32',
                                                                borderRadius: '8px',
                                                                fontSize: '0.7rem'
                                                            }}>{p}</span>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>
                                        )}

                                        {/* Contact Info */}
                                        <div style={{ marginBottom: '0.75rem' }}>
                                            <strong style={{ fontSize: '0.8rem', color: '#666' }}>Contact:</strong>
                                            <span style={{ marginLeft: '0.5rem', fontSize: '0.875rem' }}>
                                                {enquiry.customer.email}
                                            </span>
                                        </div>

                                        {/* Actions */}
                                        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem', flexWrap: 'wrap' }}>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    generateAIResponse(enquiry.id);
                                                }}
                                                disabled={generatingResponse === enquiry.id}
                                                style={{
                                                    padding: '0.5rem 1rem',
                                                    backgroundColor: generatingResponse === enquiry.id ? '#ccc' : '#952825',
                                                    color: 'white',
                                                    border: 'none',
                                                    borderRadius: '4px',
                                                    cursor: generatingResponse === enquiry.id ? 'wait' : 'pointer',
                                                    fontSize: '0.8rem'
                                                }}
                                            >
                                                {generatingResponse === enquiry.id ? 'Generating...' : 'Generate AI Response'}
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    window.location.href = `mailto:${enquiry.customer.email}?subject=Re: ${enquiry.subject || enquiry.id}`;
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
                                                Send Email
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    reanalyzeEnquiry(enquiry.id);
                                                }}
                                                disabled={reanalyzing === enquiry.id}
                                                style={{
                                                    padding: '0.5rem 1rem',
                                                    backgroundColor: reanalyzing === enquiry.id ? '#ccc' : '#f5f5f5',
                                                    color: '#666',
                                                    border: '1px solid #ddd',
                                                    borderRadius: '4px',
                                                    cursor: reanalyzing === enquiry.id ? 'wait' : 'pointer',
                                                    fontSize: '0.8rem'
                                                }}
                                            >
                                                {reanalyzing === enquiry.id ? 'Analyzing...' : 'Re-analyze'}
                                            </button>
                                            {enquiry.status !== 'response_sent' && (
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        if (confirm('Mark this enquiry as sent?')) {
                                                            markAsSent(enquiry.id);
                                                        }
                                                    }}
                                                    disabled={markingSent === enquiry.id}
                                                    style={{
                                                        padding: '0.5rem 1rem',
                                                        backgroundColor: markingSent === enquiry.id ? '#ccc' : '#4caf50',
                                                        color: 'white',
                                                        border: 'none',
                                                        borderRadius: '4px',
                                                        cursor: markingSent === enquiry.id ? 'wait' : 'pointer',
                                                        fontSize: '0.8rem'
                                                    }}
                                                >
                                                    {markingSent === enquiry.id ? 'Updating...' : 'Mark as Sent'}
                                                </button>
                                            )}
                                        </div>

                                        {/* Suggested Response */}
                                        {suggestedResponses[enquiry.id] && isSelected && (
                                            <div style={{
                                                marginTop: '1rem',
                                                padding: '1rem',
                                                backgroundColor: '#f8f9fa',
                                                borderRadius: '4px',
                                                borderLeft: '3px solid #4caf50'
                                            }}>
                                                <strong style={{ fontSize: '0.8rem', color: '#2e7d32' }}>AI Suggested Response:</strong>
                                                <textarea
                                                    value={suggestedResponses[enquiry.id] || ''}
                                                    onChange={(e) => setSuggestedResponses(prev => ({ ...prev, [enquiry.id]: e.target.value }))}
                                                    onClick={(e) => e.stopPropagation()}
                                                    style={{
                                                        width: '100%',
                                                        minHeight: '150px',
                                                        marginTop: '0.5rem',
                                                        padding: '0.75rem',
                                                        border: '1px solid #ddd',
                                                        borderRadius: '4px',
                                                        fontSize: '0.85rem',
                                                        resize: 'vertical'
                                                    }}
                                                />
                                                <div style={{ marginTop: '0.5rem', display: 'flex', gap: '0.5rem' }}>
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            navigator.clipboard.writeText(suggestedResponses[enquiry.id] || '');
                                                            alert('Response copied to clipboard!');
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
                                                        Copy Response
                                                    </button>
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            window.location.href = `mailto:${enquiry.customer.email}?subject=Re: ${enquiry.subject || enquiry.id}&body=${encodeURIComponent(suggestedResponses[enquiry.id] || '')}`;
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
                                                        Send via Email
                                                    </button>
                                                </div>
                                            </div>
                                        )}
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
