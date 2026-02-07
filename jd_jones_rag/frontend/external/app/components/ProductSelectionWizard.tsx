'use client';

import { useState } from 'react';

interface SelectionStep {
    id: string;
    question: string;
    options?: string[];
    type: 'select' | 'text' | 'multiselect';
    field: string;
}

interface ProductRecommendation {
    product_id: string;
    product_name: string;
    score: number;
    confidence?: string;
    specifications: Record<string, string | string[]>;
    applications?: string[];
    certifications?: string[];
    match_reasons?: string[];
    warnings?: string[];
    type?: string;
}


interface SelectionState {
    sessionId: string | null;
    currentStep: number;
    answers: Record<string, string | string[]>;
    recommendations: ProductRecommendation[];
    isLoading: boolean;
    error: string | null;
    isComplete: boolean;
}

// Quotation modal state
interface QuoteModalState {
    isOpen: boolean;
    product: ProductRecommendation | null;
    mode: 'choose' | 'specific' | 'generic';
    customerInfo: {
        name: string;
        email: string;
        company: string;
        phone: string;
    };
    specificDetails: {
        quantity: string;
        size_od: string;
        size_id: string;
        size_th: string;
        dimension_unit: 'mm' | 'inch';
        style: string;
        materialGrade: string;
        colour: string;
        unit: string;
        rings_per_set: string;
        specificRequirements: string;
        notes: string;
    };
    genericMessage: string;
    isSubmitting: boolean;
    submitSuccess: boolean;
}

const SELECTION_STEPS: SelectionStep[] = [
    {
        id: 'application',
        question: 'What type of application is this for?',
        options: [
            'Valve Packing',
            'Pump Sealing',
            'Flange Gasket',
            'Heat Exchanger',
            'Other Industrial'
        ],
        type: 'select',
        field: 'application_type'
    },
    {
        id: 'industry',
        question: 'What industry sector?',
        options: [
            'Oil & Gas / Refinery',
            'Chemical Processing',
            'Power Generation',
            'Pharmaceutical',
            'Food & Beverage',
            'Pulp & Paper',
            'Water Treatment',
            'Other'
        ],
        type: 'select',
        field: 'industry'
    },
    {
        id: 'temperature',
        question: 'What is the operating temperature range?',
        options: [
            'Cryogenic (-200°C to -40°C)',
            'Low Temperature (-40°C to 50°C)',
            'Ambient (50°C to 200°C)',
            'High Temperature (200°C to 400°C)',
            'Very High Temperature (400°C+)'
        ],
        type: 'select',
        field: 'temperature_range'
    },
    {
        id: 'pressure',
        question: 'What is the operating pressure?',
        options: [
            'Low (0-10 bar)',
            'Medium (10-50 bar)',
            'High (50-200 bar)',
            'Very High (200+ bar)'
        ],
        type: 'select',
        field: 'pressure_range'
    },
    {
        id: 'media',
        question: 'What media will be sealed?',
        options: [
            'Steam',
            'Water',
            'Hydrocarbons (Oil/Gas)',
            'Acids/Alkalis',
            'Solvents',
            'Cryogenic Fluids',
            'Other'
        ],
        type: 'select',
        field: 'media_type'
    },
    {
        id: 'certifications',
        question: 'Are any certifications required? (Select all that apply)',
        options: [
            'API 622 (Fugitive Emissions)',
            'API 589 (Fire Safe)',
            'ISO 15848 (Emissions)',
            'FDA Approved',
            'TA-Luft Compliant',
            'None Required'
        ],
        type: 'multiselect',
        field: 'required_certifications'
    }
];

export default function ProductSelectionWizard() {
    const [state, setState] = useState<SelectionState>({
        sessionId: null,
        currentStep: 0,
        answers: {},
        recommendations: [],
        isLoading: false,
        error: null,
        isComplete: false
    });

    const [selectedOptions, setSelectedOptions] = useState<string[]>([]);

    const currentStepData = SELECTION_STEPS[state.currentStep];
    const isLastStep = state.currentStep === SELECTION_STEPS.length - 1;

    const handleOptionSelect = async (option: string) => {
        if (currentStepData.type === 'multiselect') {
            setSelectedOptions(prev =>
                prev.includes(option)
                    ? prev.filter(o => o !== option)
                    : [...prev, option]
            );
            return;
        }

        setState(prev => ({
            ...prev,
            answers: { ...prev.answers, [currentStepData.field]: option },
            isLoading: isLastStep
        }));

        if (isLastStep) {
            await submitSelection({ ...state.answers, [currentStepData.field]: option });
        } else {
            setState(prev => ({ ...prev, currentStep: prev.currentStep + 1 }));
        }
    };

    const handleMultiSelectSubmit = async () => {
        // Immediately set loading and update answers
        const updatedAnswers = { ...state.answers, [currentStepData.field]: selectedOptions };

        setState(prev => ({
            ...prev,
            answers: updatedAnswers,
            isLoading: true,
            error: null
        }));

        // Submit with updated answers directly (not from state which may be stale)
        await submitSelection(updatedAnswers);
    };

    const submitSelection = async (answers: Record<string, string | string[]>) => {
        try {
            const response = await fetch('/api/agentic/product-selection', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    application_type: answers.application_type,
                    industry: answers.industry,
                    temperature_range: answers.temperature_range,
                    pressure_range: answers.pressure_range,
                    media_type: answers.media_type,
                    required_certifications: answers.required_certifications || [],
                    session_id: state.sessionId
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || data.error || 'Failed to get recommendations');
            }

            setState(prev => ({
                ...prev,
                sessionId: data.session_id,
                recommendations: data.recommendations || [],
                isLoading: false,
                isComplete: true
            }));
        } catch (err) {
            setState(prev => ({
                ...prev,
                error: err instanceof Error ? err.message : 'An error occurred',
                isLoading: false
            }));
        }
    };

    const handleBack = () => {
        if (state.currentStep > 0) {
            setState(prev => ({ ...prev, currentStep: prev.currentStep - 1 }));
            setSelectedOptions([]);
        }
    };

    const handleReset = () => {
        setState({
            sessionId: null,
            currentStep: 0,
            answers: {},
            recommendations: [],
            isLoading: false,
            error: null,
            isComplete: false
        });
        setSelectedOptions([]);
    };

    // Quote modal state
    const [quoteModal, setQuoteModal] = useState<QuoteModalState>({
        isOpen: false,
        product: null,
        mode: 'choose',
        customerInfo: { name: '', email: '', company: '', phone: '' },
        specificDetails: { quantity: '', size_od: '', size_id: '', size_th: '', dimension_unit: 'mm', style: '', materialGrade: '', colour: '', unit: 'Nos.', rings_per_set: '', specificRequirements: '', notes: '' },
        genericMessage: '',
        isSubmitting: false,
        submitSuccess: false
    });

    const handleRequestQuote = (product: ProductRecommendation) => {
        // Open modal with dual-path options
        setQuoteModal({
            isOpen: true,
            product,
            mode: 'choose',
            customerInfo: { name: '', email: '', company: '', phone: '' },
            specificDetails: { quantity: '', size_od: '', size_id: '', size_th: '', dimension_unit: 'mm', style: '', materialGrade: '', colour: '', unit: 'Nos.', rings_per_set: '', specificRequirements: '', notes: '' },
            genericMessage: '',
            isSubmitting: false,
            submitSuccess: false
        });
    };

    const closeQuoteModal = () => {
        setQuoteModal(prev => ({ ...prev, isOpen: false, submitSuccess: false }));
    };

    const submitQuote = async (isGeneric: boolean) => {
        if (!quoteModal.product) return;

        setQuoteModal(prev => ({ ...prev, isSubmitting: true }));

        try {
            // Use local Next.js API proxy routes to avoid CORS issues
            const endpoint = isGeneric
                ? '/api/v1/quotations/external/submit-generic'
                : '/api/v1/quotations/external/submit-specific';

            const payload = isGeneric ? {
                customer: {
                    name: quoteModal.customerInfo.name,
                    email: quoteModal.customerInfo.email,
                    company: quoteModal.customerInfo.company || 'Not specified',
                    phone: quoteModal.customerInfo.phone || undefined
                },
                product_code: quoteModal.product?.product_id,  // Always include the selected product code
                product_name: quoteModal.product?.product_name,  // Always include the selected product name
                message: quoteModal.genericMessage || `I need a quote for ${quoteModal.product?.product_id} - ${quoteModal.product?.product_name}. Please suggest specifications.`
            } : {
                customer: {
                    name: quoteModal.customerInfo.name,
                    email: quoteModal.customerInfo.email,
                    company: quoteModal.customerInfo.company || 'Not specified',
                    phone: quoteModal.customerInfo.phone || undefined
                },
                line_items: [{
                    product_code: quoteModal.product.product_id,
                    product_name: quoteModal.product.product_name,
                    quantity: parseInt(quoteModal.specificDetails.quantity) || 1,
                    size_od: quoteModal.specificDetails.size_od ? parseFloat(quoteModal.specificDetails.size_od) : null,
                    size_id: quoteModal.specificDetails.size_id ? parseFloat(quoteModal.specificDetails.size_id) : null,
                    size_th: quoteModal.specificDetails.size_th ? parseFloat(quoteModal.specificDetails.size_th) : null,
                    dimension_unit: quoteModal.specificDetails.dimension_unit || 'mm',
                    style: quoteModal.specificDetails.style || null,
                    material_grade: quoteModal.specificDetails.materialGrade || null,
                    colour: quoteModal.specificDetails.colour || null,
                    unit: quoteModal.specificDetails.unit || 'Nos.',
                    rings_per_set: quoteModal.specificDetails.rings_per_set ? parseInt(quoteModal.specificDetails.rings_per_set) : null,
                    specific_requirements: quoteModal.specificDetails.specificRequirements || null,
                    notes: quoteModal.specificDetails.notes || null,
                    is_ai_suggested: false
                }]
            };

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error('Failed to submit');

            setQuoteModal(prev => ({ ...prev, isSubmitting: false, submitSuccess: true }));
        } catch {
            setQuoteModal(prev => ({ ...prev, isSubmitting: false }));
            alert('Failed to submit quote request. Please try again.');
        }
    };

    const handleGenerateDatasheet = async (product: ProductRecommendation) => {
        try {
            const response = await fetch('/api/documents/datasheet', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    product_code: product.product_id,
                    product_name: product.product_name,
                    include_certifications: true
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to generate datasheet');
            }

            // Open the download URL
            window.open(`/api${data.download_url}`, '_blank');
        } catch (err) {
            alert(err instanceof Error ? err.message : 'Failed to generate datasheet');
        }
    };

    // Render recommendations view
    if (state.isComplete && state.recommendations.length > 0) {
        return (
            <>
                <style>{`
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
            `}</style>
                <div style={{ padding: '1.5rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                        <h2 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#2d2d2d', margin: 0 }}>
                            Recommended Products
                        </h2>
                        <button
                            onClick={handleReset}
                            style={{
                                padding: '0.5rem 1rem',
                                backgroundColor: 'transparent',
                                border: '1px solid #952825',
                                borderRadius: '4px',
                                color: '#952825',
                                cursor: 'pointer',
                                fontSize: '0.875rem'
                            }}
                        >
                            Start Over
                        </button>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {state.recommendations.map((product: ProductRecommendation, index: number) => (
                            <div
                                key={product.product_id}
                                style={{
                                    padding: '1.25rem',
                                    backgroundColor: 'white',
                                    border: '1px solid #e5e5e5',
                                    borderRadius: '8px',
                                    boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
                                }}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                    <div>
                                        <span style={{
                                            display: 'inline-block',
                                            padding: '0.25rem 0.5rem',
                                            backgroundColor: index === 0 ? '#952825' : '#666',
                                            color: 'white',
                                            borderRadius: '4px',
                                            fontSize: '0.7rem',
                                            fontWeight: '600',
                                            marginBottom: '0.5rem'
                                        }}>
                                            {index === 0 ? 'TOP MATCH' : `#${index + 1}`}
                                        </span>
                                        <h3 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#2d2d2d', margin: '0.25rem 0' }}>
                                            {product.product_id}
                                        </h3>
                                        <p style={{ color: '#666', margin: '0 0 0.75rem 0', fontSize: '0.9rem' }}>
                                            {product.product_name}
                                        </p>
                                    </div>
                                    <div style={{
                                        padding: '0.25rem 0.5rem',
                                        backgroundColor: '#e8f5e9',
                                        color: '#2e7d32',
                                        borderRadius: '4px',
                                        fontSize: '0.8rem',
                                        fontWeight: '500'
                                    }}>
                                        {product.score}% Match
                                    </div>
                                </div>

                                {/* Match Reasons */}
                                {product.match_reasons && product.match_reasons.length > 0 && (
                                    <p style={{ color: '#555', fontSize: '0.875rem', marginBottom: '1rem', lineHeight: '1.5' }}>
                                        {product.match_reasons.join(', ')}
                                    </p>
                                )}

                                {/* Specifications */}
                                {Object.keys(product.specifications).length > 0 && (
                                    <div style={{ marginBottom: '0.75rem' }}>
                                        <strong style={{ fontSize: '0.8rem', color: '#666' }}>Specifications:</strong>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.25rem' }}>
                                            {Object.entries(product.specifications).slice(0, 4).map(([key, value]: [string, unknown]) => (
                                                <span key={key} style={{
                                                    padding: '0.25rem 0.5rem',
                                                    backgroundColor: '#f5f5f5',
                                                    borderRadius: '4px',
                                                    fontSize: '0.75rem',
                                                    color: '#555'
                                                }}>
                                                    {key}: {Array.isArray(value) ? value.join(', ') : String(value)}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Certifications */}
                                {product.certifications && product.certifications.length > 0 && (
                                    <div style={{ marginBottom: '1rem' }}>
                                        <strong style={{ fontSize: '0.8rem', color: '#666' }}>Certifications:</strong>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.25rem' }}>
                                            {product.certifications.map((cert: string) => (
                                                <span key={cert} style={{
                                                    padding: '0.25rem 0.5rem',
                                                    backgroundColor: '#e3f2fd',
                                                    borderRadius: '4px',
                                                    fontSize: '0.75rem',
                                                    color: '#1565c0'
                                                }}>
                                                    {cert}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Actions */}
                                <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
                                    <button
                                        onClick={() => handleRequestQuote(product)}
                                        style={{
                                            padding: '0.5rem 1rem',
                                            backgroundColor: '#952825',
                                            border: 'none',
                                            borderRadius: '4px',
                                            color: 'white',
                                            cursor: 'pointer',
                                            fontSize: '0.875rem',
                                            fontWeight: '500'
                                        }}
                                    >
                                        Request Quote
                                    </button>
                                    <button
                                        onClick={() => handleGenerateDatasheet(product)}
                                        style={{
                                            padding: '0.5rem 1rem',
                                            backgroundColor: 'white',
                                            border: '1px solid #952825',
                                            borderRadius: '4px',
                                            color: '#952825',
                                            cursor: 'pointer',
                                            fontSize: '0.875rem'
                                        }}
                                    >
                                        Download Datasheet
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Quote Modal */}
                {quoteModal.isOpen && (
                    <div style={{
                        position: 'fixed',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        backgroundColor: 'rgba(0,0,0,0.5)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        zIndex: 1000
                    }}>
                        <div style={{
                            backgroundColor: 'white',
                            borderRadius: '12px',
                            padding: '2rem',
                            maxWidth: '500px',
                            width: '90%',
                            maxHeight: '90vh',
                            overflowY: 'auto'
                        }}>
                            {/* Success State */}
                            {quoteModal.submitSuccess ? (
                                <div style={{ textAlign: 'center' }}>
                                    <div style={{ fontSize: '48px', marginBottom: '1rem' }}></div>
                                    <h3 style={{ color: '#2e7d32', marginBottom: '0.5rem' }}>Quote Request Submitted!</h3>
                                    <p style={{ color: '#666', marginBottom: '1.5rem' }}>
                                        Our sales team will contact you within 24 hours.
                                    </p>
                                    <button
                                        onClick={closeQuoteModal}
                                        style={{
                                            padding: '0.75rem 1.5rem',
                                            backgroundColor: '#952825',
                                            color: 'white',
                                            border: 'none',
                                            borderRadius: '4px',
                                            cursor: 'pointer'
                                        }}
                                    >
                                        Close
                                    </button>
                                </div>
                            ) : quoteModal.mode === 'choose' ? (
                                /* Mode Selection */
                                <div>
                                    <h3 style={{ marginBottom: '0.5rem', color: '#2d2d2d' }}>Request a Quote</h3>
                                    <p style={{ color: '#666', fontSize: '0.875rem', marginBottom: '1.5rem' }}>
                                        For: <strong>{quoteModal.product?.product_id}</strong> - {quoteModal.product?.product_name}
                                    </p>

                                    <div
                                        onClick={() => setQuoteModal(prev => ({ ...prev, mode: 'specific' }))}
                                        style={{
                                            padding: '1.25rem',
                                            border: '2px solid #e5e5e5',
                                            borderRadius: '8px',
                                            marginBottom: '1rem',
                                            cursor: 'pointer',
                                            transition: 'all 0.2s'
                                        }}
                                        onMouseEnter={(e) => e.currentTarget.style.borderColor = '#952825'}
                                        onMouseLeave={(e) => e.currentTarget.style.borderColor = '#e5e5e5'}
                                    >
                                        <h4 style={{ margin: 0, color: '#952825' }}> I Have Specific Requirements</h4>
                                        <p style={{ margin: '0.5rem 0 0', fontSize: '0.875rem', color: '#666' }}>
                                            I know the size, quantity, and specifications I need.
                                        </p>
                                    </div>

                                    <div
                                        onClick={() => setQuoteModal(prev => ({ ...prev, mode: 'generic' }))}
                                        style={{
                                            padding: '1.25rem',
                                            border: '2px solid #e5e5e5',
                                            borderRadius: '8px',
                                            cursor: 'pointer',
                                            transition: 'all 0.2s'
                                        }}
                                        onMouseEnter={(e) => e.currentTarget.style.borderColor = '#952825'}
                                        onMouseLeave={(e) => e.currentTarget.style.borderColor = '#e5e5e5'}
                                    >
                                        <h4 style={{ margin: 0, color: '#952825' }}> I Need Help with Requirements</h4>
                                        <p style={{ margin: '0.5rem 0 0', fontSize: '0.875rem', color: '#666' }}>
                                            Let our AI analyze my application and suggest the right specifications.
                                        </p>
                                    </div>

                                    <button
                                        onClick={closeQuoteModal}
                                        style={{
                                            marginTop: '1rem',
                                            padding: '0.5rem',
                                            backgroundColor: 'transparent',
                                            border: 'none',
                                            color: '#666',
                                            cursor: 'pointer',
                                            width: '100%'
                                        }}
                                    >
                                        Cancel
                                    </button>
                                </div>
                            ) : (
                                /* Form */
                                <div>
                                    <button
                                        onClick={() => setQuoteModal(prev => ({ ...prev, mode: 'choose' }))}
                                        style={{ background: 'none', border: 'none', color: '#666', cursor: 'pointer', marginBottom: '1rem' }}
                                    >
                                        ← Back
                                    </button>

                                    <h3 style={{ marginBottom: '1rem' }}>
                                        {quoteModal.mode === 'specific' ? ' Specific Requirements' : ' AI-Assisted Quote'}
                                    </h3>

                                    {/* Customer Info */}
                                    <div style={{ marginBottom: '1rem' }}>
                                        <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: '500', fontSize: '0.875rem' }}>Name *</label>
                                        <input
                                            type="text"
                                            value={quoteModal.customerInfo.name}
                                            onChange={e => setQuoteModal(prev => ({ ...prev, customerInfo: { ...prev.customerInfo, name: e.target.value } }))}
                                            style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px' }}
                                            required
                                        />
                                    </div>
                                    <div style={{ marginBottom: '1rem' }}>
                                        <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: '500', fontSize: '0.875rem' }}>Email *</label>
                                        <input
                                            type="email"
                                            value={quoteModal.customerInfo.email}
                                            onChange={e => setQuoteModal(prev => ({ ...prev, customerInfo: { ...prev.customerInfo, email: e.target.value } }))}
                                            style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px' }}
                                            required
                                        />
                                    </div>
                                    <div style={{ marginBottom: '1rem' }}>
                                        <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: '500', fontSize: '0.875rem' }}>Company</label>
                                        <input
                                            type="text"
                                            value={quoteModal.customerInfo.company}
                                            onChange={e => setQuoteModal(prev => ({ ...prev, customerInfo: { ...prev.customerInfo, company: e.target.value } }))}
                                            style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px' }}
                                        />
                                    </div>

                                    {quoteModal.mode === 'specific' ? (
                                        <>
                                            <div style={{ marginBottom: '1rem' }}>
                                                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', fontSize: '0.875rem' }}>Dimensions (OD × ID × TH)</label>
                                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr auto', gap: '0.5rem', alignItems: 'end' }}>
                                                    <div>
                                                        <label style={{ display: 'block', marginBottom: '0.15rem', fontSize: '0.75rem', color: '#666' }}>OD</label>
                                                        <input
                                                            type="number"
                                                            step="any"
                                                            min="0"
                                                            value={quoteModal.specificDetails.size_od}
                                                            onChange={e => setQuoteModal(prev => ({ ...prev, specificDetails: { ...prev.specificDetails, size_od: e.target.value } }))}
                                                            placeholder="OD"
                                                            style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px' }}
                                                        />
                                                    </div>
                                                    <div>
                                                        <label style={{ display: 'block', marginBottom: '0.15rem', fontSize: '0.75rem', color: '#666' }}>ID</label>
                                                        <input
                                                            type="number"
                                                            step="any"
                                                            min="0"
                                                            value={quoteModal.specificDetails.size_id}
                                                            onChange={e => setQuoteModal(prev => ({ ...prev, specificDetails: { ...prev.specificDetails, size_id: e.target.value } }))}
                                                            placeholder="ID"
                                                            style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px' }}
                                                        />
                                                    </div>
                                                    <div>
                                                        <label style={{ display: 'block', marginBottom: '0.15rem', fontSize: '0.75rem', color: '#666' }}>TH</label>
                                                        <input
                                                            type="number"
                                                            step="any"
                                                            min="0"
                                                            value={quoteModal.specificDetails.size_th}
                                                            onChange={e => setQuoteModal(prev => ({ ...prev, specificDetails: { ...prev.specificDetails, size_th: e.target.value } }))}
                                                            placeholder="TH"
                                                            style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px' }}
                                                        />
                                                    </div>
                                                    <div>
                                                        <label style={{ display: 'block', marginBottom: '0.15rem', fontSize: '0.75rem', color: '#666' }}>Unit</label>
                                                        <select
                                                            value={quoteModal.specificDetails.dimension_unit}
                                                            onChange={e => setQuoteModal(prev => ({ ...prev, specificDetails: { ...prev.specificDetails, dimension_unit: e.target.value as 'mm' | 'inch' } }))}
                                                            style={{ padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px', minWidth: '65px' }}
                                                        >
                                                            <option value="mm">mm</option>
                                                            <option value="inch">inch</option>
                                                        </select>
                                                    </div>
                                                </div>
                                            </div>
                                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                                                <div>
                                                    <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: '500', fontSize: '0.875rem' }}>Style</label>
                                                    <select
                                                        value={quoteModal.specificDetails.style}
                                                        onChange={e => setQuoteModal(prev => ({ ...prev, specificDetails: { ...prev.specificDetails, style: e.target.value } }))}
                                                        style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px' }}
                                                    >
                                                        <option value="">Select Style</option>
                                                        <option value="Braided">Braided</option>
                                                        <option value="Die-formed">Die-formed</option>
                                                        <option value="Wrapped">Wrapped</option>
                                                        <option value="Moulded">Moulded</option>
                                                        <option value="Cut">Cut</option>
                                                    </select>
                                                </div>
                                                <div>
                                                    <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: '500', fontSize: '0.875rem' }}>Material Grade</label>
                                                    <select
                                                        value={quoteModal.specificDetails.materialGrade}
                                                        onChange={e => setQuoteModal(prev => ({ ...prev, specificDetails: { ...prev.specificDetails, materialGrade: e.target.value } }))}
                                                        style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px' }}
                                                    >
                                                        <option value="">Select Grade</option>
                                                        <option value="Standard">Standard</option>
                                                        <option value="High Purity">High Purity</option>
                                                        <option value="Nuclear Grade">Nuclear Grade</option>
                                                        <option value="FDA Approved">FDA Approved</option>
                                                    </select>
                                                </div>
                                            </div>
                                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                                                <div>
                                                    <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: '500', fontSize: '0.875rem' }}>Colour</label>
                                                    <input
                                                        type="text"
                                                        value={quoteModal.specificDetails.colour}
                                                        onChange={e => setQuoteModal(prev => ({ ...prev, specificDetails: { ...prev.specificDetails, colour: e.target.value } }))}
                                                        placeholder="e.g., Natural/Grey, Black"
                                                        style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px' }}
                                                    />
                                                </div>
                                                <div>
                                                    <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: '500', fontSize: '0.875rem' }}>Quantity</label>
                                                    <input
                                                        type="number"
                                                        value={quoteModal.specificDetails.quantity}
                                                        onChange={e => setQuoteModal(prev => ({ ...prev, specificDetails: { ...prev.specificDetails, quantity: e.target.value } }))}
                                                        placeholder="e.g., 100"
                                                        style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px' }}
                                                    />
                                                </div>
                                            </div>
                                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                                                <div>
                                                    <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: '500', fontSize: '0.875rem' }}>Unit</label>
                                                    <select
                                                        value={quoteModal.specificDetails.unit}
                                                        onChange={e => setQuoteModal(prev => ({ ...prev, specificDetails: { ...prev.specificDetails, unit: e.target.value } }))}
                                                        style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px' }}
                                                    >
                                                        <option value="Nos.">Nos.</option>
                                                        <option value="Meters">Meters</option>
                                                        <option value="Kg">Kg</option>
                                                        <option value="Sets">Sets</option>
                                                        <option value="Rolls">Rolls</option>
                                                        <option value="Pcs">Pcs</option>
                                                    </select>
                                                </div>
                                                <div>
                                                    <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: '500', fontSize: '0.875rem' }}>Rings/Set <span style={{ color: '#999', fontWeight: '400' }}>(optional)</span></label>
                                                    <input
                                                        type="number"
                                                        value={quoteModal.specificDetails.rings_per_set}
                                                        onChange={e => setQuoteModal(prev => ({ ...prev, specificDetails: { ...prev.specificDetails, rings_per_set: e.target.value } }))}
                                                        placeholder="e.g., 5"
                                                        min="1"
                                                        style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px' }}
                                                    />
                                                </div>
                                            </div>
                                            <div style={{ marginBottom: '1rem' }}>
                                                <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: '500', fontSize: '0.875rem' }}>Specific Requirements</label>
                                                <textarea
                                                    value={quoteModal.specificDetails.specificRequirements}
                                                    onChange={e => setQuoteModal(prev => ({ ...prev, specificDetails: { ...prev.specificDetails, specificRequirements: e.target.value } }))}
                                                    placeholder="e.g., API 622 certified, specific certifications, temperature requirements..."
                                                    style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px', minHeight: '60px' }}
                                                />
                                            </div>
                                            <div style={{ marginBottom: '1rem' }}>
                                                <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: '500', fontSize: '0.875rem' }}>Additional Notes</label>
                                                <textarea
                                                    value={quoteModal.specificDetails.notes}
                                                    onChange={e => setQuoteModal(prev => ({ ...prev, specificDetails: { ...prev.specificDetails, notes: e.target.value } }))}
                                                    placeholder="Any other information..."
                                                    style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px', minHeight: '60px' }}
                                                />
                                            </div>
                                        </>
                                    ) : (
                                        <div style={{ marginBottom: '1rem' }}>
                                            <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: '500', fontSize: '0.875rem' }}>Describe Your Application *</label>
                                            <textarea
                                                value={quoteModal.genericMessage}
                                                onChange={e => setQuoteModal(prev => ({ ...prev, genericMessage: e.target.value }))}
                                                placeholder="Describe your application, operating conditions, and any specific requirements..."
                                                style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e5e5', borderRadius: '4px', minHeight: '120px' }}
                                                required
                                            />
                                            <p style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.5rem' }}>
                                                Our AI will analyze your requirements and suggest the best specifications.
                                            </p>
                                        </div>
                                    )}

                                    <button
                                        onClick={() => submitQuote(quoteModal.mode === 'generic')}
                                        disabled={quoteModal.isSubmitting || !quoteModal.customerInfo.name || !quoteModal.customerInfo.email}
                                        style={{
                                            width: '100%',
                                            padding: '0.875rem',
                                            backgroundColor: quoteModal.isSubmitting ? '#ccc' : '#952825',
                                            color: 'white',
                                            border: 'none',
                                            borderRadius: '4px',
                                            cursor: quoteModal.isSubmitting ? 'not-allowed' : 'pointer',
                                            fontWeight: '600'
                                        }}
                                    >
                                        {quoteModal.isSubmitting ? 'Submitting...' : 'Submit Quote Request'}
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </>
        );
    }

    // Render loading state
    if (state.isLoading) {
        return (
            <div style={{
                padding: '3rem',
                textAlign: 'center'
            }}>
                <div style={{
                    width: '48px',
                    height: '48px',
                    border: '4px solid #f5f5f5',
                    borderTop: '4px solid #952825',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite',
                    margin: '0 auto 1rem'
                }} />
                <p style={{ color: '#666' }}>Analyzing your requirements...</p>
                <style>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
            </div>
        );
    }

    // Render error state
    if (state.error) {
        return (
            <div style={{ padding: '2rem', textAlign: 'center' }}>
                <div style={{
                    padding: '1rem',
                    backgroundColor: '#ffebee',
                    border: '1px solid #ffcdd2',
                    borderRadius: '8px',
                    color: '#c62828',
                    marginBottom: '1rem'
                }}>
                    {state.error}
                </div>
                <button
                    onClick={handleReset}
                    style={{
                        padding: '0.5rem 1rem',
                        backgroundColor: '#952825',
                        border: 'none',
                        borderRadius: '4px',
                        color: 'white',
                        cursor: 'pointer'
                    }}
                >
                    Try Again
                </button>
            </div>
        );
    }

    // Render wizard step
    return (
        <div style={{ padding: '1.5rem' }}>
            {/* Progress indicator */}
            <div style={{ marginBottom: '1.5rem' }}>
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    marginBottom: '0.5rem',
                    fontSize: '0.8rem',
                    color: '#666'
                }}>
                    <span>Step {state.currentStep + 1} of {SELECTION_STEPS.length}</span>
                    <span>{Math.round(((state.currentStep + 1) / SELECTION_STEPS.length) * 100)}%</span>
                </div>
                <div style={{
                    height: '4px',
                    backgroundColor: '#e5e5e5',
                    borderRadius: '2px',
                    overflow: 'hidden'
                }}>
                    <div style={{
                        height: '100%',
                        width: `${((state.currentStep + 1) / SELECTION_STEPS.length) * 100}%`,
                        backgroundColor: '#952825',
                        transition: 'width 0.3s ease'
                    }} />
                </div>
            </div>

            {/* Question */}
            <h3 style={{
                fontSize: '1.1rem',
                fontWeight: '500',
                color: '#2d2d2d',
                marginBottom: '1.25rem'
            }}>
                {currentStepData.question}
            </h3>

            {/* Options */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {currentStepData.options?.map(option => {
                    const isSelected = currentStepData.type === 'multiselect'
                        ? selectedOptions.includes(option)
                        : state.answers[currentStepData.field] === option;

                    return (
                        <button
                            key={option}
                            onClick={() => handleOptionSelect(option)}
                            style={{
                                padding: '0.875rem 1rem',
                                backgroundColor: isSelected ? '#fdf5f5' : 'white',
                                border: `1px solid ${isSelected ? '#952825' : '#e5e5e5'}`,
                                borderRadius: '6px',
                                textAlign: 'left',
                                cursor: 'pointer',
                                transition: 'all 0.2s',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.75rem'
                            }}
                        >
                            {currentStepData.type === 'multiselect' && (
                                <span style={{
                                    width: '18px',
                                    height: '18px',
                                    border: `2px solid ${isSelected ? '#952825' : '#ccc'}`,
                                    borderRadius: '3px',
                                    backgroundColor: isSelected ? '#952825' : 'transparent',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    color: 'white',
                                    fontSize: '12px'
                                }}>
                                    {isSelected && ''}
                                </span>
                            )}
                            <span style={{ color: '#2d2d2d', fontSize: '0.9rem' }}>{option}</span>
                        </button>
                    );
                })}
            </div>

            {/* Multi-select submit button */}
            {currentStepData.type === 'multiselect' && (
                <button
                    onClick={handleMultiSelectSubmit}
                    disabled={selectedOptions.length === 0 || state.isLoading}
                    style={{
                        marginTop: '1rem',
                        width: '100%',
                        padding: '0.875rem',
                        backgroundColor: (selectedOptions.length > 0 && !state.isLoading) ? '#952825' : '#ccc',
                        border: 'none',
                        borderRadius: '6px',
                        color: 'white',
                        cursor: (selectedOptions.length > 0 && !state.isLoading) ? 'pointer' : 'not-allowed',
                        fontWeight: '500',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '0.5rem'
                    }}
                >
                    {state.isLoading ? (
                        <>
                            <span style={{
                                width: '16px',
                                height: '16px',
                                border: '2px solid #fff',
                                borderTopColor: 'transparent',
                                borderRadius: '50%',
                                animation: 'spin 1s linear infinite'
                            }} />
                            Finding best products...
                        </>
                    ) : (
                        isLastStep ? 'Get Recommendations' : 'Continue'
                    )}
                </button>
            )}

            {/* Back button */}
            {state.currentStep > 0 && (
                <button
                    onClick={handleBack}
                    style={{
                        marginTop: '1rem',
                        padding: '0.5rem 1rem',
                        backgroundColor: 'transparent',
                        border: 'none',
                        color: '#666',
                        cursor: 'pointer',
                        fontSize: '0.875rem'
                    }}
                >
                    ← Back
                </button>
            )}
        </div>
    );
}
