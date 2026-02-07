'use client';

import { useState } from 'react';

interface CustomerInfo {
    name: string;
    company: string;
    email: string;
    phone: string;
    address: string;
    designation: string;
}

interface LineItem {
    product_code: string;
    product_name: string;
    size: string;  // e.g., "12mm × 12mm"
    style: string;  // e.g., "Braided", "Die-formed"
    material_grade: string;  // e.g., "Standard", "High Purity"
    material_code: string;  // Internal material code (e.g., "316SS", "PTFE")
    colour: string;  // e.g., "Natural/Grey", "Black"
    size_od: string;  // Outer diameter
    size_id: string;  // Inner diameter
    size_th: string;  // Thickness
    dimension_unit: string;  // Unit for dimensions: "mm" or "inch"
    quantity: number;
    unit: string;  // e.g., "Nos.", "Set", "Mtr.", "Kg."
    rings_per_set: string;  // Number of rings per set
    specific_requirements: string;  // Free text for any specific needs
    notes: string;
}

type SubmissionMode = 'choose' | 'specific' | 'generic';

export default function QuotationForm() {
    const [mode, setMode] = useState<SubmissionMode>('choose');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [submitResult, setSubmitResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    // Customer info (shared)
    const [customer, setCustomer] = useState<CustomerInfo>({
        name: '',
        company: '',
        email: '',
        phone: '',
        address: '',
        designation: ''
    });

    // For specific mode
    const [lineItems, setLineItems] = useState<LineItem[]>([{
        product_code: '',
        product_name: '',
        size: '',
        style: '',
        material_grade: '',
        material_code: '',
        colour: '',
        size_od: '',
        size_id: '',
        size_th: '',
        dimension_unit: 'mm',
        quantity: 1,
        unit: 'Nos.',
        rings_per_set: '',
        specific_requirements: '',
        notes: ''
    }]);
    const [industry, setIndustry] = useState('');
    const [application, setApplication] = useState('');
    const [operatingConditions, setOperatingConditions] = useState('');
    const [specialRequirements, setSpecialRequirements] = useState('');
    const [referenceRfq, setReferenceRfq] = useState('');

    // For generic mode
    const [message, setMessage] = useState('');

    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    const addLineItem = () => {
        setLineItems([...lineItems, {
            product_code: '',
            product_name: '',
            size: '',
            style: '',
            material_grade: '',
            material_code: '',
            colour: '',
            size_od: '',
            size_id: '',
            size_th: '',
            dimension_unit: 'mm',
            quantity: 1,
            unit: 'Nos.',
            rings_per_set: '',
            specific_requirements: '',
            notes: ''
        }]);
    };

    // Compute total number of products across all line items
    const totalProducts = lineItems.reduce((sum, item) => sum + (item.quantity || 0), 0);

    const removeLineItem = (index: number) => {
        if (lineItems.length > 1) {
            setLineItems(lineItems.filter((_, i) => i !== index));
        }
    };

    const updateLineItem = (index: number, field: keyof LineItem, value: any) => {
        const updated = [...lineItems];
        updated[index] = { ...updated[index], [field]: value };
        setLineItems(updated);
    };

    const handleSubmitSpecific = async () => {
        setIsSubmitting(true);
        setError(null);

        try {
            const response = await fetch(`${API_URL}/api/v1/quotations/external/submit-specific`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    customer,
                    reference_rfq: referenceRfq || null,
                    industry: industry || null,
                    application: application || null,
                    operating_conditions: operatingConditions || null,
                    special_requirements: specialRequirements || null,
                    line_items: lineItems.map(item => ({
                        product_code: item.product_code,
                        product_name: item.product_name || item.product_code,
                        size: item.size || null,
                        size_od: item.size_od ? parseFloat(item.size_od) : null,
                        size_id: item.size_id ? parseFloat(item.size_id) : null,
                        size_th: item.size_th ? parseFloat(item.size_th) : null,
                        dimension_unit: item.dimension_unit || 'mm',
                        style: item.style || null,
                        material_grade: item.material_grade || null,
                        material_code: item.material_code || null,
                        colour: item.colour || null,
                        dimensions: {
                            od: item.size_od ? parseFloat(item.size_od) : null,
                            id: item.size_id ? parseFloat(item.size_id) : null,
                            th: item.size_th ? parseFloat(item.size_th) : null
                        },
                        quantity: item.quantity,
                        unit: item.unit,
                        rings_per_set: item.rings_per_set ? parseInt(item.rings_per_set) : null,
                        specific_requirements: item.specific_requirements || null,
                        notes: item.notes || null,
                        is_ai_suggested: false
                    })),
                    total_products: totalProducts
                })
            });

            const data = await response.json();
            if (response.ok) {
                setSubmitResult(data);
            } else {
                setError(data.detail || 'Failed to submit quotation request');
            }
        } catch (e: any) {
            setError(e.message || 'Network error');
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleSubmitGeneric = async () => {
        setIsSubmitting(true);
        setError(null);

        try {
            const response = await fetch(`${API_URL}/api/v1/quotations/external/submit-generic`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    customer,
                    message,
                    reference_rfq: referenceRfq || null,
                    industry: industry || null
                })
            });

            const data = await response.json();
            if (response.ok) {
                setSubmitResult(data);
            } else {
                setError(data.detail || 'Failed to submit quotation request');
            }
        } catch (e: any) {
            setError(e.message || 'Network error');
        } finally {
            setIsSubmitting(false);
        }
    };

    // Success screen
    if (submitResult) {
        return (
            <div style={{
                padding: '2rem',
                backgroundColor: '#f0fff4',
                borderRadius: '12px',
                border: '1px solid #22c55e',
                textAlign: 'center'
            }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}></div>
                <h2 style={{ color: '#15803d', marginBottom: '1rem' }}>Request Submitted!</h2>
                <p style={{ color: '#166534', marginBottom: '1rem' }}>{submitResult.message}</p>
                <p style={{ color: '#666', fontSize: '0.875rem' }}>
                    Request ID: <code>{submitResult.request_id}</code>
                </p>
                {submitResult.ai_summary && (
                    <div style={{
                        marginTop: '1rem',
                        padding: '1rem',
                        backgroundColor: '#e0f2fe',
                        borderRadius: '8px',
                        textAlign: 'left'
                    }}>
                        <strong>AI Analysis:</strong>
                        <p>{submitResult.ai_summary.one_liner}</p>
                    </div>
                )}
                <button
                    onClick={() => {
                        setSubmitResult(null);
                        setMode('choose');
                    }}
                    style={{
                        marginTop: '1rem',
                        padding: '0.75rem 2rem',
                        backgroundColor: '#22c55e',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer'
                    }}
                >
                    Submit Another Request
                </button>
            </div>
        );
    }

    // Mode selection screen
    if (mode === 'choose') {
        return (
            <div style={{ padding: '1rem' }}>
                <h2 style={{ textAlign: 'center', marginBottom: '2rem', color: '#1e293b' }}>
                    Request a Quotation
                </h2>

                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                    gap: '1.5rem',
                    maxWidth: '800px',
                    margin: '0 auto'
                }}>
                    {/* Specific Requirements Option */}
                    <div
                        onClick={() => setMode('specific')}
                        style={{
                            padding: '2rem',
                            backgroundColor: '#f8fafc',
                            border: '2px solid #e2e8f0',
                            borderRadius: '12px',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            textAlign: 'center'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.borderColor = '#8B0000';
                            e.currentTarget.style.transform = 'translateY(-2px)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.borderColor = '#e2e8f0';
                            e.currentTarget.style.transform = 'translateY(0)';
                        }}
                    >
                        <div style={{ fontSize: '3rem', marginBottom: '1rem' }}></div>
                        <h3 style={{ color: '#1e293b', marginBottom: '0.5rem' }}>
                            I Have Specific Requirements
                        </h3>
                        <p style={{ color: '#64748b', fontSize: '0.875rem' }}>
                            I know the product codes, sizes, and quantities I need.
                            I'll provide all the details for an accurate quotation.
                        </p>
                        <div style={{
                            marginTop: '1rem',
                            padding: '0.5rem 1rem',
                            backgroundColor: '#dcfce7',
                            borderRadius: '20px',
                            color: '#15803d',
                            fontSize: '0.75rem',
                            display: 'inline-block'
                        }}>
                            Faster Processing
                        </div>
                    </div>

                    {/* Generic Request Option */}
                    <div
                        onClick={() => setMode('generic')}
                        style={{
                            padding: '2rem',
                            backgroundColor: '#f8fafc',
                            border: '2px solid #e2e8f0',
                            borderRadius: '12px',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            textAlign: 'center'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.borderColor = '#8B0000';
                            e.currentTarget.style.transform = 'translateY(-2px)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.borderColor = '#e2e8f0';
                            e.currentTarget.style.transform = 'translateY(0)';
                        }}
                    >
                        <div style={{ fontSize: '3rem', marginBottom: '1rem' }}></div>
                        <h3 style={{ color: '#1e293b', marginBottom: '0.5rem' }}>
                            I Need Help Identifying Products
                        </h3>
                        <p style={{ color: '#64748b', fontSize: '0.875rem' }}>
                            I'll describe my application and requirements.
                            Your AI system will recommend the right products for me.
                        </p>
                        <div style={{
                            marginTop: '1rem',
                            padding: '0.5rem 1rem',
                            backgroundColor: '#e0f2fe',
                            borderRadius: '20px',
                            color: '#0369a1',
                            fontSize: '0.75rem',
                            display: 'inline-block'
                        }}>
                            AI-Assisted
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    // Shared customer info section
    const CustomerSection = () => (
        <div style={{ marginBottom: '2rem' }}>
            <h3 style={{ color: '#1e293b', marginBottom: '1rem', borderBottom: '2px solid #8B0000', paddingBottom: '0.5rem' }}>
                Your Information
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                <input
                    type="text"
                    placeholder="Full Name *"
                    value={customer.name}
                    onChange={(e) => setCustomer({ ...customer, name: e.target.value })}
                    required
                    style={inputStyle}
                />
                <input
                    type="text"
                    placeholder="Company Name *"
                    value={customer.company}
                    onChange={(e) => setCustomer({ ...customer, company: e.target.value })}
                    required
                    style={inputStyle}
                />
                <input
                    type="email"
                    placeholder="Email Address *"
                    value={customer.email}
                    onChange={(e) => setCustomer({ ...customer, email: e.target.value })}
                    required
                    style={inputStyle}
                />
                <input
                    type="tel"
                    placeholder="Phone Number"
                    value={customer.phone}
                    onChange={(e) => setCustomer({ ...customer, phone: e.target.value })}
                    style={inputStyle}
                />
                <input
                    type="text"
                    placeholder="Designation"
                    value={customer.designation}
                    onChange={(e) => setCustomer({ ...customer, designation: e.target.value })}
                    style={inputStyle}
                />
                <input
                    type="text"
                    placeholder="Reference RFQ Number"
                    value={referenceRfq}
                    onChange={(e) => setReferenceRfq(e.target.value)}
                    style={inputStyle}
                />
            </div>
            <textarea
                placeholder="Address"
                value={customer.address}
                onChange={(e) => setCustomer({ ...customer, address: e.target.value })}
                style={{ ...inputStyle, width: '100%', minHeight: '80px', marginTop: '1rem' }}
            />
        </div>
    );

    const inputStyle: React.CSSProperties = {
        padding: '0.75rem',
        border: '1px solid #d1d5db',
        borderRadius: '6px',
        fontSize: '0.875rem',
        outline: 'none'
    };

    // Specific requirements form
    if (mode === 'specific') {
        return (
            <div style={{ padding: '1rem' }}>
                <button
                    onClick={() => setMode('choose')}
                    style={{
                        marginBottom: '1rem',
                        background: 'none',
                        border: 'none',
                        color: '#8B0000',
                        cursor: 'pointer',
                        fontSize: '0.875rem'
                    }}
                >
                    ← Back to options
                </button>

                <h2 style={{ color: '#1e293b', marginBottom: '1.5rem' }}>
                    Quotation with Specific Requirements
                </h2>

                <CustomerSection />

                {/* Application Details */}
                <div style={{ marginBottom: '2rem' }}>
                    <h3 style={{ color: '#1e293b', marginBottom: '1rem', borderBottom: '2px solid #8B0000', paddingBottom: '0.5rem' }}>
                        Application Details
                    </h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                        <select
                            value={industry}
                            onChange={(e) => setIndustry(e.target.value)}
                            style={inputStyle}
                        >
                            <option value="">Select Industry</option>
                            <option value="petrochemical">Petrochemical</option>
                            <option value="power">Power Generation</option>
                            <option value="pharmaceutical">Pharmaceutical</option>
                            <option value="food">Food & Beverage</option>
                            <option value="chemical">Chemical Processing</option>
                            <option value="water">Water Treatment</option>
                            <option value="other">Other</option>
                        </select>
                        <input
                            type="text"
                            placeholder="Application (e.g., Valve Packing)"
                            value={application}
                            onChange={(e) => setApplication(e.target.value)}
                            style={inputStyle}
                        />
                    </div>
                    <textarea
                        placeholder="Operating Conditions (temperature, pressure, media handled)"
                        value={operatingConditions}
                        onChange={(e) => setOperatingConditions(e.target.value)}
                        style={{ ...inputStyle, width: '100%', minHeight: '80px', marginTop: '1rem' }}
                    />
                    <textarea
                        placeholder="Special Requirements (certifications, fire-safe, etc.)"
                        value={specialRequirements}
                        onChange={(e) => setSpecialRequirements(e.target.value)}
                        style={{ ...inputStyle, width: '100%', minHeight: '80px', marginTop: '1rem' }}
                    />
                </div>

                {/* Line Items */}
                <div style={{ marginBottom: '2rem' }}>
                    <h3 style={{ color: '#1e293b', marginBottom: '1rem', borderBottom: '2px solid #8B0000', paddingBottom: '0.5rem' }}>
                        Products Required
                    </h3>

                    {lineItems.map((item, idx) => (
                        <div key={idx} style={{
                            padding: '1rem',
                            backgroundColor: '#f8fafc',
                            borderRadius: '8px',
                            marginBottom: '1rem',
                            border: '1px solid #e2e8f0'
                        }}>
                            <div style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                marginBottom: '0.75rem'
                            }}>
                                <strong>Item {idx + 1}</strong>
                                {lineItems.length > 1 && (
                                    <button
                                        onClick={() => removeLineItem(idx)}
                                        style={{
                                            background: '#fee2e2',
                                            border: 'none',
                                            color: '#dc2626',
                                            padding: '0.25rem 0.5rem',
                                            borderRadius: '4px',
                                            cursor: 'pointer',
                                            fontSize: '0.75rem'
                                        }}
                                    >
                                        Remove
                                    </button>
                                )}
                            </div>

                            {/* Row 1: Product Code, Material Grade */}
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.75rem' }}>
                                <input
                                    type="text"
                                    placeholder="Product Code (JDJ Style) *"
                                    value={item.product_code}
                                    onChange={(e) => updateLineItem(idx, 'product_code', e.target.value)}
                                    required
                                    style={inputStyle}
                                />
                                <select
                                    value={item.material_grade}
                                    onChange={(e) => updateLineItem(idx, 'material_grade', e.target.value)}
                                    style={inputStyle}
                                >
                                    <option value="">Material Grade</option>
                                    <option value="Standard">Standard</option>
                                    <option value="High Purity">High Purity</option>
                                    <option value="Nuclear Grade">Nuclear Grade</option>
                                    <option value="FDA Approved">FDA Approved</option>
                                </select>
                            </div>

                            {/* Row 2: Size dimensions OD / ID / TH + Dimension Unit + Rings/Set */}
                            <div style={{ marginTop: '0.75rem' }}>
                                <label style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: '0.25rem', display: 'block' }}>Size (in {item.dimension_unit})</label>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '0.75rem' }}>
                                    <input
                                        type="number"
                                        placeholder={`OD (${item.dimension_unit})`}
                                        value={item.size_od}
                                        onChange={(e) => updateLineItem(idx, 'size_od', e.target.value)}
                                        step="0.01"
                                        min="0"
                                        style={inputStyle}
                                    />
                                    <input
                                        type="number"
                                        placeholder={`ID (${item.dimension_unit})`}
                                        value={item.size_id}
                                        onChange={(e) => updateLineItem(idx, 'size_id', e.target.value)}
                                        step="0.01"
                                        min="0"
                                        style={inputStyle}
                                    />
                                    <input
                                        type="number"
                                        placeholder={`TH (${item.dimension_unit})`}
                                        value={item.size_th}
                                        onChange={(e) => updateLineItem(idx, 'size_th', e.target.value)}
                                        step="0.01"
                                        min="0"
                                        style={inputStyle}
                                    />
                                    <select
                                        value={item.dimension_unit}
                                        onChange={(e) => updateLineItem(idx, 'dimension_unit', e.target.value)}
                                        style={inputStyle}
                                    >
                                        <option value="mm">mm</option>
                                        <option value="inch">inch</option>
                                    </select>
                                    <input
                                        type="number"
                                        placeholder="Rings/Set"
                                        value={item.rings_per_set}
                                        onChange={(e) => updateLineItem(idx, 'rings_per_set', e.target.value)}
                                        min="1"
                                        style={inputStyle}
                                    />
                                </div>
                            </div>

                            {/* Row 3: Qty, Unit, Style, Colour */}
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '0.75rem', marginTop: '0.75rem' }}>
                                <input
                                    type="number"
                                    placeholder="Quantity *"
                                    value={item.quantity}
                                    onChange={(e) => updateLineItem(idx, 'quantity', parseInt(e.target.value) || 1)}
                                    min="1"
                                    required
                                    style={inputStyle}
                                />
                                <select
                                    value={item.unit}
                                    onChange={(e) => updateLineItem(idx, 'unit', e.target.value)}
                                    style={inputStyle}
                                >
                                    <option value="Nos.">Nos.</option>
                                    <option value="Set">Set</option>
                                    <option value="Mtr.">Mtr.</option>
                                    <option value="Kg.">Kg.</option>
                                    <option value="Pcs.">Pcs.</option>
                                </select>
                                <select
                                    value={item.style}
                                    onChange={(e) => updateLineItem(idx, 'style', e.target.value)}
                                    style={inputStyle}
                                >
                                    <option value="">Select Style</option>
                                    <option value="Braided">Braided</option>
                                    <option value="Die-formed">Die-formed</option>
                                    <option value="Wrapped">Wrapped</option>
                                    <option value="Moulded">Moulded</option>
                                    <option value="Cut">Cut</option>
                                    <option value="Spiral Wound">Spiral Wound</option>
                                </select>
                                <input
                                    type="text"
                                    placeholder="Colour"
                                    value={item.colour}
                                    onChange={(e) => updateLineItem(idx, 'colour', e.target.value)}
                                    style={inputStyle}
                                />
                            </div>



                            <textarea
                                placeholder="Specific Requirements (e.g., certifications, special conditions)"
                                value={item.specific_requirements}
                                onChange={(e) => updateLineItem(idx, 'specific_requirements', e.target.value)}
                                style={{ ...inputStyle, width: '100%', minHeight: '60px', marginTop: '0.75rem' }}
                            />
                            <input
                                type="text"
                                placeholder="Additional Notes"
                                value={item.notes}
                                onChange={(e) => updateLineItem(idx, 'notes', e.target.value)}
                                style={{ ...inputStyle, width: '100%', marginTop: '0.5rem' }}
                            />
                        </div>
                    ))}

                    {/* Total Products Summary */}
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '0.75rem 1rem',
                        backgroundColor: '#f0f9ff',
                        border: '1px solid #bae6fd',
                        borderRadius: '6px',
                        marginBottom: '0.75rem'
                    }}>
                        <span style={{ fontWeight: '600', color: '#0369a1', fontSize: '0.9rem' }}>
                            Total No. of Products: {lineItems.length} items | Total Qty: {totalProducts}
                        </span>
                    </div>

                    <button
                        onClick={addLineItem}
                        style={{
                            padding: '0.75rem 1.5rem',
                            backgroundColor: '#f1f5f9',
                            border: '1px dashed #94a3b8',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            width: '100%',
                            color: '#475569'
                        }}
                    >
                        + Add Another Product
                    </button>
                </div>

                {error && (
                    <div style={{ padding: '1rem', backgroundColor: '#fee2e2', color: '#dc2626', borderRadius: '6px', marginBottom: '1rem' }}>
                        {error}
                    </div>
                )}

                <button
                    onClick={handleSubmitSpecific}
                    disabled={isSubmitting || !customer.name || !customer.company || !customer.email || !lineItems[0].product_code}
                    style={{
                        padding: '1rem 2rem',
                        backgroundColor: isSubmitting ? '#9ca3af' : '#8B0000',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: isSubmitting ? 'not-allowed' : 'pointer',
                        fontSize: '1rem',
                        width: '100%'
                    }}
                >
                    {isSubmitting ? 'Submitting...' : 'Submit Quotation Request'}
                </button>
            </div>
        );
    }

    // Generic request form
    if (mode === 'generic') {
        return (
            <div style={{ padding: '1rem' }}>
                <button
                    onClick={() => setMode('choose')}
                    style={{
                        marginBottom: '1rem',
                        background: 'none',
                        border: 'none',
                        color: '#8B0000',
                        cursor: 'pointer',
                        fontSize: '0.875rem'
                    }}
                >
                    ← Back to options
                </button>

                <h2 style={{ color: '#1e293b', marginBottom: '1.5rem' }}>
                    AI-Assisted Quotation Request
                </h2>

                <div style={{
                    padding: '1rem',
                    backgroundColor: '#e0f2fe',
                    borderRadius: '8px',
                    marginBottom: '1.5rem',
                    border: '1px solid #7dd3fc'
                }}>
                    <strong>How it works:</strong>
                    <p style={{ fontSize: '0.875rem', color: '#0369a1', marginTop: '0.5rem' }}>
                        Simply describe your application and requirements. Our AI system will analyze your needs,
                        recommend suitable products, and our team will prepare a detailed quotation for you.
                    </p>
                </div>

                <CustomerSection />

                {/* Application Context */}
                <div style={{ marginBottom: '2rem' }}>
                    <h3 style={{ color: '#1e293b', marginBottom: '1rem', borderBottom: '2px solid #8B0000', paddingBottom: '0.5rem' }}>
                        Tell Us About Your Needs
                    </h3>

                    <select
                        value={industry}
                        onChange={(e) => setIndustry(e.target.value)}
                        style={{ ...inputStyle, width: '100%', marginBottom: '1rem' }}
                    >
                        <option value="">Select Industry (Optional)</option>
                        <option value="petrochemical">Petrochemical</option>
                        <option value="power">Power Generation</option>
                        <option value="pharmaceutical">Pharmaceutical</option>
                        <option value="food">Food & Beverage</option>
                        <option value="chemical">Chemical Processing</option>
                        <option value="water">Water Treatment</option>
                        <option value="other">Other</option>
                    </select>

                    <textarea
                        placeholder="Describe your requirements in detail... 

For example:
- What equipment do you need sealing products for? (valves, pumps, etc.)
- What are the operating conditions? (temperature, pressure, media)
- What industry/application?
- Any specific certifications needed?
- Quantity estimates if known

The more details you provide, the better we can assist you."
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        required
                        style={{
                            ...inputStyle,
                            width: '100%',
                            minHeight: '200px',
                            fontFamily: 'inherit'
                        }}
                    />
                </div>

                {error && (
                    <div style={{ padding: '1rem', backgroundColor: '#fee2e2', color: '#dc2626', borderRadius: '6px', marginBottom: '1rem' }}>
                        {error}
                    </div>
                )}

                <button
                    onClick={handleSubmitGeneric}
                    disabled={isSubmitting || !customer.name || !customer.company || !customer.email || !message}
                    style={{
                        padding: '1rem 2rem',
                        backgroundColor: isSubmitting ? '#9ca3af' : '#8B0000',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: isSubmitting ? 'not-allowed' : 'pointer',
                        fontSize: '1rem',
                        width: '100%'
                    }}
                >
                    {isSubmitting ? 'Submitting & Analyzing...' : 'Submit for AI Analysis'}
                </button>
            </div>
        );
    }

    return null;
}
