'use client';

import { useState, useEffect } from 'react';

interface EnquiryFormProps {
    prefilledProduct?: string;
}

interface EnquiryResponse {
    enquiry_id: string;
    status: string;
    category: string;
    response: string;
    routed_to?: string;
    estimated_response_time?: string;
}

export default function EnquiryForm({ prefilledProduct }: EnquiryFormProps) {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        company: '',
        phone: '',
        enquiryType: 'product_info',
        productCode: prefilledProduct || '',
        message: ''
    });

    const [isSubmitting, setIsSubmitting] = useState(false);
    const [response, setResponse] = useState<EnquiryResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [availableProducts, setAvailableProducts] = useState<Array<{ code: string, name: string, category: string, description: string }>>([]);
    const [productSearch, setProductSearch] = useState('');
    const [showProductDropdown, setShowProductDropdown] = useState(false);

    useEffect(() => {
        if (prefilledProduct) {
            setFormData(prev => ({ ...prev, productCode: prefilledProduct }));
        }
    }, [prefilledProduct]);

    // Fetch available products from knowledge system
    useEffect(() => {
        const fetchProducts = async () => {
            try {
                const res = await fetch('/api/products');
                const data = await res.json();
                if (data.success && data.products) {
                    setAvailableProducts(data.products);
                }
            } catch (err) {
                console.error('Failed to fetch products:', err);
            }
        };
        fetchProducts();
    }, []);

    const enquiryTypes = [
        { value: 'product_info', label: 'Product Information' },
        { value: 'quote_request', label: 'Quote Request' },
        { value: 'technical_support', label: 'Technical Support' },
        { value: 'order_status', label: 'Order Status' },
        { value: 'complaint', label: 'Complaint / Issue' },
        { value: 'other', label: 'Other' }
    ];

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsSubmitting(true);
        setError(null);

        try {
            const response = await fetch('/api/agentic/enquiry', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: formData.message,
                    content: formData.message + (formData.productCode ? ` (Product: ${formData.productCode})` : ''),
                    from_email: formData.email,
                    from_name: formData.name,
                    company: formData.company,
                    phone: formData.phone,
                    enquiry_type: formData.enquiryType,
                    product_code: formData.productCode || undefined
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || data.error || 'Failed to submit enquiry');
            }

            setResponse(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleReset = () => {
        setFormData({
            name: '',
            email: '',
            company: '',
            phone: '',
            enquiryType: 'product_info',
            productCode: '',
            message: ''
        });
        setResponse(null);
        setError(null);
    };

    // Show success response
    if (response) {
        return (
            <div style={{ padding: '1.5rem' }}>
                <div style={{
                    padding: '1.5rem',
                    backgroundColor: '#e8f5e9',
                    border: '1px solid #c8e6c9',
                    borderRadius: '8px',
                    marginBottom: '1.5rem'
                }}>
                    <h3 style={{ color: '#2e7d32', marginTop: 0, marginBottom: '0.75rem' }}>
                        Enquiry Submitted Successfully
                    </h3>
                    <p style={{ color: '#555', margin: 0 }}>
                        Reference: <strong>{response.enquiry_id}</strong>
                    </p>
                    {response.estimated_response_time && (
                        <p style={{ color: '#555', margin: '0.5rem 0 0' }}>
                            Estimated response: {response.estimated_response_time}
                        </p>
                    )}
                </div>

                {response.response && (
                    <div style={{
                        padding: '1.25rem',
                        backgroundColor: 'white',
                        border: '1px solid #e5e5e5',
                        borderRadius: '8px',
                        marginBottom: '1rem'
                    }}>
                        <h4 style={{ margin: '0 0 0.75rem', color: '#2d2d2d', fontSize: '1rem' }}>
                            Instant Response
                        </h4>
                        <p style={{ color: '#555', margin: 0, lineHeight: '1.6', whiteSpace: 'pre-wrap' }}>
                            {response.response}
                        </p>
                    </div>
                )}

                {response.routed_to && (
                    <p style={{ color: '#666', fontSize: '0.875rem' }}>
                        Your enquiry has been forwarded to: <strong>{response.routed_to}</strong>
                    </p>
                )}

                <button
                    onClick={handleReset}
                    style={{
                        padding: '0.75rem 1.5rem',
                        backgroundColor: '#952825',
                        border: 'none',
                        borderRadius: '6px',
                        color: 'white',
                        cursor: 'pointer',
                        fontWeight: '500'
                    }}
                >
                    Submit Another Enquiry
                </button>
            </div>
        );
    }

    return (
        <div style={{ padding: '1.5rem' }}>
            <h2 style={{
                fontSize: '1.25rem',
                fontWeight: '600',
                color: '#2d2d2d',
                marginTop: 0,
                marginBottom: '1.5rem'
            }}>
                Submit an Enquiry
            </h2>

            {error && (
                <div style={{
                    padding: '1rem',
                    backgroundColor: '#ffebee',
                    border: '1px solid #ffcdd2',
                    borderRadius: '6px',
                    color: '#c62828',
                    marginBottom: '1rem'
                }}>
                    {error}
                </div>
            )}

            <form onSubmit={handleSubmit}>
                {/* Contact Information */}
                <div style={{ marginBottom: '1.5rem' }}>
                    <h4 style={{ fontSize: '0.9rem', color: '#666', margin: '0 0 0.75rem' }}>
                        Contact Information
                    </h4>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                        <div>
                            <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                                Name *
                            </label>
                            <input
                                type="text"
                                name="name"
                                value={formData.name}
                                onChange={handleChange}
                                required
                                style={{
                                    width: '100%',
                                    padding: '0.625rem',
                                    border: '1px solid #ddd',
                                    borderRadius: '4px',
                                    fontSize: '0.9rem'
                                }}
                            />
                        </div>
                        <div>
                            <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                                Email *
                            </label>
                            <input
                                type="email"
                                name="email"
                                value={formData.email}
                                onChange={handleChange}
                                required
                                style={{
                                    width: '100%',
                                    padding: '0.625rem',
                                    border: '1px solid #ddd',
                                    borderRadius: '4px',
                                    fontSize: '0.9rem'
                                }}
                            />
                        </div>
                        <div>
                            <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                                Company
                            </label>
                            <input
                                type="text"
                                name="company"
                                value={formData.company}
                                onChange={handleChange}
                                style={{
                                    width: '100%',
                                    padding: '0.625rem',
                                    border: '1px solid #ddd',
                                    borderRadius: '4px',
                                    fontSize: '0.9rem'
                                }}
                            />
                        </div>
                        <div>
                            <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                                Phone
                            </label>
                            <input
                                type="tel"
                                name="phone"
                                value={formData.phone}
                                onChange={handleChange}
                                style={{
                                    width: '100%',
                                    padding: '0.625rem',
                                    border: '1px solid #ddd',
                                    borderRadius: '4px',
                                    fontSize: '0.9rem'
                                }}
                            />
                        </div>
                    </div>
                </div>

                {/* Enquiry Details */}
                <div style={{ marginBottom: '1.5rem' }}>
                    <h4 style={{ fontSize: '0.9rem', color: '#666', margin: '0 0 0.75rem' }}>
                        Enquiry Details
                    </h4>

                    <div style={{ marginBottom: '0.75rem' }}>
                        <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                            Enquiry Type *
                        </label>
                        <select
                            name="enquiryType"
                            value={formData.enquiryType}
                            onChange={handleChange}
                            required
                            style={{
                                width: '100%',
                                padding: '0.625rem',
                                border: '1px solid #ddd',
                                borderRadius: '4px',
                                fontSize: '0.9rem',
                                backgroundColor: 'white'
                            }}
                        >
                            {enquiryTypes.map(type => (
                                <option key={type.value} value={type.value}>
                                    {type.label}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div style={{ marginBottom: '0.75rem', position: 'relative' }}>
                        <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                            Product Code *
                        </label>
                        <div
                            onClick={() => setShowProductDropdown(!showProductDropdown)}
                            style={{
                                width: '100%',
                                padding: '0.625rem',
                                border: '1px solid #ddd',
                                borderRadius: '4px',
                                fontSize: '0.9rem',
                                backgroundColor: 'white',
                                cursor: 'pointer',
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center'
                            }}
                        >
                            <span style={{ color: formData.productCode ? '#333' : '#999' }}>
                                {formData.productCode
                                    ? `${formData.productCode}${availableProducts.find(p => p.code === formData.productCode)?.name ? ' - ' + availableProducts.find(p => p.code === formData.productCode)?.name : ''}`
                                    : 'Select a product...'}
                            </span>
                            <span style={{ fontSize: '0.7rem', color: '#999' }}>{showProductDropdown ? '\u25B2' : '\u25BC'}</span>
                        </div>
                        {showProductDropdown && (
                            <div style={{
                                position: 'absolute',
                                top: '100%',
                                left: 0,
                                right: 0,
                                backgroundColor: 'white',
                                border: '1px solid #ddd',
                                borderRadius: '0 0 4px 4px',
                                maxHeight: '250px',
                                overflowY: 'auto',
                                zIndex: 1000,
                                boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                            }}>
                                <div style={{ padding: '0.5rem', borderBottom: '1px solid #eee', position: 'sticky', top: 0, backgroundColor: 'white' }}>
                                    <input
                                        type="text"
                                        placeholder="Search products..."
                                        value={productSearch}
                                        onChange={(e) => setProductSearch(e.target.value)}
                                        onClick={(e) => e.stopPropagation()}
                                        autoFocus
                                        style={{
                                            width: '100%',
                                            padding: '0.5rem',
                                            border: '1px solid #ddd',
                                            borderRadius: '4px',
                                            fontSize: '0.85rem'
                                        }}
                                    />
                                </div>
                                <div
                                    onClick={() => {
                                        setFormData(prev => ({ ...prev, productCode: '' }));
                                        setShowProductDropdown(false);
                                        setProductSearch('');
                                    }}
                                    style={{
                                        padding: '0.625rem 0.75rem',
                                        cursor: 'pointer',
                                        borderBottom: '1px solid #f0f0f0',
                                        backgroundColor: formData.productCode === '' ? '#f5f0ff' : 'white',
                                        fontSize: '0.85rem',
                                        color: '#666',
                                        fontStyle: 'italic'
                                    }}
                                >
                                    Not sure / General enquiry
                                </div>
                                {availableProducts
                                    .filter(p =>
                                        !productSearch ||
                                        p.code.toLowerCase().includes(productSearch.toLowerCase()) ||
                                        p.name.toLowerCase().includes(productSearch.toLowerCase()) ||
                                        p.category.toLowerCase().includes(productSearch.toLowerCase())
                                    )
                                    .map(product => (
                                        <div
                                            key={product.code}
                                            onClick={() => {
                                                setFormData(prev => ({ ...prev, productCode: product.code }));
                                                setShowProductDropdown(false);
                                                setProductSearch('');
                                            }}
                                            style={{
                                                padding: '0.625rem 0.75rem',
                                                cursor: 'pointer',
                                                borderBottom: '1px solid #f0f0f0',
                                                backgroundColor: formData.productCode === product.code ? '#f5f0ff' : 'white',
                                                transition: 'background-color 0.15s'
                                            }}
                                            onMouseEnter={(e) => { if (formData.productCode !== product.code) (e.currentTarget as HTMLDivElement).style.backgroundColor = '#f9f9f9'; }}
                                            onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.backgroundColor = formData.productCode === product.code ? '#f5f0ff' : 'white'; }}
                                        >
                                            <div style={{ fontWeight: '600', fontSize: '0.85rem', color: '#333' }}>
                                                {product.code} <span style={{ fontWeight: '400', color: '#666' }}>- {product.name}</span>
                                            </div>
                                            <div style={{ fontSize: '0.75rem', color: '#999', marginTop: '2px' }}>
                                                {product.category} {product.description ? `| ${product.description}` : ''}
                                            </div>
                                        </div>
                                    ))}
                                {availableProducts.filter(p =>
                                    !productSearch ||
                                    p.code.toLowerCase().includes(productSearch.toLowerCase()) ||
                                    p.name.toLowerCase().includes(productSearch.toLowerCase()) ||
                                    p.category.toLowerCase().includes(productSearch.toLowerCase())
                                ).length === 0 && (
                                        <div style={{ padding: '1rem', textAlign: 'center', color: '#999', fontSize: '0.85rem' }}>
                                            No products found matching "{productSearch}"
                                        </div>
                                    )}
                            </div>
                        )}
                    </div>

                    <div>
                        <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                            Message *
                        </label>
                        <textarea
                            name="message"
                            value={formData.message}
                            onChange={handleChange}
                            required
                            rows={5}
                            placeholder="Please describe your enquiry in detail..."
                            style={{
                                width: '100%',
                                padding: '0.625rem',
                                border: '1px solid #ddd',
                                borderRadius: '4px',
                                fontSize: '0.9rem',
                                resize: 'vertical'
                            }}
                        />
                    </div>
                </div>

                <button
                    type="submit"
                    disabled={isSubmitting}
                    style={{
                        width: '100%',
                        padding: '0.875rem',
                        backgroundColor: isSubmitting ? '#ccc' : '#952825',
                        border: 'none',
                        borderRadius: '6px',
                        color: 'white',
                        cursor: isSubmitting ? 'not-allowed' : 'pointer',
                        fontWeight: '500',
                        fontSize: '1rem'
                    }}
                >
                    {isSubmitting ? 'Submitting...' : 'Submit Enquiry'}
                </button>
            </form>
        </div>
    );
}
