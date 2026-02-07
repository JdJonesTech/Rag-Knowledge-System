'use client';

import { useState } from 'react';

interface GeneratedDocument {
    doc_id: string;
    doc_type: string;
    title: string;
    filename: string;
    download_url: string;
    format: string;
    created_at: string;
}

interface ProductItem {
    code: string;
    name: string;
    quantity: number;
    unit_price: number;
}

export default function DocumentGenerator() {
    const [activeTab, setActiveTab] = useState<'quotation' | 'datasheet'>('quotation');
    const [isGenerating, setIsGenerating] = useState(false);
    const [generatedDoc, setGeneratedDoc] = useState<GeneratedDocument | null>(null);
    const [error, setError] = useState<string | null>(null);

    // Quotation form state
    const [quotationForm, setQuotationForm] = useState({
        customer_name: '',
        customer_email: '',
        notes: '',
        validity_days: 30
    });
    const [products, setProducts] = useState<ProductItem[]>([
        { code: '', name: '', quantity: 1, unit_price: 0 }
    ]);

    // Datasheet form state
    const [datasheetForm, setDatasheetForm] = useState({
        product_code: '',
        product_name: '',
        include_certifications: true
    });

    const addProduct = () => {
        setProducts([...products, { code: '', name: '', quantity: 1, unit_price: 0 }]);
    };

    const removeProduct = (index: number) => {
        if (products.length > 1) {
            setProducts(products.filter((_, i) => i !== index));
        }
    };

    const updateProduct = (index: number, field: keyof ProductItem, value: string | number) => {
        const updated = [...products];
        updated[index] = { ...updated[index], [field]: value };
        setProducts(updated);
    };

    const generateQuotation = async () => {
        if (!quotationForm.customer_name || !quotationForm.customer_email) {
            setError('Customer name and email are required');
            return;
        }

        if (products.some(p => !p.code)) {
            setError('All products must have a product code');
            return;
        }

        setIsGenerating(true);
        setError(null);

        try {
            const response = await fetch('/api/documents/quotation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    customer_name: quotationForm.customer_name,
                    customer_email: quotationForm.customer_email,
                    products: products.map(p => ({
                        code: p.code,
                        name: p.name,
                        quantity: p.quantity,
                        unit_price: p.unit_price
                    })),
                    notes: quotationForm.notes,
                    validity_days: quotationForm.validity_days
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to generate quotation');
            }

            setGeneratedDoc(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setIsGenerating(false);
        }
    };

    const generateDatasheet = async () => {
        if (!datasheetForm.product_code) {
            setError('Product code is required');
            return;
        }

        setIsGenerating(true);
        setError(null);

        try {
            const response = await fetch('/api/documents/datasheet', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    product_code: datasheetForm.product_code,
                    product_name: datasheetForm.product_name || undefined,
                    include_certifications: datasheetForm.include_certifications
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to generate datasheet');
            }

            setGeneratedDoc(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setIsGenerating(false);
        }
    };

    const handleDownload = () => {
        if (generatedDoc) {
            window.open(`/api${generatedDoc.download_url}`, '_blank');
        }
    };

    const handleReset = () => {
        setGeneratedDoc(null);
        setError(null);
    };

    // Success view
    if (generatedDoc) {
        return (
            <div style={{ padding: '1.5rem' }}>
                <div style={{
                    padding: '1.5rem',
                    backgroundColor: '#e8f5e9',
                    border: '1px solid #c8e6c9',
                    borderRadius: '8px',
                    marginBottom: '1.5rem',
                    textAlign: 'center'
                }}>
                    <div style={{
                        width: '60px',
                        height: '60px',
                        backgroundColor: '#4caf50',
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        margin: '0 auto 1rem',
                        color: 'white',
                        fontSize: '1.5rem'
                    }}>
                        
                    </div>
                    <h3 style={{ color: '#2e7d32', margin: '0 0 0.5rem' }}>
                        Document Generated Successfully
                    </h3>
                    <p style={{ color: '#555', margin: '0 0 0.25rem' }}>
                        <strong>{generatedDoc.title}</strong>
                    </p>
                    <p style={{ color: '#666', margin: 0, fontSize: '0.875rem' }}>
                        Format: {generatedDoc.format.toUpperCase()} | ID: {generatedDoc.doc_id}
                    </p>
                </div>

                <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                    <button
                        onClick={handleDownload}
                        style={{
                            padding: '0.75rem 1.5rem',
                            backgroundColor: '#952825',
                            border: 'none',
                            borderRadius: '6px',
                            color: 'white',
                            cursor: 'pointer',
                            fontWeight: '500',
                            fontSize: '1rem'
                        }}
                    >
                        Download {generatedDoc.format.toUpperCase()}
                    </button>
                    <button
                        onClick={handleReset}
                        style={{
                            padding: '0.75rem 1.5rem',
                            backgroundColor: 'white',
                            border: '1px solid #952825',
                            borderRadius: '6px',
                            color: '#952825',
                            cursor: 'pointer',
                            fontSize: '1rem'
                        }}
                    >
                        Generate Another
                    </button>
                </div>
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
                Document Generator
            </h2>

            {/* Tabs */}
            <div style={{
                display: 'flex',
                gap: '0.5rem',
                marginBottom: '1.5rem',
                borderBottom: '1px solid #e5e5e5',
                paddingBottom: '0.5rem'
            }}>
                <button
                    onClick={() => { setActiveTab('quotation'); setError(null); }}
                    style={{
                        padding: '0.5rem 1rem',
                        backgroundColor: activeTab === 'quotation' ? '#952825' : 'transparent',
                        color: activeTab === 'quotation' ? 'white' : '#666',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontWeight: '500'
                    }}
                >
                    Quotation
                </button>
                <button
                    onClick={() => { setActiveTab('datasheet'); setError(null); }}
                    style={{
                        padding: '0.5rem 1rem',
                        backgroundColor: activeTab === 'datasheet' ? '#952825' : 'transparent',
                        color: activeTab === 'datasheet' ? 'white' : '#666',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontWeight: '500'
                    }}
                >
                    Product Datasheet
                </button>
            </div>

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

            {/* Quotation Form */}
            {activeTab === 'quotation' && (
                <div>
                    <div style={{ marginBottom: '1.5rem' }}>
                        <h4 style={{ fontSize: '0.9rem', color: '#666', margin: '0 0 0.75rem' }}>
                            Customer Information
                        </h4>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                            <div>
                                <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                                    Customer Name *
                                </label>
                                <input
                                    type="text"
                                    value={quotationForm.customer_name}
                                    onChange={(e) => setQuotationForm(prev => ({ ...prev, customer_name: e.target.value }))}
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
                                    value={quotationForm.customer_email}
                                    onChange={(e) => setQuotationForm(prev => ({ ...prev, customer_email: e.target.value }))}
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

                    <div style={{ marginBottom: '1.5rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                            <h4 style={{ fontSize: '0.9rem', color: '#666', margin: 0 }}>Products</h4>
                            <button
                                type="button"
                                onClick={addProduct}
                                style={{
                                    padding: '0.25rem 0.5rem',
                                    backgroundColor: '#e8f5e9',
                                    border: '1px solid #c8e6c9',
                                    borderRadius: '4px',
                                    color: '#2e7d32',
                                    cursor: 'pointer',
                                    fontSize: '0.75rem'
                                }}
                            >
                                + Add Product
                            </button>
                        </div>

                        {products.map((product, index) => (
                            <div key={index} style={{
                                display: 'grid',
                                gridTemplateColumns: '1fr 1.5fr 0.5fr 0.8fr auto',
                                gap: '0.5rem',
                                marginBottom: '0.5rem',
                                alignItems: 'end'
                            }}>
                                <div>
                                    <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Code *</label>
                                    <input
                                        type="text"
                                        value={product.code}
                                        onChange={(e) => updateProduct(index, 'code', e.target.value)}
                                        placeholder="NA 701"
                                        style={{ width: '100%', padding: '0.5rem', fontSize: '0.85rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                    />
                                </div>
                                <div>
                                    <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Name</label>
                                    <input
                                        type="text"
                                        value={product.name}
                                        onChange={(e) => updateProduct(index, 'name', e.target.value)}
                                        placeholder="Graphite Packing"
                                        style={{ width: '100%', padding: '0.5rem', fontSize: '0.85rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                    />
                                </div>
                                <div>
                                    <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Qty</label>
                                    <input
                                        type="number"
                                        value={product.quantity}
                                        onChange={(e) => updateProduct(index, 'quantity', parseInt(e.target.value) || 1)}
                                        min="1"
                                        style={{ width: '100%', padding: '0.5rem', fontSize: '0.85rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                    />
                                </div>
                                <div>
                                    <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Unit Price (₹)</label>
                                    <input
                                        type="number"
                                        value={product.unit_price}
                                        onChange={(e) => updateProduct(index, 'unit_price', parseFloat(e.target.value) || 0)}
                                        min="0"
                                        step="0.01"
                                        style={{ width: '100%', padding: '0.5rem', fontSize: '0.85rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                    />
                                </div>
                                <button
                                    type="button"
                                    onClick={() => removeProduct(index)}
                                    disabled={products.length === 1}
                                    style={{
                                        padding: '0.5rem',
                                        backgroundColor: products.length > 1 ? '#ffebee' : '#f5f5f5',
                                        border: 'none',
                                        borderRadius: '4px',
                                        color: products.length > 1 ? '#c62828' : '#ccc',
                                        cursor: products.length > 1 ? 'pointer' : 'not-allowed'
                                    }}
                                >
                                    ×
                                </button>
                            </div>
                        ))}
                    </div>

                    <div style={{ marginBottom: '1.5rem' }}>
                        <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                            Notes
                        </label>
                        <textarea
                            value={quotationForm.notes}
                            onChange={(e) => setQuotationForm(prev => ({ ...prev, notes: e.target.value }))}
                            rows={3}
                            placeholder="Additional notes for the quotation..."
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

                    <button
                        onClick={generateQuotation}
                        disabled={isGenerating}
                        style={{
                            width: '100%',
                            padding: '0.875rem',
                            backgroundColor: isGenerating ? '#ccc' : '#952825',
                            border: 'none',
                            borderRadius: '6px',
                            color: 'white',
                            cursor: isGenerating ? 'not-allowed' : 'pointer',
                            fontWeight: '500',
                            fontSize: '1rem'
                        }}
                    >
                        {isGenerating ? 'Generating...' : 'Generate Quotation PDF'}
                    </button>
                </div>
            )}

            {/* Datasheet Form */}
            {activeTab === 'datasheet' && (
                <div>
                    <div style={{ marginBottom: '1rem' }}>
                        <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                            Product Code *
                        </label>
                        <input
                            type="text"
                            value={datasheetForm.product_code}
                            onChange={(e) => setDatasheetForm(prev => ({ ...prev, product_code: e.target.value }))}
                            placeholder="e.g., NA 701"
                            style={{
                                width: '100%',
                                padding: '0.625rem',
                                border: '1px solid #ddd',
                                borderRadius: '4px',
                                fontSize: '0.9rem'
                            }}
                        />
                    </div>

                    <div style={{ marginBottom: '1rem' }}>
                        <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                            Product Name (Optional)
                        </label>
                        <input
                            type="text"
                            value={datasheetForm.product_name}
                            onChange={(e) => setDatasheetForm(prev => ({ ...prev, product_name: e.target.value }))}
                            placeholder="Override product name (optional)"
                            style={{
                                width: '100%',
                                padding: '0.625rem',
                                border: '1px solid #ddd',
                                borderRadius: '4px',
                                fontSize: '0.9rem'
                            }}
                        />
                    </div>

                    <div style={{ marginBottom: '1.5rem' }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                            <input
                                type="checkbox"
                                checked={datasheetForm.include_certifications}
                                onChange={(e) => setDatasheetForm(prev => ({ ...prev, include_certifications: e.target.checked }))}
                            />
                            <span style={{ fontSize: '0.875rem', color: '#555' }}>Include certifications (API 622, ISO 15848, etc.)</span>
                        </label>
                    </div>

                    <button
                        onClick={generateDatasheet}
                        disabled={isGenerating}
                        style={{
                            width: '100%',
                            padding: '0.875rem',
                            backgroundColor: isGenerating ? '#ccc' : '#952825',
                            border: 'none',
                            borderRadius: '6px',
                            color: 'white',
                            cursor: isGenerating ? 'not-allowed' : 'pointer',
                            fontWeight: '500',
                            fontSize: '1rem'
                        }}
                    >
                        {isGenerating ? 'Generating...' : 'Generate Datasheet PDF'}
                    </button>
                </div>
            )}
        </div>
    );
}
