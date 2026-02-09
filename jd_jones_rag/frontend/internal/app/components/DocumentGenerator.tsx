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
    material_code: string;
    material_grade: string;
    size_od: string;
    size_id: string;
    size_th: string;
    dimension_unit: string;
    rings_per_set: string;
    quantity: number;
    unit: string;
    unit_price: number;
    style: string;
    colour: string;
    specific_requirements: string;
    item_notes: string;
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
        customer_company: '',
        customer_phone: '',
        customer_address: '',
        customer_designation: '',
        rfq_number: '',
        notes: '',
        validity_days: 30
    });
    const emptyProduct: ProductItem = {
        code: '', name: '', material_code: '', material_grade: '',
        size_od: '', size_id: '', size_th: '', dimension_unit: 'mm',
        rings_per_set: '', quantity: 1, unit: 'Nos.', unit_price: 0,
        style: '', colour: '', specific_requirements: '', item_notes: ''
    };
    const [products, setProducts] = useState<ProductItem[]>([{ ...emptyProduct }]);

    // Datasheet form state
    const [datasheetForm, setDatasheetForm] = useState({
        product_code: '',
        product_name: '',
        include_certifications: true
    });

    const addProduct = () => {
        setProducts([...products, { ...emptyProduct }]);
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
                    customer_company: quotationForm.customer_company || undefined,
                    customer_designation: quotationForm.customer_designation || undefined,
                    customer_address: quotationForm.customer_address || undefined,
                    rfq_number: quotationForm.rfq_number || undefined,
                    products: products.map(p => ({
                        code: p.code,
                        name: p.name,
                        material_code: p.material_code || undefined,
                        material_grade: p.material_grade || undefined,
                        size: (p.size_od || p.size_id || p.size_th) ? {
                            od: p.size_od || '-',
                            id: p.size_id || '-',
                            th: p.size_th || '-'
                        } : undefined,
                        dimension_unit: p.dimension_unit || 'mm',
                        rings_per_set: p.rings_per_set ? parseInt(p.rings_per_set) : undefined,
                        quantity: p.quantity,
                        unit: p.unit || 'Nos.',
                        unit_price: p.unit_price,
                        style: p.style || undefined,
                        colour: p.colour || undefined,
                        specific_requirements: p.specific_requirements || undefined,
                        notes: p.item_notes || undefined
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
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem' }}>
                            <div>
                                <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                                    Customer Name *
                                </label>
                                <input
                                    type="text"
                                    value={quotationForm.customer_name}
                                    onChange={(e) => setQuotationForm(prev => ({ ...prev, customer_name: e.target.value }))}
                                    style={{ width: '100%', padding: '0.5rem', border: '1px solid #ddd', borderRadius: '4px', fontSize: '0.85rem' }}
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
                                    style={{ width: '100%', padding: '0.5rem', border: '1px solid #ddd', borderRadius: '4px', fontSize: '0.85rem' }}
                                />
                            </div>
                            <div>
                                <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                                    Company
                                </label>
                                <input
                                    type="text"
                                    value={quotationForm.customer_company}
                                    onChange={(e) => setQuotationForm(prev => ({ ...prev, customer_company: e.target.value }))}
                                    style={{ width: '100%', padding: '0.5rem', border: '1px solid #ddd', borderRadius: '4px', fontSize: '0.85rem' }}
                                />
                            </div>
                            <div>
                                <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                                    Phone
                                </label>
                                <input
                                    type="tel"
                                    value={quotationForm.customer_phone}
                                    onChange={(e) => setQuotationForm(prev => ({ ...prev, customer_phone: e.target.value }))}
                                    style={{ width: '100%', padding: '0.5rem', border: '1px solid #ddd', borderRadius: '4px', fontSize: '0.85rem' }}
                                />
                            </div>
                            <div>
                                <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                                    Designation
                                </label>
                                <input
                                    type="text"
                                    value={quotationForm.customer_designation}
                                    onChange={(e) => setQuotationForm(prev => ({ ...prev, customer_designation: e.target.value }))}
                                    style={{ width: '100%', padding: '0.5rem', border: '1px solid #ddd', borderRadius: '4px', fontSize: '0.85rem' }}
                                />
                            </div>
                            <div>
                                <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                                    Reference RFQ No.
                                </label>
                                <input
                                    type="text"
                                    value={quotationForm.rfq_number}
                                    onChange={(e) => setQuotationForm(prev => ({ ...prev, rfq_number: e.target.value }))}
                                    placeholder="e.g. RFQ-2026-001"
                                    style={{ width: '100%', padding: '0.5rem', border: '1px solid #ddd', borderRadius: '4px', fontSize: '0.85rem' }}
                                />
                            </div>
                        </div>
                        <div style={{ marginTop: '0.75rem' }}>
                            <label style={{ display: 'block', fontSize: '0.8rem', color: '#555', marginBottom: '0.25rem' }}>
                                Address
                            </label>
                            <textarea
                                value={quotationForm.customer_address}
                                onChange={(e) => setQuotationForm(prev => ({ ...prev, customer_address: e.target.value }))}
                                rows={2}
                                placeholder="Customer address"
                                style={{ width: '100%', padding: '0.5rem', border: '1px solid #ddd', borderRadius: '4px', fontSize: '0.85rem', resize: 'vertical' }}
                            />
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
                                border: '1px solid #e5e5e5',
                                borderRadius: '6px',
                                padding: '0.75rem',
                                marginBottom: '0.75rem',
                                backgroundColor: '#fafafa',
                                position: 'relative'
                            }}>
                                {/* Remove button */}
                                <button
                                    type="button"
                                    onClick={() => removeProduct(index)}
                                    disabled={products.length === 1}
                                    style={{
                                        position: 'absolute', top: '0.5rem', right: '0.5rem',
                                        padding: '0.25rem 0.5rem',
                                        backgroundColor: products.length > 1 ? '#ffebee' : '#f5f5f5',
                                        border: 'none', borderRadius: '4px',
                                        color: products.length > 1 ? '#c62828' : '#ccc',
                                        cursor: products.length > 1 ? 'pointer' : 'not-allowed',
                                        fontSize: '0.85rem'
                                    }}
                                >
                                    ×
                                </button>
                                <div style={{ fontSize: '0.7rem', color: '#999', marginBottom: '0.5rem', fontWeight: '600' }}>
                                    Item {index + 1}
                                </div>
                                {/* Row 1: Code, Name, Material Code, Material Grade */}
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr 1fr 1fr', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>JDJ Style / Code *</label>
                                        <input type="text" value={product.code}
                                            onChange={(e) => updateProduct(index, 'code', e.target.value)}
                                            placeholder="NA 701"
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                        />
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Product Name</label>
                                        <input type="text" value={product.name}
                                            onChange={(e) => updateProduct(index, 'name', e.target.value)}
                                            placeholder="Graphite Packing"
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                        />
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Material Code</label>
                                        <input type="text" value={product.material_code}
                                            onChange={(e) => updateProduct(index, 'material_code', e.target.value)}
                                            placeholder="e.g. C/PTFE"
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                        />
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Material Grade</label>
                                        <select value={product.material_grade}
                                            onChange={(e) => updateProduct(index, 'material_grade', e.target.value)}
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px', backgroundColor: 'white' }}
                                        >
                                            <option value="">Select Grade</option>
                                            <option value="Standard">Standard</option>
                                            <option value="High Purity">High Purity</option>
                                            <option value="Nuclear Grade">Nuclear Grade</option>
                                            <option value="FDA Approved">FDA Approved</option>
                                        </select>
                                    </div>
                                </div>
                                {/* Row 2: Size OD, ID, TH, Unit (mm/inch), Rings/Set */}
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 0.8fr 0.8fr', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Size OD</label>
                                        <input type="text" value={product.size_od}
                                            onChange={(e) => updateProduct(index, 'size_od', e.target.value)}
                                            placeholder="e.g. 25"
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                        />
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Size ID</label>
                                        <input type="text" value={product.size_id}
                                            onChange={(e) => updateProduct(index, 'size_id', e.target.value)}
                                            placeholder="e.g. 15"
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                        />
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Size TH</label>
                                        <input type="text" value={product.size_th}
                                            onChange={(e) => updateProduct(index, 'size_th', e.target.value)}
                                            placeholder="e.g. 5"
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                        />
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Dim Unit</label>
                                        <select value={product.dimension_unit}
                                            onChange={(e) => updateProduct(index, 'dimension_unit', e.target.value)}
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px', backgroundColor: 'white' }}
                                        >
                                            <option value="mm">mm</option>
                                            <option value="inch">inch</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Rings/Set</label>
                                        <input type="text" value={product.rings_per_set}
                                            onChange={(e) => updateProduct(index, 'rings_per_set', e.target.value)}
                                            placeholder="e.g. 5"
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                        />
                                    </div>
                                </div>
                                {/* Row 3: Qty, Unit, Unit Price, Style, Colour */}
                                <div style={{ display: 'grid', gridTemplateColumns: '0.7fr 0.8fr 1fr 1fr 0.8fr', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Qty *</label>
                                        <input type="number" value={product.quantity}
                                            onChange={(e) => updateProduct(index, 'quantity', parseInt(e.target.value) || 1)}
                                            min="1"
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                        />
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Unit</label>
                                        <select value={product.unit}
                                            onChange={(e) => updateProduct(index, 'unit', e.target.value)}
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px', backgroundColor: 'white' }}
                                        >
                                            <option value="Nos.">Nos.</option>
                                            <option value="Set">Set</option>
                                            <option value="Mtr.">Mtr.</option>
                                            <option value="Kg.">Kg.</option>
                                            <option value="Pcs.">Pcs.</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Unit Price (INR) *</label>
                                        <input type="number" value={product.unit_price}
                                            onChange={(e) => updateProduct(index, 'unit_price', parseFloat(e.target.value) || 0)}
                                            min="0" step="0.01"
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                        />
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Style</label>
                                        <select value={product.style}
                                            onChange={(e) => updateProduct(index, 'style', e.target.value)}
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px', backgroundColor: 'white' }}
                                        >
                                            <option value="">Select Style</option>
                                            <option value="Braided">Braided</option>
                                            <option value="Die-formed">Die-formed</option>
                                            <option value="Wrapped">Wrapped</option>
                                            <option value="Moulded">Moulded</option>
                                            <option value="Cut">Cut</option>
                                            <option value="Spiral Wound">Spiral Wound</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Colour</label>
                                        <input type="text" value={product.colour}
                                            onChange={(e) => updateProduct(index, 'colour', e.target.value)}
                                            placeholder="e.g. Grey"
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                        />
                                    </div>
                                </div>
                                {/* Row 4: Specific Requirements & Notes */}
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Specific Requirements</label>
                                        <input type="text" value={product.specific_requirements}
                                            onChange={(e) => updateProduct(index, 'specific_requirements', e.target.value)}
                                            placeholder="e.g. Fire-safe, API 622"
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                        />
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.7rem', color: '#666' }}>Item Notes</label>
                                        <input type="text" value={product.item_notes}
                                            onChange={(e) => updateProduct(index, 'item_notes', e.target.value)}
                                            placeholder="Additional notes for this item"
                                            style={{ width: '100%', padding: '0.4rem', fontSize: '0.8rem', border: '1px solid #ddd', borderRadius: '4px' }}
                                        />
                                    </div>
                                </div>
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
