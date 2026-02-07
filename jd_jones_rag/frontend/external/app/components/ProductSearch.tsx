'use client';

import React, { useState } from 'react';

interface Product {
    id: string;
    code: string;
    name: string;
    category: string;
    material: string;
    temperature_range: string;
    pressure_range: string;
    applications: string[];
    standards: string[];
    description: string;
}

// Sample product data for demonstration
const SAMPLE_PRODUCTS: Product[] = [
    {
        id: '1',
        code: 'NA 701',
        name: 'Flexible Graphite Packing',
        category: 'Valve Packing',
        material: 'Expanded Graphite',
        temperature_range: '-200°C to +650°C',
        pressure_range: 'Up to 400 bar',
        applications: ['Valve stems', 'Pumps', 'High temperature service'],
        standards: ['API 622', 'ISO 15848-1'],
        description: 'High-performance flexible graphite packing for demanding applications.'
    },
    {
        id: '2',
        code: 'NA 702',
        name: 'Graphite with PTFE Corners',
        category: 'Valve Packing',
        material: 'Graphite + PTFE',
        temperature_range: '-100°C to +280°C',
        pressure_range: 'Up to 250 bar',
        applications: ['Chemical processing', 'Moderate temperature valves'],
        standards: ['API 622'],
        description: 'Combines graphite performance with PTFE lubricity.'
    },
    {
        id: '3',
        code: 'NA 750',
        name: 'Carbon Fiber Reinforced Packing',
        category: 'Valve Packing',
        material: 'Carbon Fiber + Graphite',
        temperature_range: '-40°C to +450°C',
        pressure_range: 'Up to 350 bar',
        applications: ['Steam service', 'Rotating equipment'],
        standards: ['API 622', 'Shell SPE'],
        description: 'Carbon fiber reinforcement for enhanced durability.'
    },
    {
        id: '4',
        code: 'FLEXSEAL 100',
        name: 'Universal Gasket Sheet',
        category: 'Gaskets',
        material: 'Compressed Fiber',
        temperature_range: '-50°C to +200°C',
        pressure_range: 'Up to 50 bar',
        applications: ['General industrial', 'HVAC', 'Water systems'],
        standards: ['DIN 28091'],
        description: 'Versatile gasket material for general purpose sealing.'
    },
    {
        id: '5',
        code: 'THERMEX 500',
        name: 'High Temperature Gasket',
        category: 'Gaskets',
        material: 'Ceramic Fiber',
        temperature_range: '-40°C to +1260°C',
        pressure_range: 'Up to 100 bar',
        applications: ['Furnaces', 'Kilns', 'Exhaust systems'],
        standards: ['ASTM F37'],
        description: 'Extreme temperature resistance for industrial applications.'
    }
];

interface ProductSearchProps {
    onProductSelect?: (product: Product) => void;
}

export default function ProductSearch({ onProductSelect }: ProductSearchProps) {
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedCategory, setSelectedCategory] = useState<string>('all');
    const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);

    const categories = ['all', ...Array.from(new Set(SAMPLE_PRODUCTS.map(p => p.category)))];

    const filteredProducts = SAMPLE_PRODUCTS.filter(product => {
        const matchesSearch = searchQuery === '' ||
            product.code.toLowerCase().includes(searchQuery.toLowerCase()) ||
            product.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            product.material.toLowerCase().includes(searchQuery.toLowerCase()) ||
            product.applications.some(app => app.toLowerCase().includes(searchQuery.toLowerCase()));

        const matchesCategory = selectedCategory === 'all' || product.category === selectedCategory;

        return matchesSearch && matchesCategory;
    });

    const handleProductClick = (product: Product) => {
        setSelectedProduct(product);
        onProductSelect?.(product);
    };

    return (
        <div style={{
            backgroundColor: 'white',
            borderRadius: '8px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            overflow: 'hidden'
        }}>
            {/* Search Header */}
            <div style={{
                padding: '1.25rem',
                borderBottom: '1px solid #e5e5e5',
                backgroundColor: '#fafafa'
            }}>
                <h2 style={{
                    fontSize: '1.125rem',
                    fontWeight: '600',
                    color: '#2d2d2d',
                    marginBottom: '1rem'
                }}>
                    Product Search
                </h2>
                <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                    <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search by product code, name, or application..."
                        style={{
                            flex: '1',
                            minWidth: '200px',
                            padding: '0.75rem 1rem',
                            border: '1px solid #e5e5e5',
                            borderRadius: '4px',
                            fontSize: '0.875rem',
                            outline: 'none'
                        }}
                    />
                    <select
                        value={selectedCategory}
                        onChange={(e) => setSelectedCategory(e.target.value)}
                        style={{
                            padding: '0.75rem 1rem',
                            border: '1px solid #e5e5e5',
                            borderRadius: '4px',
                            fontSize: '0.875rem',
                            backgroundColor: 'white'
                        }}
                    >
                        {categories.map(cat => (
                            <option key={cat} value={cat}>
                                {cat === 'all' ? 'All Categories' : cat}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {/* Results */}
            <div style={{ display: 'flex', minHeight: '400px' }}>
                {/* Product List */}
                <div style={{
                    flex: '1',
                    borderRight: selectedProduct ? '1px solid #e5e5e5' : 'none',
                    overflowY: 'auto',
                    maxHeight: '500px'
                }}>
                    {filteredProducts.length === 0 ? (
                        <div style={{ padding: '2rem', textAlign: 'center', color: '#666' }}>
                            No products found matching your search.
                        </div>
                    ) : (
                        filteredProducts.map(product => (
                            <div
                                key={product.id}
                                onClick={() => handleProductClick(product)}
                                style={{
                                    padding: '1rem 1.25rem',
                                    borderBottom: '1px solid #f0f0f0',
                                    cursor: 'pointer',
                                    backgroundColor: selectedProduct?.id === product.id ? '#fdf8f8' : 'white',
                                    borderLeft: selectedProduct?.id === product.id ? '3px solid #952825' : '3px solid transparent',
                                    transition: 'all 0.2s'
                                }}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                    <div>
                                        <div style={{
                                            fontSize: '0.875rem',
                                            fontWeight: '600',
                                            color: '#952825'
                                        }}>
                                            {product.code}
                                        </div>
                                        <div style={{
                                            fontSize: '0.875rem',
                                            color: '#2d2d2d',
                                            marginTop: '0.25rem'
                                        }}>
                                            {product.name}
                                        </div>
                                    </div>
                                    <span style={{
                                        padding: '0.25rem 0.5rem',
                                        backgroundColor: '#e8f5e9',
                                        color: '#2e7d32',
                                        borderRadius: '4px',
                                        fontSize: '0.625rem',
                                        fontWeight: '600'
                                    }}>
                                        {product.category}
                                    </span>
                                </div>
                                <div style={{
                                    fontSize: '0.75rem',
                                    color: '#666',
                                    marginTop: '0.5rem'
                                }}>
                                    {product.material} • {product.temperature_range}
                                </div>
                            </div>
                        ))
                    )}
                </div>

                {/* Product Detail */}
                {selectedProduct && (
                    <div style={{
                        flex: '1',
                        padding: '1.5rem',
                        overflowY: 'auto'
                    }}>
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'flex-start',
                            marginBottom: '1rem'
                        }}>
                            <div>
                                <h3 style={{
                                    fontSize: '1.5rem',
                                    fontWeight: '700',
                                    color: '#952825',
                                    margin: 0
                                }}>
                                    {selectedProduct.code}
                                </h3>
                                <p style={{
                                    fontSize: '1rem',
                                    color: '#2d2d2d',
                                    margin: '0.25rem 0 0'
                                }}>
                                    {selectedProduct.name}
                                </p>
                            </div>
                            <button
                                onClick={() => setSelectedProduct(null)}
                                style={{
                                    background: 'none',
                                    border: 'none',
                                    fontSize: '1.5rem',
                                    color: '#999',
                                    cursor: 'pointer'
                                }}
                            >
                                ×
                            </button>
                        </div>

                        <p style={{
                            fontSize: '0.875rem',
                            color: '#666',
                            lineHeight: '1.6',
                            marginBottom: '1.5rem'
                        }}>
                            {selectedProduct.description}
                        </p>

                        {/* Specifications */}
                        <div style={{ marginBottom: '1.5rem' }}>
                            <h4 style={{
                                fontSize: '0.75rem',
                                fontWeight: '600',
                                color: '#999',
                                textTransform: 'uppercase',
                                marginBottom: '0.75rem'
                            }}>
                                Specifications
                            </h4>
                            <div style={{
                                display: 'grid',
                                gridTemplateColumns: '1fr 1fr',
                                gap: '0.75rem'
                            }}>
                                <div style={{
                                    padding: '0.75rem',
                                    backgroundColor: '#f8f8f8',
                                    borderRadius: '4px'
                                }}>
                                    <div style={{ fontSize: '0.625rem', color: '#999' }}>Material</div>
                                    <div style={{ fontSize: '0.875rem', color: '#2d2d2d' }}>{selectedProduct.material}</div>
                                </div>
                                <div style={{
                                    padding: '0.75rem',
                                    backgroundColor: '#f8f8f8',
                                    borderRadius: '4px'
                                }}>
                                    <div style={{ fontSize: '0.625rem', color: '#999' }}>Temperature</div>
                                    <div style={{ fontSize: '0.875rem', color: '#2d2d2d' }}>{selectedProduct.temperature_range}</div>
                                </div>
                                <div style={{
                                    padding: '0.75rem',
                                    backgroundColor: '#f8f8f8',
                                    borderRadius: '4px'
                                }}>
                                    <div style={{ fontSize: '0.625rem', color: '#999' }}>Pressure</div>
                                    <div style={{ fontSize: '0.875rem', color: '#2d2d2d' }}>{selectedProduct.pressure_range}</div>
                                </div>
                                <div style={{
                                    padding: '0.75rem',
                                    backgroundColor: '#f8f8f8',
                                    borderRadius: '4px'
                                }}>
                                    <div style={{ fontSize: '0.625rem', color: '#999' }}>Category</div>
                                    <div style={{ fontSize: '0.875rem', color: '#2d2d2d' }}>{selectedProduct.category}</div>
                                </div>
                            </div>
                        </div>

                        {/* Applications */}
                        <div style={{ marginBottom: '1.5rem' }}>
                            <h4 style={{
                                fontSize: '0.75rem',
                                fontWeight: '600',
                                color: '#999',
                                textTransform: 'uppercase',
                                marginBottom: '0.75rem'
                            }}>
                                Applications
                            </h4>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                                {selectedProduct.applications.map((app, i) => (
                                    <span key={i} style={{
                                        padding: '0.375rem 0.75rem',
                                        backgroundColor: '#e3f2fd',
                                        color: '#1565c0',
                                        borderRadius: '16px',
                                        fontSize: '0.75rem'
                                    }}>
                                        {app}
                                    </span>
                                ))}
                            </div>
                        </div>

                        {/* Standards */}
                        <div style={{ marginBottom: '1.5rem' }}>
                            <h4 style={{
                                fontSize: '0.75rem',
                                fontWeight: '600',
                                color: '#999',
                                textTransform: 'uppercase',
                                marginBottom: '0.75rem'
                            }}>
                                Standards & Certifications
                            </h4>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                                {selectedProduct.standards.map((std, i) => (
                                    <span key={i} style={{
                                        padding: '0.375rem 0.75rem',
                                        backgroundColor: '#fff3e0',
                                        color: '#e65100',
                                        borderRadius: '16px',
                                        fontSize: '0.75rem'
                                    }}>
                                        {std}
                                    </span>
                                ))}
                            </div>
                        </div>

                        {/* Contact Button */}
                        <button style={{
                            width: '100%',
                            padding: '0.875rem',
                            backgroundColor: '#952825',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            fontSize: '0.875rem',
                            fontWeight: '600',
                            cursor: 'pointer'
                        }}>
                            Request Quote for {selectedProduct.code}
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
