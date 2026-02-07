'use client';

import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface Document {
    id: string;
    filename: string;
    document_type: string;
    upload_date: string;
    department: string;
    chunk_count: number;
    status: 'processing' | 'indexed' | 'error';
}

interface DocumentManagerProps {
    apiBaseUrl?: string;
}

export default function DocumentManager({ apiBaseUrl = '/api' }: DocumentManagerProps) {
    const [documents, setDocuments] = useState<Document[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [uploadProgress, setUploadProgress] = useState<number | null>(null);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [department, setDepartment] = useState('general');

    const departments = [
        'general',
        'engineering',
        'sales',
        'manufacturing',
        'quality',
        'hr'
    ];

    useEffect(() => {
        fetchDocuments();
    }, []);

    const fetchDocuments = async () => {
        try {
            setIsLoading(true);
            const response = await axios.get(`${apiBaseUrl}/documents`);
            setDocuments(response.data.documents || []);
        } catch (err) {
            setError('Failed to fetch documents');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setSelectedFile(e.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('department', department);

        try {
            setUploadProgress(0);
            await axios.post(`${apiBaseUrl}/documents/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                onUploadProgress: (progressEvent) => {
                    if (progressEvent.total) {
                        setUploadProgress(Math.round((progressEvent.loaded * 100) / progressEvent.total));
                    }
                }
            });
            setSelectedFile(null);
            setUploadProgress(null);
            fetchDocuments();
        } catch (err) {
            setError('Upload failed');
            console.error(err);
        }
    };

    const handleDelete = async (docId: string) => {
        if (!confirm('Are you sure you want to delete this document?')) return;

        try {
            await axios.delete(`${apiBaseUrl}/documents/${docId}`);
            fetchDocuments();
        } catch (err) {
            setError('Delete failed');
            console.error(err);
        }
    };

    const formatDate = (dateStr: string) => {
        return new Date(dateStr).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    };

    const formatFileSize = (bytes: number) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };

    return (
        <div style={{
            backgroundColor: 'white',
            borderRadius: '8px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            padding: '1.5rem'
        }}>
            <h2 style={{
                fontSize: '1.25rem',
                fontWeight: '600',
                color: '#2d2d2d',
                marginBottom: '1rem',
                borderBottom: '2px solid #952825',
                paddingBottom: '0.5rem'
            }}>
                Document Manager
            </h2>

            {/* Upload Section */}
            <div style={{
                backgroundColor: '#f8f8f8',
                padding: '1rem',
                borderRadius: '6px',
                marginBottom: '1.5rem'
            }}>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
                    <div style={{ flex: '1', minWidth: '200px' }}>
                        <label style={{ display: 'block', fontSize: '0.875rem', color: '#666', marginBottom: '0.25rem' }}>
                            File
                        </label>
                        <input
                            type="file"
                            onChange={handleFileSelect}
                            accept=".pdf,.docx,.txt,.xlsx,.csv"
                            style={{
                                width: '100%',
                                padding: '0.5rem',
                                border: '1px solid #e5e5e5',
                                borderRadius: '4px',
                                backgroundColor: 'white'
                            }}
                        />
                    </div>
                    <div>
                        <label style={{ display: 'block', fontSize: '0.875rem', color: '#666', marginBottom: '0.25rem' }}>
                            Department
                        </label>
                        <select
                            value={department}
                            onChange={(e) => setDepartment(e.target.value)}
                            style={{
                                padding: '0.5rem 1rem',
                                border: '1px solid #e5e5e5',
                                borderRadius: '4px',
                                backgroundColor: 'white'
                            }}
                        >
                            {departments.map(dept => (
                                <option key={dept} value={dept}>
                                    {dept.charAt(0).toUpperCase() + dept.slice(1)}
                                </option>
                            ))}
                        </select>
                    </div>
                    <button
                        onClick={handleUpload}
                        disabled={!selectedFile || uploadProgress !== null}
                        style={{
                            padding: '0.5rem 1.5rem',
                            backgroundColor: selectedFile ? '#952825' : '#ccc',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: selectedFile ? 'pointer' : 'not-allowed',
                            fontWeight: '500'
                        }}
                    >
                        {uploadProgress !== null ? `Uploading ${uploadProgress}%` : 'Upload'}
                    </button>
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div style={{
                    padding: '0.75rem',
                    backgroundColor: '#ffebee',
                    color: '#c62828',
                    borderRadius: '4px',
                    marginBottom: '1rem',
                    fontSize: '0.875rem'
                }}>
                    {error}
                    <button
                        onClick={() => setError(null)}
                        style={{
                            float: 'right',
                            background: 'none',
                            border: 'none',
                            cursor: 'pointer',
                            color: '#c62828'
                        }}
                    >
                        ×
                    </button>
                </div>
            )}

            {/* Documents List */}
            {isLoading ? (
                <div style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
                    Loading documents...
                </div>
            ) : documents.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
                    No documents uploaded yet. Upload your first document above.
                </div>
            ) : (
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ backgroundColor: '#f5f5f5' }}>
                            <th style={{ padding: '0.75rem', textAlign: 'left', fontSize: '0.75rem', fontWeight: '600', color: '#666' }}>
                                Filename
                            </th>
                            <th style={{ padding: '0.75rem', textAlign: 'left', fontSize: '0.75rem', fontWeight: '600', color: '#666' }}>
                                Department
                            </th>
                            <th style={{ padding: '0.75rem', textAlign: 'left', fontSize: '0.75rem', fontWeight: '600', color: '#666' }}>
                                Type
                            </th>
                            <th style={{ padding: '0.75rem', textAlign: 'left', fontSize: '0.75rem', fontWeight: '600', color: '#666' }}>
                                Date
                            </th>
                            <th style={{ padding: '0.75rem', textAlign: 'left', fontSize: '0.75rem', fontWeight: '600', color: '#666' }}>
                                Status
                            </th>
                            <th style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.75rem', fontWeight: '600', color: '#666' }}>
                                Actions
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {documents.map((doc) => (
                            <tr key={doc.id} style={{ borderBottom: '1px solid #e5e5e5' }}>
                                <td style={{ padding: '0.75rem', fontSize: '0.875rem' }}>
                                    {doc.filename}
                                </td>
                                <td style={{ padding: '0.75rem', fontSize: '0.875rem' }}>
                                    <span style={{
                                        padding: '0.25rem 0.5rem',
                                        backgroundColor: '#e3f2fd',
                                        color: '#1565c0',
                                        borderRadius: '4px',
                                        fontSize: '0.75rem'
                                    }}>
                                        {doc.department}
                                    </span>
                                </td>
                                <td style={{ padding: '0.75rem', fontSize: '0.875rem', color: '#666' }}>
                                    {doc.document_type}
                                </td>
                                <td style={{ padding: '0.75rem', fontSize: '0.875rem', color: '#666' }}>
                                    {formatDate(doc.upload_date)}
                                </td>
                                <td style={{ padding: '0.75rem' }}>
                                    <span style={{
                                        padding: '0.25rem 0.5rem',
                                        borderRadius: '4px',
                                        fontSize: '0.75rem',
                                        backgroundColor: doc.status === 'indexed' ? '#e8f5e9' :
                                            doc.status === 'processing' ? '#fff3e0' : '#ffebee',
                                        color: doc.status === 'indexed' ? '#2e7d32' :
                                            doc.status === 'processing' ? '#ef6c00' : '#c62828'
                                    }}>
                                        {doc.status}
                                    </span>
                                </td>
                                <td style={{ padding: '0.75rem', textAlign: 'center' }}>
                                    <button
                                        onClick={() => handleDelete(doc.id)}
                                        style={{
                                            padding: '0.25rem 0.5rem',
                                            backgroundColor: 'transparent',
                                            color: '#c62828',
                                            border: '1px solid #c62828',
                                            borderRadius: '4px',
                                            cursor: 'pointer',
                                            fontSize: '0.75rem'
                                        }}
                                    >
                                        Delete
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    );
}
