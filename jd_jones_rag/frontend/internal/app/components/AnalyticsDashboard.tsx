'use client';

import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface AnalyticsData {
    total_queries: number;
    total_documents: number;
    total_chunks: number;
    cache_hit_rate: number;
    avg_response_time_ms: number;
    queries_by_day: { date: string; count: number }[];
    top_topics: { topic: string; count: number }[];
}

interface AnalyticsDashboardProps {
    apiBaseUrl?: string;
}

export default function AnalyticsDashboard({ apiBaseUrl = '/api' }: AnalyticsDashboardProps) {
    const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('7d');

    useEffect(() => {
        fetchAnalytics();
    }, [timeRange]);

    const fetchAnalytics = async () => {
        try {
            setIsLoading(true);
            const response = await axios.get(`${apiBaseUrl}/analytics?range=${timeRange}`);
            setAnalytics(response.data);
        } catch (err) {
            // Use mock data for demonstration
            setAnalytics({
                total_queries: 1247,
                total_documents: 156,
                total_chunks: 4523,
                cache_hit_rate: 0.67,
                avg_response_time_ms: 342,
                queries_by_day: [
                    { date: '2026-01-28', count: 145 },
                    { date: '2026-01-29', count: 178 },
                    { date: '2026-01-30', count: 156 },
                    { date: '2026-01-31', count: 201 },
                    { date: '2026-02-01', count: 189 },
                    { date: '2026-02-02', count: 167 },
                    { date: '2026-02-03', count: 211 },
                ],
                top_topics: [
                    { topic: 'NA 701 Packing', count: 89 },
                    { topic: 'High Temperature Sealing', count: 67 },
                    { topic: 'API 622 Compliance', count: 54 },
                    { topic: 'PTFE Gaskets', count: 48 },
                    { topic: 'Valve Stem Packing', count: 41 },
                ]
            });
        } finally {
            setIsLoading(false);
        }
    };

    const StatCard = ({ title, value, subtitle, color = '#952825' }: {
        title: string;
        value: string | number;
        subtitle?: string;
        color?: string;
    }) => (
        <div style={{
            backgroundColor: 'white',
            borderRadius: '8px',
            padding: '1.25rem',
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            borderTop: `3px solid ${color}`
        }}>
            <div style={{ fontSize: '0.75rem', color: '#666', marginBottom: '0.5rem' }}>
                {title}
            </div>
            <div style={{ fontSize: '1.75rem', fontWeight: '700', color: '#2d2d2d' }}>
                {value}
            </div>
            {subtitle && (
                <div style={{ fontSize: '0.75rem', color: '#999', marginTop: '0.25rem' }}>
                    {subtitle}
                </div>
            )}
        </div>
    );

    if (isLoading) {
        return (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
                Loading analytics...
            </div>
        );
    }

    if (!analytics) return null;

    const maxQueryCount = Math.max(...analytics.queries_by_day.map(d => d.count));

    return (
        <div>
            {/* Header with Time Range */}
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '1.5rem'
            }}>
                <h2 style={{
                    fontSize: '1.25rem',
                    fontWeight: '600',
                    color: '#2d2d2d',
                    margin: 0
                }}>
                    Analytics Dashboard
                </h2>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                    {(['7d', '30d', '90d'] as const).map(range => (
                        <button
                            key={range}
                            onClick={() => setTimeRange(range)}
                            style={{
                                padding: '0.5rem 1rem',
                                backgroundColor: timeRange === range ? '#952825' : 'white',
                                color: timeRange === range ? 'white' : '#666',
                                border: '1px solid #e5e5e5',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '0.875rem'
                            }}
                        >
                            {range === '7d' ? '7 Days' : range === '30d' ? '30 Days' : '90 Days'}
                        </button>
                    ))}
                </div>
            </div>

            {/* Stats Grid */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '1rem',
                marginBottom: '1.5rem'
            }}>
                <StatCard
                    title="Total Queries"
                    value={analytics.total_queries.toLocaleString()}
                    subtitle="All time"
                />
                <StatCard
                    title="Documents"
                    value={analytics.total_documents}
                    subtitle={`${analytics.total_chunks.toLocaleString()} chunks`}
                    color="#1565c0"
                />
                <StatCard
                    title="Cache Hit Rate"
                    value={`${(analytics.cache_hit_rate * 100).toFixed(1)}%`}
                    subtitle="FAQ + Embedding cache"
                    color="#2e7d32"
                />
                <StatCard
                    title="Avg Response Time"
                    value={`${analytics.avg_response_time_ms}ms`}
                    subtitle="P50 latency"
                    color="#7b1fa2"
                />
            </div>

            {/* Charts Row */}
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1rem' }}>
                {/* Query Volume Chart */}
                <div style={{
                    backgroundColor: 'white',
                    borderRadius: '8px',
                    padding: '1.25rem',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                }}>
                    <h3 style={{ fontSize: '0.875rem', fontWeight: '600', color: '#666', marginBottom: '1rem' }}>
                        Query Volume
                    </h3>
                    <div style={{ display: 'flex', alignItems: 'flex-end', gap: '4px', height: '150px' }}>
                        {analytics.queries_by_day.map((day, i) => (
                            <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                                <div
                                    style={{
                                        width: '100%',
                                        height: `${(day.count / maxQueryCount) * 120}px`,
                                        backgroundColor: '#952825',
                                        borderRadius: '4px 4px 0 0',
                                        opacity: 0.8 + (i * 0.03)
                                    }}
                                    title={`${day.date}: ${day.count} queries`}
                                />
                                <div style={{ fontSize: '0.625rem', color: '#999', marginTop: '4px' }}>
                                    {new Date(day.date).toLocaleDateString('en-US', { weekday: 'short' })}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Top Topics */}
                <div style={{
                    backgroundColor: 'white',
                    borderRadius: '8px',
                    padding: '1.25rem',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                }}>
                    <h3 style={{ fontSize: '0.875rem', fontWeight: '600', color: '#666', marginBottom: '1rem' }}>
                        Top Topics
                    </h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        {analytics.top_topics.map((topic, i) => (
                            <div key={i}>
                                <div style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    fontSize: '0.75rem',
                                    marginBottom: '0.25rem'
                                }}>
                                    <span style={{ color: '#2d2d2d' }}>{topic.topic}</span>
                                    <span style={{ color: '#666' }}>{topic.count}</span>
                                </div>
                                <div style={{
                                    height: '4px',
                                    backgroundColor: '#f0f0f0',
                                    borderRadius: '2px',
                                    overflow: 'hidden'
                                }}>
                                    <div style={{
                                        height: '100%',
                                        width: `${(topic.count / analytics.top_topics[0].count) * 100}%`,
                                        backgroundColor: '#952825',
                                        borderRadius: '2px'
                                    }} />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
