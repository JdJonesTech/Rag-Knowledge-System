'use client';

import { useState, useRef, useEffect } from 'react';
import DocumentGenerator from './components/DocumentGenerator';
import EnquiriesDashboard from './components/EnquiriesDashboard';
import QuotationsDashboard from './components/QuotationsDashboard';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{ source?: string; file_name?: string; document?: string; relevance?: number }>;
  timestamp: Date;
}


export default function InternalPortal() {
  const [activeTab, setActiveTab] = useState<'chat' | 'documents' | 'enquiries' | 'quotations'>('chat');

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);



  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/demo/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.content,
          session_id: sessionId,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.error || 'Failed to get response');
      }

      // Store session_id for conversation continuity
      if (data.session_id) {
        setSessionId(data.session_id);
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response || 'I apologize, I could not generate a response.',
        sources: data.sources,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      backgroundColor: '#f8f8f8',
      fontFamily: "'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', Arial, sans-serif",
    }}>
      {/* Header */}
      <header style={{
        backgroundColor: '#952825',
        padding: '1rem 2rem',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <h1 style={{ color: 'white', fontSize: '1.25rem', fontWeight: '600', margin: 0 }}>
              JD Jones Knowledge Assistant
            </h1>
            <span style={{ color: 'rgba(255,255,255,0.9)', fontSize: '0.75rem' }}>
              Internal Knowledge Portal
            </span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              onClick={() => setActiveTab('chat')}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: activeTab === 'chat' ? 'rgba(255,255,255,0.2)' : 'transparent',
                color: 'white',
                border: '1px solid rgba(255,255,255,0.3)',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.875rem',
                fontWeight: activeTab === 'chat' ? '600' : '400'
              }}
            >
              Chat
            </button>
            <button
              onClick={() => setActiveTab('documents')}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: activeTab === 'documents' ? 'rgba(255,255,255,0.2)' : 'transparent',
                color: 'white',
                border: '1px solid rgba(255,255,255,0.3)',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.875rem',
                fontWeight: activeTab === 'documents' ? '600' : '400'
              }}
            >
              Documents
            </button>
            <button
              onClick={() => setActiveTab('enquiries')}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: activeTab === 'enquiries' ? 'rgba(255,255,255,0.2)' : 'transparent',
                color: 'white',
                border: '1px solid rgba(255,255,255,0.3)',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.875rem',
                fontWeight: activeTab === 'enquiries' ? '600' : '400'
              }}
            >
              Enquiries
            </button>
            <button
              onClick={() => setActiveTab('quotations')}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: activeTab === 'quotations' ? 'rgba(255,255,255,0.2)' : 'transparent',
                color: 'white',
                border: '1px solid rgba(255,255,255,0.3)',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.875rem',
                fontWeight: activeTab === 'quotations' ? '600' : '400'
              }}
            >
              Quotations
            </button>
          </div>

        </div>
      </header>


      {/* Main Content Area */}
      <main style={{
        flex: 1,
        overflowY: 'auto',
        padding: '2rem',
      }}>
        {/* Documents Tab */}
        {activeTab === 'documents' ? (
          <div style={{ maxWidth: '800px', margin: '0 auto' }}>
            <div style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              border: '1px solid #e5e5e5'
            }}>
              <DocumentGenerator />
            </div>
          </div>
        ) : activeTab === 'enquiries' ? (
          /* Enquiries Tab */
          <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
            <div style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              border: '1px solid #e5e5e5'
            }}>
              <EnquiriesDashboard />
            </div>
          </div>
        ) : activeTab === 'quotations' ? (
          /* Quotations Tab */
          <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
            <div style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              border: '1px solid #e5e5e5'
            }}>
              <QuotationsDashboard />
            </div>
          </div>
        ) : (
          /* Chat Tab */
          <div style={{ maxWidth: '800px', margin: '0 auto' }}>
            {messages.length === 0 && (


              <div style={{ textAlign: 'center', padding: '3rem 1rem' }}>
                <h2 style={{ fontSize: '1.5rem', color: '#2d2d2d', marginBottom: '0.5rem' }}>
                  Welcome to JD Jones Knowledge Portal
                </h2>
                <p style={{ color: '#666', marginBottom: '2rem' }}>
                  Ask questions about products, procedures, or company information.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', alignItems: 'center' }}>
                  {[
                    'What products are best for high-temperature applications?',
                    'Tell me about NA 701 graphite packing',
                    'What valve packing do you recommend for steam service?',
                  ].map((suggestion, i) => (
                    <button
                      key={i}
                      onClick={() => setInput(suggestion)}
                      style={{
                        padding: '0.75rem 1.25rem',
                        backgroundColor: 'white',
                        border: '1px solid #e5e5e5',
                        borderRadius: '6px',
                        fontSize: '0.875rem',
                        color: '#444',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        maxWidth: '500px',
                        width: '100%',
                        textAlign: 'left',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor = '#952825';
                        e.currentTarget.style.backgroundColor = '#fdf8f8';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor = '#e5e5e5';
                        e.currentTarget.style.backgroundColor = 'white';
                      }}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((message) => (
              <div
                key={message.id}
                style={{
                  display: 'flex',
                  justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
                  marginBottom: '1rem',
                }}
              >
                <div
                  style={{
                    maxWidth: '75%',
                    padding: '1rem 1.25rem',
                    borderRadius: '8px',
                    backgroundColor: message.role === 'user' ? '#952825' : 'white',
                    color: message.role === 'user' ? 'white' : '#2d2d2d',
                    boxShadow: message.role === 'assistant' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                    border: message.role === 'assistant' ? '1px solid #e5e5e5' : 'none',
                  }}
                >
                  <p style={{ margin: 0, whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
                    {message.content}
                  </p>
                  {message.sources && message.sources.length > 0 && (
                    <div style={{
                      marginTop: '0.75rem',
                      paddingTop: '0.75rem',
                      borderTop: '1px solid #e5e5e5',
                      fontSize: '0.75rem',
                      color: '#666',
                    }}>
                      <strong>Sources:</strong>
                      <ul style={{ margin: '0.25rem 0 0', paddingLeft: '1rem' }}>
                        {message.sources.slice(0, 3).map((src, i) => (
                          <li key={i}>{src.file_name || src.source || src.document || 'Unknown source'}</li>
                        ))}

                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isLoading && (
              <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: '1rem' }}>
                <div style={{
                  padding: '1rem 1.25rem',
                  backgroundColor: 'white',
                  borderRadius: '8px',
                  border: '1px solid #e5e5e5',
                }}>
                  <span style={{ color: '#666' }}>Thinking...</span>
                </div>
              </div>
            )}

            {error && (
              <div style={{
                padding: '1rem',
                backgroundColor: '#ffebee',
                border: '1px solid #ffcdd2',
                borderRadius: '4px',
                color: '#c62828',
                fontSize: '0.875rem',
                marginBottom: '1rem',
              }}>
                {error}
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </main>


      {/* Input Area - only show for chat tab */}
      {activeTab === 'chat' && (
        <footer style={{
          padding: '1rem 2rem',
          backgroundColor: 'white',
          borderTop: '1px solid #e5e5e5',
        }}>

          <form
            onSubmit={handleSubmit}
            style={{
              maxWidth: '800px',
              margin: '0 auto',
              display: 'flex',
              gap: '0.75rem',
            }}
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your question here..."
              disabled={isLoading}
              style={{
                flex: 1,
                padding: '0.875rem 1rem',
                fontSize: '0.875rem',
                border: '1px solid #e5e5e5',
                borderRadius: '6px',
                outline: 'none',
              }}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              style={{
                padding: '0.875rem 1.5rem',
                backgroundColor: isLoading || !input.trim() ? '#e5e5e5' : '#952825',
                color: isLoading || !input.trim() ? '#666' : 'white',
                border: 'none',
                borderRadius: '6px',
                fontSize: '0.875rem',
                fontWeight: '500',
                cursor: isLoading || !input.trim() ? 'not-allowed' : 'pointer',
              }}
            >
              Send
            </button>
          </form>
        </footer>
      )}
    </div>

  );
}
