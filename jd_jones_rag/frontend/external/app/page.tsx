'use client';

import { useState, useRef, useEffect, FormEvent } from 'react';
import ProductSelectionWizard from './components/ProductSelectionWizard';
import EnquiryForm from './components/EnquiryForm';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

// Example questions to guide users
const EXAMPLE_QUESTIONS = [
  "What's the temperature rating for NA 701 graphite packing?",
  "Do you have API 622 certified products for valve applications?",
  "I need a sealing solution for high-pressure steam service",
  "What materials are suitable for chemical resistance?",
];

export default function ExternalPortal() {
  const [view, setView] = useState<'home' | 'chat' | 'wizard' | 'enquiry'>('home');

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleBack = () => {
    setView('home');
    setMessages([]);
    setSessionId(null);
  };

  const startChat = (initialQuestion?: string) => {
    setView('chat');
    setMessages([{
      id: '1',
      role: 'assistant',
      content: 'Hello! I\'m your JD Jones sealing solutions assistant. I can help you with technical specifications, product recommendations, pricing inquiries, and more. How can I assist you today?',
      timestamp: new Date(),
    }]);
    if (initialQuestion) {
      setInput(initialQuestion);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev: Message[]) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/external/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage.content,
          session_id: sessionId,
        }),
      });

      const data = await response.json();

      if (data.session_id) {
        setSessionId(data.session_id);
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response || data.message || 'Thank you for your enquiry. Our team will review and respond shortly.',
        timestamp: new Date(),
      };

      setMessages((prev: Message[]) => [...prev, assistantMessage]);
    } catch {
      const fallbackMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Thank you for your enquiry. Our customer service team will review your question and respond within 24 hours.',
        timestamp: new Date(),
      };
      setMessages((prev: Message[]) => [...prev, fallbackMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickQuestion = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      startChat();
      setTimeout(() => {
        handleSubmit(e);
      }, 100);
    }
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      minHeight: '100vh',
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
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div>
            <h1 style={{ color: 'white', fontSize: '1.25rem', fontWeight: '600', margin: 0 }}>
              JD Jones
            </h1>
            <span style={{ color: 'rgba(255,255,255,0.9)', fontSize: '0.75rem' }}>
              Sealing Solutions Support
            </span>
          </div>
          {view !== 'home' && (
            <button
              onClick={handleBack}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: 'rgba(255,255,255,0.1)',
                color: 'white',
                border: '1px solid rgba(255,255,255,0.3)',
                borderRadius: '4px',
                fontSize: '0.75rem',
                cursor: 'pointer',
              }}
            >
              Back to Home
            </button>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main style={{ flex: 1, padding: '2rem' }}>
        <div style={{ maxWidth: '900px', margin: '0 auto' }}>

          {/* Home View */}
          {view === 'home' && (
            <>
              <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1.75rem', color: '#2d2d2d', marginBottom: '0.5rem' }}>
                  How can we help you today?
                </h2>
                <p style={{ color: '#666', fontSize: '1rem' }}>
                  Ask any question and our AI will route it to the right specialist
                </p>
              </div>

              {/* Quick Actions */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '1rem',
                marginBottom: '1.5rem',
              }}>
                <button
                  onClick={() => setView('wizard')}
                  style={{
                    padding: '1.25rem',
                    backgroundColor: '#952825',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    textAlign: 'center',
                    cursor: 'pointer',
                    boxShadow: '0 2px 6px rgba(149, 40, 37, 0.3)',
                  }}
                >
                  <div style={{ fontWeight: '600', marginBottom: '0.25rem' }}>Product Selection Wizard</div>
                  <div style={{ fontSize: '0.75rem', opacity: 0.9 }}>Find the right product for your application</div>
                </button>
                <button
                  onClick={() => setView('enquiry')}
                  style={{
                    padding: '1.25rem',
                    backgroundColor: 'white',
                    color: '#2d2d2d',
                    border: '2px solid #952825',
                    borderRadius: '8px',
                    textAlign: 'center',
                    cursor: 'pointer',
                  }}
                >
                  <div style={{ fontWeight: '600', marginBottom: '0.25rem', color: '#952825' }}>Submit an Enquiry</div>
                  <div style={{ fontSize: '0.75rem', color: '#666' }}>Get in touch with our team</div>
                </button>
              </div>

              {/* Ask a Question */}
              <div style={{
                backgroundColor: 'white',
                border: '1px solid #e5e5e5',
                borderRadius: '8px',
                padding: '1.5rem',
                marginBottom: '1.5rem',
              }}>
                <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: '#2d2d2d' }}>
                  Ask a Question
                </h3>
                <form onSubmit={handleQuickQuestion} style={{ display: 'flex', gap: '0.75rem' }}>
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your question here..."
                    style={{
                      flex: 1,
                      padding: '0.75rem 1rem',
                      fontSize: '0.875rem',
                      border: '1px solid #e5e5e5',
                      borderRadius: '4px',
                      outline: 'none',
                    }}
                  />
                  <button
                    type="submit"
                    style={{
                      padding: '0.75rem 1.5rem',
                      backgroundColor: '#952825',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      fontSize: '0.875rem',
                      fontWeight: '500',
                      cursor: 'pointer',
                    }}
                  >
                    Ask
                  </button>
                </form>
              </div>

              {/* Example Questions */}
              <div style={{
                backgroundColor: '#fafafa',
                border: '1px solid #e5e5e5',
                borderRadius: '8px',
                padding: '1.5rem',
              }}>
                <h3 style={{ fontSize: '0.875rem', marginBottom: '1rem', color: '#666' }}>
                  Example Questions
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {EXAMPLE_QUESTIONS.map((question, i) => (
                    <button
                      key={i}
                      onClick={() => {
                        setInput(question);
                        startChat(question);
                      }}
                      style={{
                        padding: '0.75rem 1rem',
                        backgroundColor: 'white',
                        border: '1px solid #e5e5e5',
                        borderRadius: '6px',
                        fontSize: '0.875rem',
                        color: '#444',
                        cursor: 'pointer',
                        textAlign: 'left',
                        transition: 'all 0.2s',
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
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Chat View */}
          {view === 'chat' && (
            <div style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              border: '1px solid #e5e5e5',
              display: 'flex',
              flexDirection: 'column',
              height: 'calc(100vh - 200px)',
              minHeight: '500px',
            }}>
              {/* Chat Header */}
              <div style={{
                padding: '1rem 1.5rem',
                borderBottom: '1px solid #e5e5e5',
                backgroundColor: '#fafafa',
              }}>
                <h3 style={{ fontSize: '1rem', color: '#2d2d2d', margin: 0 }}>
                  JD Jones Assistant
                </h3>
                <p style={{ fontSize: '0.75rem', color: '#666', margin: '0.25rem 0 0' }}>
                  Ask about products, specifications, pricing, or technical support
                </p>
              </div>

              {/* Messages */}
              <div style={{
                flex: 1,
                overflowY: 'auto',
                padding: '1.5rem',
              }}>
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
                        padding: '0.875rem 1.125rem',
                        borderRadius: '8px',
                        backgroundColor: message.role === 'user' ? '#952825' : '#f5f5f5',
                        color: message.role === 'user' ? 'white' : '#2d2d2d',
                      }}
                    >
                      <p style={{ margin: 0, lineHeight: '1.5', whiteSpace: 'pre-wrap' }}>
                        {message.content}
                      </p>
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: '1rem' }}>
                    <div style={{
                      padding: '0.875rem 1.125rem',
                      backgroundColor: '#f5f5f5',
                      borderRadius: '8px',
                    }}>
                      <span style={{ color: '#666' }}>Thinking...</span>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div style={{
                padding: '1rem 1.5rem',
                borderTop: '1px solid #e5e5e5',
              }}>
                <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '0.75rem' }}>
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message..."
                    disabled={isLoading}
                    style={{
                      flex: 1,
                      padding: '0.75rem 1rem',
                      fontSize: '0.875rem',
                      border: '1px solid #e5e5e5',
                      borderRadius: '4px',
                      outline: 'none',
                    }}
                  />
                  <button
                    type="submit"
                    disabled={isLoading || !input.trim()}
                    style={{
                      padding: '0.75rem 1.5rem',
                      backgroundColor: isLoading || !input.trim() ? '#e5e5e5' : '#952825',
                      color: isLoading || !input.trim() ? '#666' : 'white',
                      border: 'none',
                      borderRadius: '4px',
                      fontSize: '0.875rem',
                      fontWeight: '500',
                      cursor: isLoading || !input.trim() ? 'not-allowed' : 'pointer',
                    }}
                  >
                    Send
                  </button>
                </form>
              </div>
            </div>
          )}

          {/* Product Selection Wizard View */}
          {view === 'wizard' && (
            <div style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              border: '1px solid #e5e5e5'
            }}>
              <ProductSelectionWizard />
            </div>
          )}

          {/* Enquiry Form View */}
          {view === 'enquiry' && (
            <div style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              border: '1px solid #e5e5e5'
            }}>
              <EnquiryForm />
            </div>
          )}
        </div>

      </main>

      {/* Footer */}
      <footer style={{
        padding: '1.5rem 2rem',
        backgroundColor: '#2d2d2d',
        color: 'rgba(255,255,255,0.7)',
        fontSize: '0.75rem',
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          display: 'flex',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
        }}>
          <div>
            <strong style={{ color: 'white' }}>JD Jones</strong> - Sealing Solutions
          </div>
          <div>
            For urgent enquiries, contact us at: sales@jdjones.com
          </div>
        </div>
      </footer>
    </div>
  );
}
