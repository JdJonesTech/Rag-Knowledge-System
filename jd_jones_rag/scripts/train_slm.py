"""
Enhanced SLM Training Script for High Confidence (>98%)
Generates more training examples and uses optimized hyperparameters.
"""

import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_company_data():
    """Load ALL JD Jones data from both JSON files."""
    documents = []
    
    # File 1: Scraped data
    scraped_path = Path("data/scraped_jd_jones.json")
    if scraped_path.exists():
        with open(scraped_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} documents from scraped_jd_jones.json")
        documents.extend(data)
    
    # File 2: Product knowledge base
    products_path = Path("data/knowledge_base/jd_jones_products.json")
    if products_path.exists():
        with open(products_path, "r", encoding="utf-8") as f:
            product_data = json.load(f)
        
        # Convert product data to document format
        products = product_data.get("products", [])
        for p in products:
            doc = {
                "content": f"Product: {p.get('code', '')} - {p.get('name', '')}. {p.get('description', '')}. Features: {', '.join(p.get('features', []))}. Applications: {', '.join(p.get('applications', []))}. Certifications: {', '.join(p.get('certifications', []))}",
                "metadata": {"type": "product", "code": p.get("code")}
            }
            documents.append(doc)
        
        # Add company info
        company = product_data.get("company_info", {})
        if company:
            documents.append({
                "content": f"Company: {company.get('name')} - {company.get('tagline')}. Location: {company.get('location')}. Email: {company.get('email')}",
                "metadata": {"type": "company_info"}
            })
        
        # Add categories
        for cat in product_data.get("product_categories", []):
            documents.append({
                "content": f"Product Category: {cat.get('name')} - {cat.get('description')}",
                "metadata": {"type": "category"}
            })
        
        # Add industries
        for industry in product_data.get("industries_served", []):
            documents.append({
                "content": f"Industry Served: {industry.get('name')}",
                "metadata": {"type": "industry"}
            })
        
        print(f"Loaded {len(products)} products + categories + industries from jd_jones_products.json")
    
    print(f"Total documents: {len(documents)}")
    return documents


def generate_enhanced_intent_examples(documents):
    """Generate MORE intent classification training examples with higher quality."""
    examples = []
    
    # Extract product codes from documents
    product_codes = set()
    for doc in documents:
        content = doc.get("content", doc.get("text", str(doc)))
        codes = re.findall(r'NA\s*\d+|NJ\s*\d+', content, re.IGNORECASE)
        product_codes.update([c.upper().replace(" ", " ") for c in codes])
    
    product_codes = list(product_codes)[:100]  # Use more products
    if not product_codes:
        product_codes = [f"NA {700+i}" for i in range(50)] + [f"NJ {100+i}" for i in range(20)]
    
    print(f"Using {len(product_codes)} product codes for training data generation")
    
    # ENHANCED intent definitions with MANY more templates
    intent_patterns = {
        "product_inquiry": {
            "templates": [
                "What is {product}?",
                "Tell me about {product}",
                "I need information on {product}",
                "Do you have {product}?",
                "Details about {product}",
                "Show me {product}",
                "I want to know more about {product}",
                "Can you describe {product}?",
                "What are the features of {product}?",
                "Give me info on {product}",
                "I'm looking for {product}",
                "Do you sell {product}?",
                "Is {product} available?",
                "Tell me everything about {product}",
                "Information about {product} please",
                "What do you know about {product}?",
                "Describe {product} to me",
                "I need details on {product}",
                "What exactly is {product}?",
                "Can I get information about {product}?",
                "{product} information",
                "{product} details",
                "{product} data sheet",
                "Product info for {product}",
                "What's {product}?",
                "About {product}",
                "Looking for details on {product}",
                "Need info on {product}",
                "What materials go into {product}?",
            ],
        },
        "technical_question": {
            "templates": [
                "What is the temperature rating for {product}?",
                "What is the temperature limit of {product}?",
                "What temperature can {product} handle?",
                "Pressure limit of {product}?",
                "Maximum pressure for {product}?",
                "What is the pressure rating of {product}?",
                "Chemical compatibility of {product}",
                "What chemicals is {product} compatible with?",
                "Maximum operating temperature for {product}",
                "Can {product} handle 400°C?",
                "Can {product} handle 500°F?",
                "What chemicals can {product} resist?",
                "Chemical resistance of {product}",
                "Pressure rating for {product}",
                "Technical specifications of {product}",
                "Technical specs for {product}",
                "What are the specs of {product}?",
                "Operating temperature range of {product}",
                "Temperature range for {product}",
                "Thermal limits of {product}",
                "What PSI can {product} handle?",
                "Pressure capacity of {product}",
                "Max temp for {product}",
                "Min temp for {product}",
                "Cryogenic rating of {product}",
                "Can {product} be used at -100°C?",
                "High temperature performance of {product}",
                "Low temperature capability of {product}",
                "What is the tensile strength of {product}?",
                "Compression rating for {product}",
                "Recovery rate of {product}",
                "Creep relaxation of {product}",
                "Stress limits of {product}",
                "Physical properties of {product}",
                "Mechanical properties of {product}",
                "What is the hardness of {product}?",
                "Density of {product}",
            ],
        },
        "compliance_check": {
            "templates": [
                "Is {product} API 622 certified?",
                "Does {product} have API 622 certification?",
                "API 622 compliance for {product}",
                "{product} API 622 status",
                "Is {product} API 624 certified?",
                "Does {product} meet API 624?",
                "API 624 certification for {product}",
                "Does {product} meet FDA requirements?",
                "Is {product} FDA approved?",
                "FDA compliance for {product}",
                "Fire-safe certification for {product}",
                "Is {product} fire-safe certified?",
                "Fire safe status of {product}",
                "Is {product} ATEX approved?",
                "ATEX certification for {product}",
                "What certifications does {product} have?",
                "List certifications for {product}",
                "Certifications of {product}",
                "Shell SPE compliance for {product}",
                "Does {product} meet Saudi Aramco specs?",
                "Saudi Aramco approval for {product}",
                "Is {product} TA Luft compliant?",
                "TA Luft certification for {product}",
                "ISO certification for {product}",
                "ASME compliance for {product}",
                "Fugitive emissions compliance for {product}",
                "Does {product} meet emissions standards?",
                "Environmental certifications for {product}",
                "ABS certification for {product}",
                "DNV approval for {product}",
                "Lloyd's certification for {product}",
                "Nuclear grade certification for {product}",
                "NACE compliance for {product}",
                "Sour service certification for {product}",
                "Clean room compatible {product}?",
                "Pharmaceutical grade for {product}?",
            ],
        },
        "pricing_request": {
            "templates": [
                "How much does {product} cost?",
                "What is the price of {product}?",
                "Price for {product}",
                "Cost of {product}",
                "{product} price",
                "{product} cost",
                "What's the cost of {product}?",
                "Quote for {product}",
                "I need a quote for {product}",
                "Get me a quote for {product}",
                "Can I have a quote for {product}?",
                "Pricing for {product}",
                "{product} pricing",
                "What is {product} pricing?",
                "Quote for 100 units of {product}",
                "Bulk pricing for {product}",
                "Discount on {product}?",
                "What discounts are available for {product}?",
                "Volume discount for {product}",
                "How much for 1000 pieces of {product}?",
                "What's the unit price for {product}?",
                "Per unit cost of {product}",
                "I want to buy {product}, how much?",
                "Looking for pricing on {product}",
                "Send me a quote for {product}",
                "Need pricing info for {product}",
                "What will {product} cost me?",
                "Quotation for {product}",
                "Price list for {product}",
                "Give me a price on {product}",
            ],
        },
        "troubleshooting": {
            "templates": [
                "My {product} is leaking",
                "{product} is leaking",
                "Problem with {product}",
                "I have a problem with {product}",
                "{product} failed",
                "{product} failed after installation",
                "Why is {product} not sealing properly?",
                "{product} not sealing",
                "Troubleshoot {product} issue",
                "How to fix {product}?",
                "{product} degradation problem",
                "Seal failure with {product}",
                "Why did {product} fail?",
                "{product} is not working",
                "{product} broke",
                "My {product} is damaged",
                "Issue with {product}",
                "Having trouble with {product}",
                "{product} installation problem",
                "{product} is deteriorating",
                "What went wrong with {product}?",
                "{product} keeps failing",
                "{product} wears out too fast",
                "Premature failure of {product}",
                "Root cause of {product} failure",
                "Why is {product} degrading?",
                "{product} leak detection",
                "Fix leaking {product}",
                "Repair {product}",
                "How to troubleshoot {product}",
            ],
        },
        "application_guidance": {
            "templates": [
                "Which gasket for high temperature?",
                "What gasket for high temperature?",
                "Best gasket for high temperature",
                "Recommend a gasket for high temperature",
                "Which sealing material for {application}?",
                "Best sealing material for {application}?",
                "What product for cryogenic applications?",
                "Which product for cryogenic use?",
                "Recommend a gasket for steam service",
                "Best gasket for steam",
                "Suitable packing for {application}",
                "What should I use for {application}?",
                "Which product for {application}?",
                "What do you recommend for {application}?",
                "Best solution for {application}",
                "Suggest a product for {application}",
                "Need a gasket for corrosive fluids",
                "Which material for acidic applications?",
                "What to use for high pressure?",
                "Recommendation for extreme conditions",
                "Which product handles hydrogen?",
                "Best seal for oxygen service",
                "What for nuclear applications?",
                "Recommend for pharmaceutical use",
                "Which for food processing?",
                "Best for chemical applications",
                "What material for hot oil?",
                "Suggest for superheated steam",
                "Which packing for pumps?",
                "Best seal for compressors",
            ],
        },
        "order_status": {
            "templates": [
                "Where is my order?",
                "Order status",
                "Check my order",
                "Track my order",
                "Order status for #12345",
                "When will my order arrive?",
                "Delivery status",
                "Has my order shipped?",
                "Order tracking",
                "My order number is 12345",
                "Where's my shipment?",
                "Track shipment",
                "Delivery date for my order",
                "Expected delivery date",
                "When is delivery?",
                "Order not received",
                "Missing order",
                "Delayed order",
                "Order confirmation",
                "Did my order ship?",
                "Shipping status",
                "Track package",
                "Package status",
                "Order #12345 status",
                "What happened to my order?",
                "Check order #54321",
                "Status of my purchase",
                "When will I receive my order?",
                "Estimated arrival",
                "Order ETA",
            ],
        },
        "general": {
            "templates": [
                "Hello",
                "Hi there",
                "Hi",
                "Hey",
                "Good morning",
                "Good afternoon",
                "Contact information",
                "How do I contact you?",
                "Phone number",
                "Email address",
                "Who are you?",
                "What company is this?",
                "Help",
                "I need help",
                "Can you help me?",
                "Thank you",
                "Thanks",
                "Thanks a lot",
                "What can you do?",
                "What are your capabilities?",
                "Goodbye",
                "Bye",
                "See you later",
                "Ok",
                "Okay",
                "Sure",
                "Yes",
                "No",
                "Location",
                "Where are you located?",
            ],
        }
    }
    
    applications = [
        "valves", "pumps", "flanges", "heat exchangers", "reactors", 
        "pipelines", "compressors", "turbines", "boilers", "vessels",
        "agitators", "mixers", "pressure vessels", "storage tanks",
        "refineries", "chemical plants", "power plants", "offshore"
    ]
    
    # Generate examples for each intent
    print("\nGenerating examples for each intent:")
    for intent, config in intent_patterns.items():
        templates = config["templates"]
        count = 0
        
        for template in templates:
            # Generate with product codes
            if "{product}" in template:
                for product in product_codes:
                    text = template.replace("{product}", product)
                    if "{application}" in text:
                        for app in applications:
                            text2 = text.replace("{application}", app)
                            examples.append({"input": text2, "label": intent})
                            count += 1
                    else:
                        examples.append({"input": text, "label": intent})
                        count += 1
            elif "{application}" in template:
                for app in applications:
                    text = template.replace("{application}", app)
                    examples.append({"input": text, "label": intent})
                    count += 1
            else:
                examples.append({"input": template, "label": intent})
                count += 1
        
        print(f"  {intent}: {count} examples")
    
    print(f"\nTotal generated: {len(examples)} training examples")
    return examples


def train_high_confidence_classifier(examples):
    """Train sklearn classifier optimized for high confidence."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        import pickle
        import numpy as np
    except ImportError:
        print("Error: sklearn not installed. Run: pip install scikit-learn")
        return None
    
    # Prepare data
    texts = [ex["input"] for ex in examples]
    labels = [ex["label"] for ex in examples]
    
    print(f"\nTraining data statistics:")
    print(f"  Total examples: {len(examples)}")
    print(f"  Unique labels: {len(set(labels))}")
    for label in sorted(set(labels)):
        count = labels.count(label)
        print(f"    {label}: {count} ({100*count/len(labels):.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    print(f"\n  Training set: {len(X_train)}")
    print(f"  Test set: {len(X_test)}")
    
    # Use Calibrated SVM for better probability estimates
    # This typically gives higher confidence for correct predictions
    print("\nTraining CalibratedClassifierCV with LinearSVC...")
    
    # Create TF-IDF vectorizer with optimized parameters
    vectorizer = TfidfVectorizer(
        max_features=10000,      # More features
        ngram_range=(1, 3),      # Include trigrams
        min_df=1,                # Keep rare terms
        max_df=0.95,
        sublinear_tf=True,       # Apply sublinear tf scaling
        analyzer='word',
        strip_accents='unicode',
        lowercase=True
    )
    
    # Fit vectorizer
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Use CalibratedClassifierCV for well-calibrated probabilities
    base_clf = LinearSVC(C=1.0, max_iter=10000)
    clf = CalibratedClassifierCV(base_clf, cv=5, method='sigmoid')
    clf.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Check confidence scores
    y_proba = clf.predict_proba(X_test_tfidf)
    max_proba = np.max(y_proba, axis=1)
    
    print(f"\n{'='*60}")
    print(f"TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {np.mean(max_proba):.2%}")
    print(f"  Min confidence: {np.min(max_proba):.2%}")
    print(f"  Predictions >95%: {np.sum(max_proba > 0.95)/len(max_proba):.1%}")
    print(f"  Predictions >98%: {np.sum(max_proba > 0.98)/len(max_proba):.1%}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save full pipeline (vectorizer + classifier)
    model_dir = Path("models/slm")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a full pipeline for easier inference
    full_pipeline = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "classes": clf.classes_.tolist()
    }
    
    model_path = model_dir / f"intent_classifier_high_conf_{version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(full_pipeline, f)
    
    print(f"\nModel saved to: {model_path}")
    
    # Update registry
    registry_path = model_dir / "registry.json"
    registry = {"models": []}
    
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
    
    # Deactivate old models
    for model in registry.get("models", []):
        if model["slm_type"] == "intent_classifier":
            model["is_active"] = False
    
    # Add new model
    registry["models"].append({
        "slm_type": "intent_classifier",
        "model_name": "intent_classifier_high_conf",
        "version": version,
        "trained_at": datetime.now().isoformat(),
        "training_examples": len(examples),
        "accuracy": accuracy,
        "metrics": {
            "test_accuracy": accuracy,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "num_classes": len(set(labels)),
            "mean_confidence": float(np.mean(max_proba)),
            "pct_above_95": float(np.sum(max_proba > 0.95)/len(max_proba)),
            "pct_above_98": float(np.sum(max_proba > 0.98)/len(max_proba))
        },
        "model_path": str(model_path),
        "is_active": True
    })
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"Registry updated: {registry_path}")
    
    return full_pipeline, accuracy


def test_high_confidence(pipeline):
    """Test classifier with diverse queries."""
    import numpy as np
    
    vectorizer = pipeline["vectorizer"]
    clf = pipeline["classifier"]
    
    test_queries = [
        # Product inquiry
        ("What is NA 701?", "product_inquiry"),
        ("Tell me about NA 702", "product_inquiry"),
        ("Do you have NA 750?", "product_inquiry"),
        
        # Technical
        ("What is the temperature rating of NA 701?", "technical_question"),
        ("Pressure limit for NA 702?", "technical_question"),
        ("Chemical resistance of NA 750", "technical_question"),
        
        # Pricing
        ("How much does NA 701 cost?", "pricing_request"),
        ("Quote for NA 702", "pricing_request"),
        ("Price for 100 units of NA 750", "pricing_request"),
        
        # Compliance
        ("Is NA 701 API 622 certified?", "compliance_check"),
        ("FDA approval for NA 702?", "compliance_check"),
        ("Certifications for NA 750", "compliance_check"),
        
        # Troubleshooting
        ("NA 701 is leaking", "troubleshooting"),
        ("Problem with my NA 702", "troubleshooting"),
        ("Why did NA 750 fail?", "troubleshooting"),
        
        # Application
        ("Which gasket for high temperature?", "application_guidance"),
        ("What product for steam?", "application_guidance"),
        ("Best seal for pumps?", "application_guidance"),
        
        # Order
        ("Where is my order?", "order_status"),
        ("Track order #12345", "order_status"),
        ("Delivery status", "order_status"),
        
        # General
        ("Hello", "general"),
        ("Thank you", "general"),
        ("Contact info", "general"),
    ]
    
    print(f"\n{'='*60}")
    print(f"HIGH CONFIDENCE TEST PREDICTIONS")
    print(f"{'='*60}")
    
    correct = 0
    above_98 = 0
    
    for query, expected in test_queries:
        X = vectorizer.transform([query])
        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]
        conf = max(proba)
        
        is_correct = pred == expected
        correct += 1 if is_correct else 0
        above_98 += 1 if conf > 0.98 else 0
        
        status = "✓" if is_correct else "✗"
        conf_mark = "★" if conf > 0.98 else ""
        
        print(f"\n{status} Q: {query}")
        print(f"   Expected: {expected}")
        print(f"   Predicted: {pred} ({conf:.1%}) {conf_mark}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Correct: {correct}/{len(test_queries)} ({100*correct/len(test_queries):.1f}%)")
    print(f"Above 98% confidence: {above_98}/{len(test_queries)} ({100*above_98/len(test_queries):.1f}%)")


def main():
    print("="*60)
    print("   JD JONES RAG - HIGH CONFIDENCE SLM TRAINING")
    print("   Target: >98% confidence on predictions")
    print("="*60)
    print()
    
    # Step 1: Load company data
    print("Step 1: Loading company data...")
    documents = load_company_data()
    
    if not documents:
        print("No documents found. Please run data ingestion first.")
        return
    
    # Step 2: Generate MANY training examples
    print("\nStep 2: Generating enhanced training examples...")
    examples = generate_enhanced_intent_examples(documents)
    
    # Step 3: Train high-confidence classifier
    print("\nStep 3: Training high-confidence classifier...")
    result = train_high_confidence_classifier(examples)
    
    if result is None:
        return
    
    pipeline, accuracy = result
    
    # Step 4: Test classifier
    test_high_confidence(pipeline)
    
    print(f"\n{'='*60}")
    print("   TRAINING COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
