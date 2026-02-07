"""Direct test of trained SLM model."""
import pickle
import json
from pathlib import Path

print("=" * 60)
print("   SLM DIRECT TEST - Trained Intent Classifier")
print("=" * 60)

# Find the latest model
model_dir = Path("models/slm")
registry_path = model_dir / "registry.json"

if not registry_path.exists():
    print("ERROR: No models trained yet!")
    exit(1)

# Load registry
with open(registry_path, "r") as f:
    registry = json.load(f)

print(f"\nRegistered Models: {len(registry.get('models', []))}")

# Find active model
active_model = None
for model in registry.get("models", []):
    print(f"\n  Type: {model['slm_type']}")
    print(f"  Name: {model['model_name']}")
    print(f"  Version: {model['version']}")
    print(f"  Accuracy: {model['accuracy']:.1%}")
    print(f"  Trained on: {model['training_examples']} examples")
    print(f"  Active: {model['is_active']}")
    
    if model['is_active']:
        active_model = model

if active_model:
    model_path = Path(active_model['model_path'])
    
    if model_path.exists():
        print(f"\n  Loading model: {model_path}")
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)
        
        # Handle different model formats
        if isinstance(loaded, dict):
            # New format: vectorizer + classifier
            vectorizer = loaded.get("vectorizer")
            classifier = loaded.get("classifier")
        else:
            # Old format: sklearn pipeline
            vectorizer = None
            classifier = loaded
        
        # Test predictions
        test_queries = [
            "What is NA 701?",
            "How much does NA 702 cost?",
            "Is NA 750 API 622 certified?",
            "My gasket is leaking",
            "Which sealing product for steam?",
            "Temperature rating for NA 750",
            "Where is my order?",
            "Hello, can you help me?"
        ]
        
        print("\n" + "=" * 60)
        print("   TEST PREDICTIONS")
        print("=" * 60)
        
        for query in test_queries:
            if vectorizer:
                X = vectorizer.transform([query])
                pred = classifier.predict(X)[0]
                proba = classifier.predict_proba(X)[0]
            else:
                pred = classifier.predict([query])[0]
                proba = classifier.predict_proba([query])[0]
            
            confidence = max(proba)
            status = "[OK]" if confidence > 0.98 else "[  ]"
            print(f"\n  {status} Q: {query}")
            print(f"      -> {pred} ({confidence:.1%} confidence)")
    else:
        print(f"  ERROR: Model file not found: {model_path}")
else:
    print("\nERROR: No active model found!")

print("\n" + "=" * 60)
print("   TEST COMPLETE!")
print("=" * 60)
