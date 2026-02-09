"""Mine certifications from product features/descriptions text."""
import json

with open("data/products_structured.json", "r") as f:
    products = json.load(f)

cert_patterns = {
    "DIN 28091": ["din 28091"],
    "DIN 3535": ["din 3535"],
    "DIN 3754": ["din 3754"],
    "BS 7531": ["bs 7531"],
    "EN 1514": ["en 1514"],
    "ASTM F38": ["astm f38"],
    "API 589": ["api 589", "api589"],
    "API 622": ["api 622", "api622"],
    "API 607": ["api 607", "api607"],
    "Fire Safe": ["fire safe", "fire test", "fire tested"],
    "Low Emission": ["low emission", "fugitive emission"],
    "FDA": ["fda"],
    "Food Grade": ["food grade"],
    "ISO 10497": ["iso 10497"],
    "ISO 15848": ["iso 15848"],
    "Shell SPE": ["shell spe", "shell specification"],
    "TA-Luft": ["ta-luft", "ta luft"],
}

print("=== Certifications mined from text (not yet in explicit field) ===")
all_mined = {}
for p in products:
    combined = " ".join(
        p.get("features", []) + p.get("applications", []) + [p.get("description", "")]
    ).lower()
    found = {}
    for cert_name, keywords in cert_patterns.items():
        for kw in keywords:
            if kw in combined:
                found[cert_name] = True
                break
    explicit = set(c.lower() for c in p.get("certifications", []))
    mined = [c for c in found.keys() if c.lower() not in explicit]
    if mined:
        code = p["code"]
        all_mined[code] = mined
        print("  {}: explicit={}, mined_new={}".format(code, list(p.get("certifications", [])), mined))

print("\n=== Products with NO certifications (explicit or mined) ===")
no_certs = []
for p in products:
    code = p["code"]
    has_explicit = bool(p.get("certifications"))
    has_mined = code in all_mined
    if not has_explicit and not has_mined:
        no_certs.append(code)
print("  Count: {}".format(len(no_certs)))
print("  Products: {}".format(", ".join(no_certs[:20])))

print("\n=== Summary ===")
print("  Total products: {}".format(len(products)))
print("  With explicit certs: {}".format(sum(1 for p in products if p.get("certifications"))))
print("  With mined certs: {}".format(len(all_mined)))
print("  With no certs at all: {}".format(len(no_certs)))
