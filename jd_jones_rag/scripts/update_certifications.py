"""Update products_structured.json with verified certifications from jdjones.com."""
import json

with open("data/products_structured.json", "r") as f:
    products = json.load(f)

# Verified certifications from jdjones.com product pages (Feb 2026)
VERIFIED_CERTS = {
    "NA 715": [
        "API 622 (3rd Edition)",
        "API 589 (Fire Safe)",
        "API 607 (Fire Safe)",
        "Low Emission (<100ppm)",
    ],
    "NA 719": [
        "API 589 (7th Edition)",
        "API 607 (7th Edition)",
        "ISO 10497:2010 (Fire Safe)",
    ],
    "NA 701": [
        "Shell SPE MESC 77/312 Class B",
    ],
    "NA B-3 + 707": [
        "API 622 (3rd Edition)",
        "API 589 (Fire Safe)",
        "API 607 (Fire Safe)",
        "API 624",
        "ISO 15848",
        "Low Emission (<100ppm)",
    ],
    "NA SP-1": [
        "API 622 (2nd & 3rd Edition + Annex C)",
        "API 589 (Fire Safe)",
        "API 607 (Fire Safe)",
        "ISO 15848 Part 1",
        "Low Emission (Ultra low, 12ppm)",
    ],
    "NA 740": ["Food Grade"],
    "NA 740 MA": ["Food Grade"],
    "NA 747": ["Food Grade"],
    "NA 781": ["FDA Approved", "Food Grade"],
    "NA 759": ["FDA Approved"],
    "NA 714": ["Food Grade"],
}

updated = 0
for p in products:
    code = p["code"]
    if code in VERIFIED_CERTS:
        existing = set(p.get("certifications", []))
        new_certs = VERIFIED_CERTS[code]
        merged = list(existing.union(new_certs))
        if set(merged) != existing:
            p["certifications"] = merged
            updated += 1
            print("Updated {}: {}".format(code, merged))
        else:
            print("Already up to date: {}".format(code))

with open("data/products_structured.json", "w") as f:
    json.dump(products, f, indent=2, ensure_ascii=False)

print("\nUpdated {} products".format(updated))
print("Total products: {}".format(len(products)))
