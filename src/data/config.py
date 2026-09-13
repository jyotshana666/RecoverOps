# Authoritative Class Mapping
CLASS_MAPPING = {
    "amount_due": 0,
    "date_due": 1,
    "document_id": 2,
    "date_issue": 3,
    "vendor_name": 4
}

REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
