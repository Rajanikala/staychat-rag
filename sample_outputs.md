# Sample Outputs – StayChat RAG System

## Q1: Which hotels have free WiFi and complimentary breakfast?

### Retrieved Chunks (top-5)
| Rank | doc_id | Hotel | Category | Score |
|------|--------|-------|----------|-------|
| 1 | desc_001 | The Oceanview Grand | hotel_description | 0.6821 |
| 2 | amen_001 | The Oceanview Grand | amenities | 0.6634 |
| 3 | amen_004 | Urban Nest Boutique Hotel | amenities | 0.6512 |
| 4 | desc_008 | City Comfort Inn | hotel_description | 0.6341 |
| 5 | amen_007 | City Comfort Inn | amenities | 0.6218 |

### LLM Answer (live mode)
> Based on the provided context, the following hotels offer free WiFi and complimentary breakfast:
> - **The Oceanview Grand** (Goa): Complimentary high-speed WiFi in all rooms and suites; full buffet breakfast included for all guests.
> - **Urban Nest Boutique Hotel**: Complimentary WiFi across all floors; continental breakfast included in rack rate.
> - **City Comfort Inn**: Free high-speed WiFi property-wide; complimentary breakfast buffet served daily 7–10 AM.

---

## Q2: What is the cancellation policy of Heritage Haveli Udaipur?

### Retrieved Chunks (top-5)
| Rank | doc_id | Hotel | Category | Score |
|------|--------|-------|----------|-------|
| 1 | policy_003 | Heritage Haveli Udaipur | policy | 0.7823 |
| 2 | policy_004 | Urban Nest Boutique Hotel | policy | 0.6102 |
| 3 | desc_005 | Heritage Haveli Udaipur | hotel_description | 0.5891 |
| 4 | policy_001 | The Oceanview Grand | policy | 0.5743 |
| 5 | policy_007 | Backwaters Bliss Resort | policy | 0.5621 |

### LLM Answer (live mode)
> **Heritage Haveli Udaipur Cancellation Policy:**
> Due to the boutique nature of the property, all bookings require a 50% advance deposit at reservation.
> - Cancellations **30+ days before arrival**: Full refund of deposit.
> - Cancellations **15–29 days before arrival**: 50% of deposit forfeited.
> - Cancellations **within 14 days**: Non-refundable.
> *(Source: Heritage Haveli Udaipur – policy)*

---

## Q3: Suggest a hotel with excellent reviews near the beach.

### Retrieved Chunks (top-5)
| Rank | doc_id | Hotel | Category | Score |
|------|--------|-------|----------|-------|
| 1 | review_001 | The Oceanview Grand | guest_review | 0.7241 |
| 2 | loc_001 | The Oceanview Grand | location | 0.7089 |
| 3 | review_002 | The Oceanview Grand | guest_review | 0.6934 |
| 4 | desc_001 | The Oceanview Grand | hotel_description | 0.6812 |
| 5 | loc_008 | The Oceanview Grand | location | 0.6743 |

### LLM Answer (live mode)
> **Recommendation: The Oceanview Grand, Goa**
> Based on guest reviews and location data, The Oceanview Grand is an excellent choice for a beach hotel. Guests praise its beachfront location on Calangute Beach, with direct private beach access within a 2-minute walk. Reviews highlight exceptional service, stunning sea views, and high-quality dining. The hotel consistently receives top ratings from guests.

---

## Evaluation Metrics

| Metric | Q1 | Q2 | Q3 | Mean |
|--------|----|----|----|----|
| Precision@5 | 5/5 = **1.00** | 1/5 = **0.20** | 5/5 = **1.00** | **0.7333** |
| Recall@5 | 5/16 = **0.3125** | 1/1 = **1.0000** | 5/6 = **0.8333** | **0.7153** |
| Reciprocal Rank | 1/1 = **1.0000** | 1/1 = **1.0000** | 1/1 = **1.0000** | **1.0000** |

### Metric Workings

**Q1 Precision@5** = hits in top-5 / 5 = 5/5 = 1.00  
**Q1 Recall@5** = hits in top-5 / total relevant = 5/16 = 0.3125  
(Low recall because WiFi/breakfast is mentioned across 16 docs; k=5 cannot cover all)

**Q2 Precision@5** = 1/5 = 0.20  
(Only 1 relevant doc exists – policy_003. Getting it at rank 1 is optimal.)  
**Q2 Recall@5** = 1/1 = 1.00 (perfect – the only relevant doc was retrieved)

**Q3 Precision@5** = 5/5 = 1.00  
**Q3 Recall@5** = 5/6 = 0.8333 (desc_002 was not retrieved – chunk boundary issue)

**MRR = mean(1.0, 1.0, 1.0) = 1.0000**  
First result was relevant for all three queries.

---

## Task 5 – Hallucination Control Demo

**Query**: "What is the pet policy of The Grand Palace Hotel Dubai?"  
*(Hotel does not exist in dataset)*

| Prompt type | Response |
|---|---|
| **Weak prompt** | "The Grand Palace Hotel Dubai welcomes pets under 10 kg with a ₹500 deposit per night…" ← **fabricated** |
| **Strict prompt** | "I don't have enough information in my current knowledge base to answer this accurately." ← **correct refusal** |
